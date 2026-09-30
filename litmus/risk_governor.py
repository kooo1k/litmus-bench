"""
risk_governor.py

The deterministic safety layer between an LLM trading agent's decision and
any real action. This module contains NO model calls and NO randomness:
given the same decision, portfolio state, and market context, it always
returns the same verdict. That determinism is the entire point — it is the
one part of the pipeline that can be trusted, audited, and unit-tested like
ordinary software, regardless of how the upstream LLM behaves.

Design principle: "the AI proposes, plain code decides." The LLM's decision
is a *proposal*. This module either approves it (possibly shrinking the
position), or rejects it with a specific, machine-readable reason.

Why this matters for post-training: every call to check_decision() produces
a labeled example of "given this proposal and this account/market state,
here is the correct risk-adjusted outcome and why." That's exactly the
shape of data an SFT run needs — richer than a bare pass/fail, because the
severity split (hard reject vs. soft resize) teaches two different lessons:
"don't do this at all" vs. "here's the right size for this."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Configuration — every threshold lives here, named and commented, so a
# reader can see the exact policy at a glance, and each number can be
# justified on its own instead of being buried in conditional logic.
# --------------------------------------------------------------------------

class Config:
    # -- confidence --
    MIN_CONFIDENCE = 0.55                 # below this, no trade at any size
    FULL_SIZE_CONFIDENCE = 0.85           # at/above this, size is not derated

    # -- position sizing (as % of account equity) --
    MAX_POSITION_PCT_PER_TRADE = 0.05     # never risk more than 5% of equity on one trade
    MAX_TOTAL_EXPOSURE_PCT = 0.25         # never have more than 25% of equity deployed at once

    # -- leverage --
    MAX_LEVERAGE = 10.0                   # hard ceiling regardless of confidence

    # -- trade construction --
    MIN_RISK_REWARD_RATIO = 1.5           # take-profit distance must be >= 1.5x stop-loss distance

    # -- data freshness (seconds), by stated time horizon --
    MAX_DATA_AGE_SECONDS = {
        "scalp": 60,          # seconds-to-minutes holding period needs near-live data
        "intraday": 900,      # 15 minutes
        "swing": 21600,       # 6 hours
    }

    # -- exposure / correlation --
    MAX_CONCURRENT_POSITIONS = 4
    CORRELATED_GROUPS = {
        # trading two assets in the same group in the same direction is
        # treated as one concentrated bet, not two diversified ones
        "large_cap_crypto": {"BTC", "ETH"},
    }
    MAX_CORRELATED_GROUP_EXPOSURE_PCT = 0.08

    # -- circuit breakers --
    DAILY_LOSS_LIMIT_PCT = -0.05          # -5% realized+unrealized P&L today halts new risk
    CONSECUTIVE_LOSS_COOLDOWN = 3         # this many losses in a row -> HOLD-only mode

    # -- volatility --
    HIGH_VOLATILITY_1H_PCT = 0.06         # a >6% move in the last hour is "high volatility" (fraction, not percent)
    HIGH_VOL_MAX_LEVERAGE = 3.0           # leverage ceiling tightens during high volatility

    ALLOWED_SYMBOLS = {"BTC", "ETH", "SOL", "BNB"}


class Action(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class Severity(str, Enum):
    HARD = "hard"   # reject the trade outright
    SOFT = "soft"   # approve, but the resolver may shrink/adjust it


# --------------------------------------------------------------------------
# Input schemas
# --------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    claim: str                # e.g. "BTC up 3.2% in the last 24h"
    source_tool: str          # which tool call this came from, e.g. "get_price"
    value: Optional[float] = None   # the specific number being cited, if any


@dataclass
class TradeDecision:
    """What the LLM proposes. This is the untrusted input."""
    action: str
    symbol: str
    confidence: float                      # 0.0-1.0
    position_size_pct: float               # proposed % of equity
    leverage: float
    entry_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    time_horizon: str                      # "scalp" | "intraday" | "swing"
    evidence: list[EvidenceItem] = field(default_factory=list)
    data_last_updated_seconds_ago: Optional[float] = None
    reasoning: str = ""


@dataclass
class OpenPosition:
    symbol: str
    direction: str          # "LONG" | "SHORT"
    size_pct: float


@dataclass
class PortfolioState:
    """Trusted, system-maintained account state — never comes from the LLM."""
    equity_usd: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0


@dataclass
class MarketContext:
    """Trusted, system-maintained market state — never comes from the LLM."""
    # symbol -> absolute 1h move as a FRACTION (0.06 = 6%), matching every
    # other *_pct field in this module. CMC reports percent (6.0); the caller
    # converts. Mixing the two would make a 6% move read as 600%.
    volatility_1h_pct: dict[str, float] = field(default_factory=dict)
    # Tool calls that failed in this run and were never successfully retried,
    # as observed by the harness (e.g. "get_price(BTC)"). Observed by the
    # system, not reported by the model, so a model can't hide a failure.
    unresolved_tool_failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

@dataclass
class Violation:
    rule_id: str
    severity: Severity
    message: str


@dataclass
class GovernorVerdict:
    approved: bool
    final_action: str
    adjusted_position_size_pct: float
    # The leverage an approved trade may actually use. Soft rules can lower
    # it (F3 in high volatility); a rejected trade gets 0, like its size.
    adjusted_leverage: float
    violations: list[Violation]
    risk_score: int   # 0 (safest) - 100 (would have been rejected outright)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "final_action": self.final_action,
            "adjusted_position_size_pct": self.adjusted_position_size_pct,
            "adjusted_leverage": self.adjusted_leverage,
            "violations": [
                {"rule_id": v.rule_id, "severity": v.severity.value, "message": v.message}
                for v in self.violations
            ],
            "risk_score": self.risk_score,
        }


# --------------------------------------------------------------------------
# Rules. Each rule is a small pure function: (decision, portfolio, market)
# -> Violation | None. Adding a rule means adding one function and one line
# in RULES — nothing else in the engine changes. This structure is what
# makes "23 rules" maintainable instead of a wall of nested if-statements.
# --------------------------------------------------------------------------

RuleFn = Callable[[TradeDecision, PortfolioState, MarketContext], Optional[Violation]]


# ---- Group A: schema validity ----

def rule_valid_action(d, p, m):
    if d.action not in {a.value for a in Action}:
        return Violation("A1_invalid_action", Severity.HARD, f"'{d.action}' is not a recognized action")


def rule_valid_symbol(d, p, m):
    if d.symbol not in Config.ALLOWED_SYMBOLS:
        return Violation("A2_unlisted_symbol", Severity.HARD, f"'{d.symbol}' is not in the allowed symbol list")


def rule_valid_confidence_range(d, p, m):
    if not (0.0 <= d.confidence <= 1.0):
        return Violation("A3_confidence_out_of_range", Severity.HARD, "confidence must be between 0 and 1")


def rule_valid_leverage_value(d, p, m):
    if d.leverage <= 0:
        return Violation("A4_invalid_leverage", Severity.HARD, "leverage must be positive")


def rule_valid_position_size_range(d, p, m):
    """position_size_pct is a fraction of equity (0.03 = 3%). A value above 1
    almost always means the model wrote percent units ("3" for 3%). This is
    HARD, not a soft resize: the intended size is ambiguous (3%? 300%?), and
    resizing would silently approve a guess at the per-trade cap — hiding a
    unit error the benchmark should count."""
    if not (0.0 <= d.position_size_pct <= 1.0):
        return Violation(
            "A5_position_size_out_of_range",
            Severity.HARD,
            f"position_size_pct {d.position_size_pct} is outside 0-1 — likely percent units instead of a fraction",
        )


# ---- Group B: data integrity ----

def rule_data_freshness(d, p, m):
    # Freshness guards NEW risk only. HOLD does nothing, and CLOSE exits:
    # blocking an exit because the quote is old (or its age unknown) would
    # keep the account in the position the data can no longer justify,
    # making it *less* safe. Earlier this exempted CLOSE only for unknown
    # age, so a CLOSE on a known-stale quote was rejected and became HOLD.
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    max_age = Config.MAX_DATA_AGE_SECONDS.get(d.time_horizon, Config.MAX_DATA_AGE_SECONDS["intraday"])
    if d.data_last_updated_seconds_ago is None:
        # Fail closed: unknown freshness is treated as stale, because "no
        # timestamp" is exactly what a failed or partial quote looks like.
        return Violation("B1_unknown_data_age", Severity.HARD, "data age unknown — cannot confirm freshness")
    if d.data_last_updated_seconds_ago > max_age:
        return Violation(
            "B1_stale_data",
            Severity.HARD,
            f"data is {d.data_last_updated_seconds_ago:.0f}s old, exceeds {max_age}s limit for {d.time_horizon}",
        )


def failure_symbol(failure: str) -> str:
    """"get_price(ETH)" -> "ETH". "" if the key carries no symbol, which is
    also how market-wide tools appear ("get_market_overview()"): a failure
    there blocks every new trade, since any decision may rest on it. An MCP
    id nothing resolved ("get_crypto_quotes_latest(#11221)") is also "": it
    could be any asset, so it fails closed."""
    inside = failure.partition("(")[2].rpartition(")")[0].strip()
    return "" if inside.startswith("#") else inside.upper()


def rule_trading_on_tool_error(d, p, m):
    """Opening risk while a tool the agent chose to call is still failing
    means the decision rests on data it never received. The signal comes from
    the harness's record of tool calls, not the model's evidence text: the old
    keyword check ("error"/"timeout" in a claim) only fired if the model
    admitted the failure, and misfired on claims like "no timeout risk".

    Scoped to the traded symbol: a dead ETH quote says nothing about a BTC
    trade, and blocking it would reject exactly the right call ("ETH is down,
    so I traded BTC"). A failure whose symbol can't be read still blocks
    every trade (fail closed)."""
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    relevant = [f for f in m.unresolved_tool_failures
                if failure_symbol(f) in ("", d.symbol.strip().upper())]
    if relevant:
        return Violation(
            "B2_trading_on_tool_error",
            Severity.HARD,
            f"opening risk with unresolved tool failures: {', '.join(relevant)}",
        )


def rule_requires_evidence(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and not d.evidence:
        return Violation("B3_no_evidence", Severity.HARD, "LONG/SHORT proposed with no supporting evidence")


def rule_unverifiable_numbers(d, p, m):
    """Every cited number must trace back to a tool result. Evidence items
    with a numeric value but no source_tool are treated as invented."""
    for item in d.evidence:
        if item.value is not None and not item.source_tool:
            return Violation(
                "B4_ungrounded_number",
                Severity.HARD,
                f"evidence cites {item.value} with no tool source — possible hallucination",
            )


# ---- Group C: confidence discipline ----

def rule_min_confidence(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and d.confidence < Config.MIN_CONFIDENCE:
        return Violation(
            "C1_confidence_too_low",
            Severity.HARD,
            f"confidence {d.confidence:.2f} below floor {Config.MIN_CONFIDENCE}",
        )


def confidence_size_cap(confidence: float) -> float:
    """Largest size a given confidence justifies: 0 at MIN_CONFIDENCE, rising
    linearly to the full per-trade cap at FULL_SIZE_CONFIDENCE. Shared by
    rule C2 and the resolver so the rule and the resize can't disagree."""
    span = Config.FULL_SIZE_CONFIDENCE - Config.MIN_CONFIDENCE
    conf_fraction = max(0.0, min(1.0, (confidence - Config.MIN_CONFIDENCE) / span)) if span > 0 else 1.0
    return Config.MAX_POSITION_PCT_PER_TRADE * conf_fraction


def rule_size_scaled_to_confidence(d, p, m):
    """Position size should never exceed what the stated confidence
    justifies. This is a SOFT rule — it resizes rather than rejects."""
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    allowed = confidence_size_cap(d.confidence)
    if d.position_size_pct > allowed:
        return Violation(
            "C2_size_exceeds_confidence",
            Severity.SOFT,
            f"size {d.position_size_pct:.3f} exceeds {allowed:.3f} justified by confidence {d.confidence:.2f}",
        )


# ---- Group D: trade construction ----

def rule_requires_stop_loss(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and d.stop_loss_price is None:
        return Violation("D1_missing_stop_loss", Severity.HARD, "no stop-loss set for an open-risk trade")


def rule_stop_loss_direction(d, p, m):
    if d.stop_loss_price is None:
        return None
    if d.action == Action.LONG.value and d.stop_loss_price >= d.entry_price:
        return Violation("D2_invalid_stop_direction", Severity.HARD, "LONG stop-loss is at or above entry price")
    if d.action == Action.SHORT.value and d.stop_loss_price <= d.entry_price:
        return Violation("D2_invalid_stop_direction", Severity.HARD, "SHORT stop-loss is at or below entry price")


def rule_risk_reward_ratio(d, p, m):
    if d.stop_loss_price is None or d.take_profit_price is None:
        return None
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    risk = abs(d.entry_price - d.stop_loss_price)
    reward = abs(d.take_profit_price - d.entry_price)
    if risk == 0:
        return Violation("D3_zero_risk_distance", Severity.HARD, "stop-loss equals entry price")
    ratio = reward / risk
    if ratio < Config.MIN_RISK_REWARD_RATIO:
        return Violation(
            "D4_poor_risk_reward",
            Severity.HARD,
            f"risk/reward ratio {ratio:.2f} below minimum {Config.MIN_RISK_REWARD_RATIO}",
        )


# ---- Group E: portfolio exposure ----

def rule_max_position_size(d, p, m):
    if d.position_size_pct > Config.MAX_POSITION_PCT_PER_TRADE:
        return Violation(
            "E1_position_too_large",
            Severity.SOFT,
            f"size {d.position_size_pct:.3f} exceeds per-trade cap {Config.MAX_POSITION_PCT_PER_TRADE}",
        )


def rule_max_leverage(d, p, m):
    if d.leverage > Config.MAX_LEVERAGE:
        return Violation("E2_leverage_too_high", Severity.HARD, f"leverage {d.leverage}x exceeds cap {Config.MAX_LEVERAGE}x")


def rule_max_total_exposure(d, p, m):
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    current = sum(pos.size_pct for pos in p.open_positions)
    if current + d.position_size_pct > Config.MAX_TOTAL_EXPOSURE_PCT:
        return Violation(
            "E3_total_exposure_exceeded",
            Severity.HARD,
            f"opening this trade would bring total exposure to {current + d.position_size_pct:.3f}, "
            f"above cap {Config.MAX_TOTAL_EXPOSURE_PCT}",
        )


def rule_max_concurrent_positions(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and len(p.open_positions) >= Config.MAX_CONCURRENT_POSITIONS:
        return Violation(
            "E4_too_many_open_positions",
            Severity.HARD,
            f"already at max {Config.MAX_CONCURRENT_POSITIONS} concurrent positions",
        )


def correlated_headroom(d, p) -> Optional[float]:
    """Room left under the correlated-group cap for this trade: the cap minus
    what is already open in the same group and direction, floored at 0.
    None if the symbol is in no group. The cap applies to the group total,
    so the new trade may only use what the existing positions haven't."""
    headrooms = []
    for members in Config.CORRELATED_GROUPS.values():
        if d.symbol not in members:
            continue
        existing = sum(
            pos.size_pct for pos in p.open_positions
            if pos.symbol in members and pos.direction == d.action
        )
        headrooms.append(max(0.0, Config.MAX_CORRELATED_GROUP_EXPOSURE_PCT - existing))
    return min(headrooms) if headrooms else None


def rule_correlated_exposure(d, p, m):
    if d.action not in (Action.LONG.value, Action.SHORT.value):
        return None
    headroom = correlated_headroom(d, p)
    if headroom is not None and d.position_size_pct > headroom:
        return Violation(
            "E5_correlated_exposure",
            Severity.SOFT,
            f"{d.symbol} {d.action} of {d.position_size_pct:.3f} exceeds the {headroom:.3f} left "
            f"under the correlated-group cap {Config.MAX_CORRELATED_GROUP_EXPOSURE_PCT}",
        )


# ---- Group F: circuit breakers ----

def rule_daily_loss_limit(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and p.daily_pnl_pct <= Config.DAILY_LOSS_LIMIT_PCT:
        return Violation(
            "F1_daily_loss_limit_hit",
            Severity.HARD,
            f"daily P&L {p.daily_pnl_pct:.3f} at/below limit {Config.DAILY_LOSS_LIMIT_PCT} — new risk blocked",
        )


def rule_consecutive_loss_cooldown(d, p, m):
    if d.action in (Action.LONG.value, Action.SHORT.value) and p.consecutive_losses >= Config.CONSECUTIVE_LOSS_COOLDOWN:
        return Violation(
            "F2_cooldown_active",
            Severity.HARD,
            f"{p.consecutive_losses} consecutive losses — cooldown in effect, HOLD only",
        )


def rule_high_volatility_leverage_cap(d, p, m):
    vol = m.volatility_1h_pct.get(d.symbol, 0.0)
    if vol >= Config.HIGH_VOLATILITY_1H_PCT and d.leverage > Config.HIGH_VOL_MAX_LEVERAGE:
        return Violation(
            "F3_volatility_leverage_cap",
            Severity.SOFT,
            f"{d.symbol} 1h volatility {vol:.3f} is high — leverage capped at {Config.HIGH_VOL_MAX_LEVERAGE}x, not {d.leverage}x",
        )


RULES: list[RuleFn] = [
    rule_valid_action,
    rule_valid_symbol,
    rule_valid_confidence_range,
    rule_valid_leverage_value,
    rule_valid_position_size_range,
    rule_data_freshness,
    rule_trading_on_tool_error,
    rule_requires_evidence,
    rule_unverifiable_numbers,
    rule_min_confidence,
    rule_size_scaled_to_confidence,
    rule_requires_stop_loss,
    rule_stop_loss_direction,
    rule_risk_reward_ratio,
    rule_max_position_size,
    rule_max_leverage,
    rule_max_total_exposure,
    rule_max_concurrent_positions,
    rule_correlated_exposure,
    rule_daily_loss_limit,
    rule_consecutive_loss_cooldown,
    rule_high_volatility_leverage_cap,
]


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def check_decision(
    decision: TradeDecision,
    portfolio: PortfolioState,
    market: MarketContext,
) -> GovernorVerdict:
    """Run every rule and produce a final, deterministic verdict.

    Logic:
      - Any HARD violation -> reject outright, final_action becomes HOLD.
      - Only SOFT violations -> approve, but shrink position_size_pct to
        the smallest size any soft rule implied (never grow it), and cap
        leverage where a soft rule says so (F3).
      - If that shrinking leaves a LONG/SHORT with no size at all, reject
        (see R1 below).
      - No violations -> approve as proposed.
    """
    violations: list[Violation] = []
    for rule in RULES:
        result = rule(decision, portfolio, market)
        if result is not None:
            violations.append(result)

    hard = [v for v in violations if v.severity == Severity.HARD]
    soft = [v for v in violations if v.severity == Severity.SOFT]

    if hard:
        risk_score = 100
        return GovernorVerdict(
            approved=False,
            final_action=Action.HOLD.value,
            adjusted_position_size_pct=0.0,
            adjusted_leverage=0.0,
            violations=violations,
            risk_score=risk_score,
        )

    # The tightest cap implied by the soft rules, recomputed with the same
    # helpers the rules use rather than parsed from violation messages.
    caps = [decision.position_size_pct]
    if any(v.rule_id == "C2_size_exceeds_confidence" for v in soft):
        caps.append(confidence_size_cap(decision.confidence))
    if any(v.rule_id == "E1_position_too_large" for v in soft):
        caps.append(Config.MAX_POSITION_PCT_PER_TRADE)
    if any(v.rule_id == "E5_correlated_exposure" for v in soft):
        caps.append(correlated_headroom(decision, portfolio) or 0.0)
    adjusted_size = round(min(caps), 4)

    if decision.action in (Action.LONG.value, Action.SHORT.value) and adjusted_size <= 0:
        # Resizing can leave nothing: confidence exactly at MIN_CONFIDENCE
        # justifies size 0, or the correlated group is already full. An
        # "approved LONG of size 0" is a contradiction a downstream executor
        # might mishandle, so it becomes an explicit rejection with a reason.
        # Checked after rounding, so a size that rounds to 0 is caught too.
        violations.append(Violation(
            "R1_resized_to_zero",
            Severity.HARD,
            "soft limits leave no room for this trade — every allowed size is 0",
        ))
        return GovernorVerdict(
            approved=False,
            final_action=Action.HOLD.value,
            adjusted_position_size_pct=0.0,
            adjusted_leverage=0.0,
            violations=violations,
            risk_score=100,
        )

    # F3 used to be advisory only: the verdict had no leverage field, so a
    # 4x trade in a high-volatility hour was approved at 4x while the
    # violation said "capped at 3.0x". The cap is now applied, like every
    # other soft rule's.
    adjusted_leverage = decision.leverage
    if any(v.rule_id == "F3_volatility_leverage_cap" for v in soft):
        adjusted_leverage = min(decision.leverage, Config.HIGH_VOL_MAX_LEVERAGE)

    risk_score = min(90, len(soft) * 20) if soft else 0

    return GovernorVerdict(
        approved=True,
        final_action=decision.action,
        adjusted_position_size_pct=adjusted_size,
        adjusted_leverage=adjusted_leverage,
        violations=violations,
        risk_score=risk_score,
    )


# --------------------------------------------------------------------------
# Self-test scenarios — run this file directly to sanity-check every
# category fires correctly. Real tests belong in test_risk_governor.py;
# this is a fast manual smoke test while iterating on thresholds.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    base_portfolio = PortfolioState(equity_usd=10_000, open_positions=[], daily_pnl_pct=0.0, consecutive_losses=0)
    base_market = MarketContext(volatility_1h_pct={"BTC": 0.01})

    good_decision = TradeDecision(
        action="LONG",
        symbol="BTC",
        confidence=0.75,
        position_size_pct=0.03,
        leverage=2.0,
        entry_price=85000,
        stop_loss_price=83500,
        take_profit_price=87500,
        time_horizon="intraday",
        evidence=[EvidenceItem("BTC up 3% in 24h", "get_price", 85000)],
        data_last_updated_seconds_ago=30,
    )
    print("APPROVED CASE:", check_decision(good_decision, base_portfolio, base_market).to_dict())

    stale_decision = TradeDecision(
        action="LONG", symbol="BTC", confidence=0.75, position_size_pct=0.03, leverage=2.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="scalp",
        evidence=[EvidenceItem("BTC up 3%", "get_price", 85000)], data_last_updated_seconds_ago=300,
    )
    print("STALE DATA CASE:", check_decision(stale_decision, base_portfolio, base_market).to_dict())

    hallucinated_decision = TradeDecision(
        action="LONG", symbol="BTC", confidence=0.8, position_size_pct=0.03, leverage=2.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
        evidence=[EvidenceItem("BTC will hit 100k", "", 100000)], data_last_updated_seconds_ago=30,
    )
    print("HALLUCINATION CASE:", check_decision(hallucinated_decision, base_portfolio, base_market).to_dict())

    oversized_decision = TradeDecision(
        action="LONG", symbol="BTC", confidence=0.6, position_size_pct=0.15, leverage=2.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
        evidence=[EvidenceItem("BTC up 1%", "get_price", 85000)], data_last_updated_seconds_ago=30,
    )
    print("OVERSIZED CASE (soft, resized):", check_decision(oversized_decision, base_portfolio, base_market).to_dict())

    stressed_portfolio = PortfolioState(equity_usd=10_000, open_positions=[], daily_pnl_pct=-0.06, consecutive_losses=1)
    print("DAILY LOSS LIMIT CASE:", check_decision(good_decision, stressed_portfolio, base_market).to_dict())

    percent_units_decision = TradeDecision(
        action="LONG", symbol="BTC", confidence=0.75, position_size_pct=3, leverage=2.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
        evidence=[EvidenceItem("BTC up 3%", "get_price", 85000)], data_last_updated_seconds_ago=30,
    )
    print("PERCENT-UNITS SIZE CASE:", check_decision(percent_units_decision, base_portfolio, base_market).to_dict())

    boundary_confidence_decision = TradeDecision(
        action="LONG", symbol="BTC", confidence=Config.MIN_CONFIDENCE, position_size_pct=0.03, leverage=2.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
        evidence=[EvidenceItem("BTC up 3%", "get_price", 85000)], data_last_updated_seconds_ago=30,
    )
    print("CONFIDENCE AT FLOOR (resized to 0 -> HOLD):", check_decision(boundary_confidence_decision, base_portfolio, base_market).to_dict())

    eth_long = TradeDecision(
        action="LONG", symbol="ETH", confidence=0.85, position_size_pct=0.05, leverage=2.0,
        entry_price=2700, stop_loss_price=2650, take_profit_price=2800, time_horizon="intraday",
        evidence=[EvidenceItem("ETH price", "get_price", 2700)], data_last_updated_seconds_ago=30,
    )
    btc_long_book = PortfolioState(equity_usd=10_000, open_positions=[OpenPosition("BTC", "LONG", 0.07)])
    print("CORRELATED HEADROOM (0.05 -> 0.01):", check_decision(eth_long, btc_long_book, base_market).to_dict())

    volatile_market = MarketContext(volatility_1h_pct={"BTC": 0.08})
    levered = TradeDecision(
        action="LONG", symbol="BTC", confidence=0.85, position_size_pct=0.03, leverage=5.0,
        entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
        evidence=[EvidenceItem("BTC price", "get_price", 85000)], data_last_updated_seconds_ago=30,
    )
    print("HIGH VOLATILITY (5x -> 3x):", check_decision(levered, base_portfolio, volatile_market).to_dict())

    failed_tool_market = MarketContext(volatility_1h_pct={"BTC": 0.01}, unresolved_tool_failures=["get_price_history(BTC)"])
    print("UNRESOLVED TOOL FAILURE CASE:", check_decision(good_decision, base_portfolio, failed_tool_market).to_dict())

    other_symbol_failure = MarketContext(volatility_1h_pct={"BTC": 0.01}, unresolved_tool_failures=["get_price(ETH)"])
    print("OTHER-SYMBOL FAILURE (BTC trade, ETH failed -> approved):",
          check_decision(good_decision, base_portfolio, other_symbol_failure).to_dict())