"""risk_governor.py: one test per rule (fires, with the right severity, and
doesn't fire at its boundary), plus the engine's verdict logic. Pure
functions; no I/O."""

import re
from dataclasses import replace
from pathlib import Path

import pytest

from litmus import risk_governor as rg
from litmus.risk_governor import (Config, EvidenceItem, MarketContext, OpenPosition, PortfolioState,
                           TradeDecision, check_decision)

# A trade that passes every rule: confidence at the full-size threshold, 3%
# size, 2x, stop 1500 below, target 2500 above (R:R 1.67), 30s-old data.
BASE = TradeDecision(
    action="LONG", symbol="BTC", confidence=0.85, position_size_pct=0.03, leverage=2.0,
    entry_price=85000, stop_loss_price=83500, take_profit_price=87500, time_horizon="intraday",
    evidence=[EvidenceItem("BTC price", "get_price", 85000)], data_last_updated_seconds_ago=30)
EMPTY_BOOK = PortfolioState(equity_usd=10_000)
CALM = MarketContext(volatility_1h_pct={"BTC": 0.01})


def verdict(decision=BASE, portfolio=EMPTY_BOOK, market=CALM):
    return check_decision(decision, portfolio, market)


def rule_ids(v):
    return [x.rule_id for x in v.violations]


def severity(v, rule_id):
    return next(x.severity for x in v.violations if x.rule_id == rule_id)


def test_base_trade_passes_every_rule():
    v = verdict()
    assert v.approved and v.violations == [] and v.final_action == "LONG"
    assert v.adjusted_position_size_pct == 0.03 and v.adjusted_leverage == 2.0 and v.risk_score == 0


# --------------------------------------------------------------------------
# Every rule fires, with its severity. (rule_id, decision change,
# portfolio, market, severity)
# --------------------------------------------------------------------------

def d(**changes):
    return replace(BASE, **changes)


LOSING_BOOK = PortfolioState(equity_usd=10_000, daily_pnl_pct=-0.05)
COOLDOWN_BOOK = PortfolioState(equity_usd=10_000, consecutive_losses=3)
FULL_BOOK = PortfolioState(equity_usd=10_000, open_positions=[OpenPosition(s, "LONG", 0.01) for s in ("SOL", "BNB", "SOL", "BNB")])
EXPOSED_BOOK = PortfolioState(equity_usd=10_000, open_positions=[OpenPosition("SOL", "LONG", 0.23)])
BTC_ETH_BOOK = PortfolioState(equity_usd=10_000, open_positions=[OpenPosition("ETH", "LONG", 0.07)])
VOLATILE = MarketContext(volatility_1h_pct={"BTC": 0.06})
BTC_FAILED = MarketContext(volatility_1h_pct={"BTC": 0.01}, unresolved_tool_failures=["get_price(BTC)"])

HARD, SOFT = rg.Severity.HARD, rg.Severity.SOFT

CASES = [
    ("A1_invalid_action", d(action="BUY"), EMPTY_BOOK, CALM, HARD),
    ("A2_unlisted_symbol", d(symbol="DOGE"), EMPTY_BOOK, CALM, HARD),
    ("A2_unlisted_symbol", d(symbol="btc"), EMPTY_BOOK, CALM, HARD),  # not normalized, on purpose
    ("A3_confidence_out_of_range", d(confidence=1.2), EMPTY_BOOK, CALM, HARD),
    ("A4_invalid_leverage", d(leverage=0), EMPTY_BOOK, CALM, HARD),
    ("A5_position_size_out_of_range", d(position_size_pct=3), EMPTY_BOOK, CALM, HARD),
    ("B1_stale_data", d(data_last_updated_seconds_ago=901), EMPTY_BOOK, CALM, HARD),
    ("B1_unknown_data_age", d(data_last_updated_seconds_ago=None), EMPTY_BOOK, CALM, HARD),
    ("B2_trading_on_tool_error", BASE, EMPTY_BOOK, BTC_FAILED, HARD),
    ("B3_no_evidence", d(evidence=[]), EMPTY_BOOK, CALM, HARD),
    ("B4_ungrounded_number", d(evidence=[EvidenceItem("BTC to 100k", "", 100000)]), EMPTY_BOOK, CALM, HARD),
    ("C1_confidence_too_low", d(confidence=0.5), EMPTY_BOOK, CALM, HARD),
    ("C2_size_exceeds_confidence", d(confidence=0.7), EMPTY_BOOK, CALM, SOFT),
    ("D1_missing_stop_loss", d(stop_loss_price=None), EMPTY_BOOK, CALM, HARD),
    ("D2_invalid_stop_direction", d(stop_loss_price=86000), EMPTY_BOOK, CALM, HARD),
    ("D2_invalid_stop_direction", d(action="SHORT", stop_loss_price=84000, take_profit_price=80000), EMPTY_BOOK, CALM, HARD),
    ("D3_zero_risk_distance", d(stop_loss_price=85000), EMPTY_BOOK, CALM, HARD),
    ("D4_poor_risk_reward", d(take_profit_price=86000), EMPTY_BOOK, CALM, HARD),
    ("E1_position_too_large", d(position_size_pct=0.06), EMPTY_BOOK, CALM, SOFT),
    ("E2_leverage_too_high", d(leverage=11), EMPTY_BOOK, CALM, HARD),
    ("E3_total_exposure_exceeded", BASE, EXPOSED_BOOK, CALM, HARD),
    ("E4_too_many_open_positions", BASE, FULL_BOOK, CALM, HARD),
    ("E5_correlated_exposure", BASE, BTC_ETH_BOOK, CALM, SOFT),
    ("F1_daily_loss_limit_hit", BASE, LOSING_BOOK, CALM, HARD),
    ("F2_cooldown_active", BASE, COOLDOWN_BOOK, CALM, HARD),
    ("F3_volatility_leverage_cap", d(leverage=4), EMPTY_BOOK, VOLATILE, SOFT),
    ("R1_resized_to_zero", d(confidence=Config.MIN_CONFIDENCE), EMPTY_BOOK, CALM, HARD),
]


@pytest.mark.parametrize("rule_id,decision,portfolio,market,sev", CASES,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_rule_fires(rule_id, decision, portfolio, market, sev):
    v = verdict(decision, portfolio, market)
    assert rule_id in rule_ids(v)
    assert severity(v, rule_id) == sev
    if sev == HARD:
        assert not v.approved and v.final_action == "HOLD" and v.adjusted_position_size_pct == 0.0
        assert v.adjusted_leverage == 0.0 and v.risk_score == 100


def test_every_rule_in_the_source_has_a_firing_test():
    """If a new rule_id is added to risk_governor.py, this fails until it
    gets a case above."""
    source = Path(rg.__file__).read_text(encoding="utf-8")
    defined = set(re.findall(r'"([A-Z]\d_[a-z_]+)"', source))
    assert defined == {c[0] for c in CASES}


# --------------------------------------------------------------------------
# Boundaries: each threshold is inclusive on the allowed side
# --------------------------------------------------------------------------

@pytest.mark.parametrize("decision,portfolio,absent", [
    (d(data_last_updated_seconds_ago=900), EMPTY_BOOK, "B1_stale_data"),              # intraday limit exactly
    (d(time_horizon="scalp", data_last_updated_seconds_ago=60), EMPTY_BOOK, "B1_stale_data"),
    (d(time_horizon="swing", data_last_updated_seconds_ago=21600), EMPTY_BOOK, "B1_stale_data"),
    (d(confidence=Config.MIN_CONFIDENCE), EMPTY_BOOK, "C1_confidence_too_low"),        # floor is allowed (R1 then fires)
    (d(take_profit_price=85000 + 1.5 * 1500), EMPTY_BOOK, "D4_poor_risk_reward"),        # R:R exactly 1.5
    (d(position_size_pct=0.05), EMPTY_BOOK, "E1_position_too_large"),                  # per-trade cap exactly
    (d(leverage=10), EMPTY_BOOK, "E2_leverage_too_high"),                              # leverage cap exactly
    (BASE, PortfolioState(10_000, [OpenPosition("SOL", "LONG", 0.22)]), "E3_total_exposure_exceeded"),  # 0.25 exactly
    (BASE, PortfolioState(10_000, [OpenPosition("SOL", "LONG", 0.01)] * 3), "E4_too_many_open_positions"),
    (BASE, PortfolioState(10_000, [OpenPosition("ETH", "LONG", 0.05)]), "E5_correlated_exposure"),  # headroom 0.03 exactly
    (BASE, PortfolioState(10_000, daily_pnl_pct=-0.049), "F1_daily_loss_limit_hit"),
    (BASE, PortfolioState(10_000, consecutive_losses=2), "F2_cooldown_active"),
])
def test_rule_does_not_fire_at_its_boundary(decision, portfolio, absent):
    assert absent not in rule_ids(verdict(decision, portfolio))


def test_confidence_scaled_size_cap_is_linear_between_floor_and_full():
    assert rg.confidence_size_cap(Config.MIN_CONFIDENCE) == 0
    assert rg.confidence_size_cap(Config.FULL_SIZE_CONFIDENCE) == Config.MAX_POSITION_PCT_PER_TRADE
    mid = (Config.MIN_CONFIDENCE + Config.FULL_SIZE_CONFIDENCE) / 2
    assert rg.confidence_size_cap(mid) == pytest.approx(Config.MAX_POSITION_PCT_PER_TRADE / 2)
    assert rg.confidence_size_cap(1.0) == Config.MAX_POSITION_PCT_PER_TRADE  # clamped


def test_high_volatility_threshold_is_inclusive():
    assert "F3_volatility_leverage_cap" in rule_ids(verdict(d(leverage=4), market=VOLATILE))
    assert "F3_volatility_leverage_cap" not in rule_ids(verdict(d(leverage=3), market=VOLATILE))
    assert "F3_volatility_leverage_cap" not in rule_ids(
        verdict(d(leverage=4), market=MarketContext(volatility_1h_pct={"BTC": 0.059})))


# --------------------------------------------------------------------------
# Rule-specific behavior
# --------------------------------------------------------------------------

def test_hold_is_never_blocked_by_data_freshness():
    for age in (None, 10**6):
        assert verdict(d(action="HOLD", data_last_updated_seconds_ago=age)).approved


@pytest.mark.parametrize("age", [None, 10**6], ids=["unknown-age", "known-stale"])
def test_close_is_never_blocked_by_data_freshness(age):
    # An exit must not be blocked because the data is old: that would keep the
    # position the data can no longer justify. (Previously only unknown age
    # was exempt; a known-stale CLOSE was rejected and turned into HOLD.)
    v = verdict(d(action="CLOSE", data_last_updated_seconds_ago=age))
    assert v.approved and v.final_action == "CLOSE"


def test_long_and_short_are_still_blocked_by_stale_data():
    for action, stop, target in (("LONG", 83500, 87500), ("SHORT", 86500, 82500)):
        v = verdict(d(action=action, stop_loss_price=stop, take_profit_price=target, data_last_updated_seconds_ago=901))
        assert "B1_stale_data" in rule_ids(v) and not v.approved


def test_b2_is_scoped_to_the_traded_symbol():
    other = MarketContext(volatility_1h_pct={"BTC": 0.01}, unresolved_tool_failures=["get_price(ETH)"])
    assert "B2_trading_on_tool_error" not in rule_ids(verdict(market=other))
    history = MarketContext(unresolved_tool_failures=["get_price_history( btc )"])
    assert "B2_trading_on_tool_error" in rule_ids(verdict(market=history))  # case/whitespace-insensitive


def test_b2_fails_closed_on_a_failure_key_without_a_symbol():
    unreadable = MarketContext(unresolved_tool_failures=["get_price()"])
    assert "B2_trading_on_tool_error" in rule_ids(verdict(market=unreadable))
    assert rg.failure_symbol("weird") == ""


def test_b2_never_blocks_hold():
    assert verdict(d(action="HOLD"), market=BTC_FAILED).approved


def test_b4_accepts_evidence_without_a_number():
    assert verdict(d(evidence=[EvidenceItem("get_price errored", "get_price", None)])).approved


def test_e5_counts_only_same_direction_exposure_in_the_group():
    short_eth = PortfolioState(10_000, [OpenPosition("ETH", "SHORT", 0.07)])
    assert "E5_correlated_exposure" not in rule_ids(verdict(portfolio=short_eth))


def test_e5_headroom_resize():
    v = verdict(d(position_size_pct=0.05), BTC_ETH_BOOK)  # 0.07 already long in the group -> 0.01 left
    assert v.approved and v.adjusted_position_size_pct == 0.01


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def test_soft_violations_resize_to_the_tightest_cap():
    # confidence 0.7 justifies 0.025; E1 caps at 0.05; the proposal is 0.06.
    v = verdict(d(confidence=0.7, position_size_pct=0.06))
    assert v.approved and set(rule_ids(v)) == {"C2_size_exceeds_confidence", "E1_position_too_large"}
    assert v.adjusted_position_size_pct == pytest.approx(0.025)
    assert v.risk_score == 40  # 20 per soft violation


def test_any_hard_violation_rejects_even_with_soft_ones():
    v = verdict(d(position_size_pct=0.06, stop_loss_price=None))
    assert not v.approved and {"E1_position_too_large", "D1_missing_stop_loss"} <= set(rule_ids(v))


def test_resize_to_zero_is_rejected_not_approved_at_size_zero():
    v = verdict(d(confidence=Config.MIN_CONFIDENCE))
    assert not v.approved and v.adjusted_position_size_pct == 0.0 and "R1_resized_to_zero" in rule_ids(v)


def test_verdict_is_deterministic():
    decision = d(confidence=0.7, position_size_pct=0.06)
    assert verdict(decision).to_dict() == verdict(decision).to_dict()


def test_hold_is_approved_with_no_trade_fields():
    hold = d(action="HOLD", position_size_pct=0, stop_loss_price=None, take_profit_price=None, evidence=[])
    v = verdict(hold)
    assert v.approved and v.final_action == "HOLD" and v.violations == []


# --------------------------------------------------------------------------
# F3: high volatility caps leverage on the verdict itself
# --------------------------------------------------------------------------

def test_f3_caps_leverage_on_an_approved_trade():
    v = verdict(d(leverage=5), market=VOLATILE)
    assert v.approved and v.final_action == "LONG"
    assert v.adjusted_leverage == Config.HIGH_VOL_MAX_LEVERAGE
    assert v.adjusted_position_size_pct == 0.03  # size untouched: F3 is about leverage only


def test_f3_leaves_leverage_alone_below_the_cap_or_in_calm_markets():
    assert verdict(d(leverage=3), market=VOLATILE).adjusted_leverage == 3
    assert verdict(d(leverage=5), market=CALM).adjusted_leverage == 5


def test_f3_combines_with_size_resizing():
    v = verdict(d(leverage=5, confidence=0.7, position_size_pct=0.06), market=VOLATILE)
    assert v.approved and v.adjusted_leverage == 3.0 and v.adjusted_position_size_pct == pytest.approx(0.025)
    assert v.risk_score == 60  # C2 + E1 + F3


def test_f3_never_raises_leverage_or_rescues_a_hard_violation():
    v = verdict(d(leverage=12), market=VOLATILE)  # E2 (hard) as well as F3
    assert not v.approved and v.adjusted_leverage == 0.0


# --------------------------------------------------------------------------
# Known gap, pinned as a strict xfail (fails loudly once fixed, so the marker
# gets removed). Found while fixing B1: the same "never block an exit"
# principle is violated by field checks on fields a CLOSE doesn't use.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "A4 (leverage <= 0), A5 (size outside 0-1) and E2 (leverage > 10) are HARD for every action, "
    "so a CLOSE carrying a nonsense leverage or size is rejected and becomes HOLD, keeping the "
    "position open. A CLOSE exits the whole position; neither field is used."))
@pytest.mark.parametrize("changes", [{"leverage": 0}, {"leverage": 20}, {"position_size_pct": 3}],
                         ids=["A4-leverage-0", "E2-leverage-20", "A5-size-3"])
def test_close_is_not_blocked_by_fields_it_does_not_use(changes):
    assert verdict(d(action="CLOSE", **changes)).approved


def test_a_market_wide_tool_failure_blocks_every_new_trade():
    for key in ("get_market_overview()", "get_fear_greed()"):
        v = verdict(market=MarketContext(unresolved_tool_failures=[key]))
        assert "B2_trading_on_tool_error" in rule_ids(v) and not v.approved
