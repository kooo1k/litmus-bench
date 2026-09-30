"""
live_tasks.py

Tasks graded on today's market instead of a recording. A recorded task's
right answer was worked out once, for the moment it was recorded. A live
task's right answer is computed by Litmus from its own fetch of the same
CoinMarketCap data just before the agent runs. The agent then runs on CMC's
MCP server, live, and is graded against that answer. Just after the run,
Litmus fetches again: if the market crossed the line in between, the grade
is marked inconclusive instead of being trusted.

    litmus --task live_eur_threshold

live_eur_threshold is the headline failure (mcp_eur_threshold) rebuilt for
any day. The threshold is set halfway between ETH's price in euros and its
price in dollars, so the right reading (euros, via EURC) and the trap (the
dollar quote read as euros) land on opposite sides of it, each by several
percent. That margin is far wider than ETH moves in the minute a run takes.
Which side means "go long" is drawn at random each run ("worth more than"
or "worth less than"), so the right answer is LONG or SHORT and an agent
can't pass by always making the same trade. On a day when the euro and the
dollar are too close to tell apart, the task refuses to run rather than
grade a coin flip.
"""

import random
from datetime import datetime, timezone

from litmus.provenance import rows_as_dicts

ETH_ID, EURC_ID = "1027", "20641"

# Both readings must clear the threshold by at least this much, or the
# task isn't set: a smaller gap would grade market noise, not the misread.
MIN_MARGIN = 0.01

EUR_THRESHOLD = {
    "id": "live_eur_threshold",
    "category": "mcp_live",
    "toolset": "mcp",
    "live": "eur_threshold",
    "question": "If 1 ETH is worth more (or less) than €<set from today's prices> right now, go long ETH; "
                "otherwise short it.  (graded on today's live market)",
    "faults": {},
}

LIVE_TASKS = [EUR_THRESHOLD]


class LiveTaskUnavailable(RuntimeError):
    """Today's market can't support the task (or its data couldn't be
    fetched); the message says which."""


def _prices(quote_tool):
    """(ETH in USD, EURC in USD) from one quotes call for both ids."""
    try:
        rows = quote_tool(id=f"{ETH_ID},{EURC_ID}")
    except Exception as e:  # network, auth, server
        raise LiveTaskUnavailable(f"couldn't fetch today's ETH and EURC prices: {type(e).__name__}: {e}") from e
    if isinstance(rows, dict) and "error" in rows:
        raise LiveTaskUnavailable(f"couldn't fetch today's ETH and EURC prices: {rows['error']}")
    by_id = {str(r.get("id")): r.get("price") for r in rows_as_dicts(rows)}
    eth, eurc = by_id.get(ETH_ID), by_id.get(EURC_ID)
    if not isinstance(eth, (int, float)) or not isinstance(eurc, (int, float)) or eth <= 0 or eurc <= 0:
        raise LiveTaskUnavailable(f"the quotes call didn't return both prices (got {by_id})")
    return float(eth), float(eurc)


def build_eur_threshold(quote_tool, now=None, long_if=None):
    """The task, with today's threshold, expected answer and rationale.
    quote_tool is CMC's get_crypto_quotes_latest (live). long_if is "more"
    or "less" (which side of the threshold means go long); None draws it at
    random."""
    long_if = long_if or random.choice(("more", "less"))
    if long_if not in ("more", "less"):
        raise ValueError(f"long_if must be 'more' or 'less', not {long_if!r}")
    eth_usd, eurc_usd = _prices(quote_tool)
    eth_eur = eth_usd / eurc_usd
    # Halfway (in ratio) between the euro price and the dollar price, rounded
    # to a round number a person would write.
    threshold = round(eth_eur * (1 + (eurc_usd - 1) / 2) / 10) * 10
    below, above = (threshold - eth_eur) / eth_eur, (eth_usd - threshold) / threshold
    if below < MIN_MARGIN or above < MIN_MARGIN:
        raise LiveTaskUnavailable(
            f"today EURC is ${eurc_usd:.4f}: the euro and the dollar are too close for this task to tell "
            f"a misread apart (ETH is €{eth_eur:,.0f} vs ${eth_usd:,.0f}); try a recorded task instead")
    fetched_at = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    # ETH in euros is always below the threshold and the misread always
    # above, so "long if more" makes SHORT right and "long if less" LONG.
    right, trap = ("SHORT", "LONG") if long_if == "more" else ("LONG", "SHORT")
    other = "less" if long_if == "more" else "more"
    return {
        **{k: v for k, v in EUR_THRESHOLD.items() if k != "question"},
        "question": (f"If 1 ETH is worth {long_if} than €{threshold:,} right now, go long ETH; "
                     f"if it's worth {other}, short it."),
        "expected": {
            "required_tools": ["get_crypto_quotes_latest(ETH)", "get_crypto_quotes_latest(EURC)"],
            "acceptable_actions": [right],
            "acceptable_symbols": ["ETH"],
        },
        "rationale": (
            f"Set from Litmus's own fetch at {fetched_at}: ETH ${eth_usd:,.2f}, EURC (the euro stablecoin) "
            f"${eurc_usd:.4f}, so 1 ETH is about €{eth_eur:,.0f}, {below:.1%} below €{threshold:,}: {right}. "
            f"Reading the dollar quote as euros gives {eth_usd:,.0f}, {above:.1%} above it: {trap}. CMC's quote "
            "tool is USD-only and silently ignores a convert argument; EURC is the only tool-sourced euro rate. "
            f"Whether 'more' or 'less' means long is drawn at random each run, so the right answer isn't "
            "always the same trade."),
        "reference": {"eth_usd": eth_usd, "eurc_usd": eurc_usd, "eth_eur": eth_eur, "threshold": threshold,
                      "long_if": long_if, "fetched_at": fetched_at},
    }


BUILDERS = {"eur_threshold": build_eur_threshold}


def build(task, quote_tool, **options):
    """A live task stub (from LIVE_TASKS) -> the task for right now.
    options go to the task's builder (e.g. long_if, to fix the direction)."""
    return BUILDERS[task["live"]](quote_tool, **options)


def recheck(task, quote_tool):
    """After the run: is today's right answer still the same? Returns None
    if it is (or the check itself failed: the answer before the run
    stands), else a message saying why the grade is inconclusive."""
    try:
        eth_usd, eurc_usd = _prices(quote_tool)
    except LiveTaskUnavailable:
        return None
    eth_eur = eth_usd / eurc_usd
    threshold = task["reference"]["threshold"]
    if eth_eur > threshold:
        return (f"inconclusive: ETH rose to €{eth_eur:,.0f} during the run, above the €{threshold:,} threshold, "
                "so the right answer changed while the agent was working")
    if eth_usd < threshold:
        return (f"inconclusive: ETH fell to ${eth_usd:,.0f} during the run, below €{threshold:,}, so the "
                "euro misread no longer leads to a different answer")
    return None
