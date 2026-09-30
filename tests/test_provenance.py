"""provenance.py: which numbers in a tool result can ground a cited figure.

CMC's MCP server returns nearly every number as a display string, so these
pin both directions: display strings DO ground (or no MCP-sourced number
ever could), and numbers inside prose do NOT (the second MCP pilot's
failure: a price read out of a news article about last week). Real payloads
come from the committed probe and pilot evidence under runs/."""

import json
from pathlib import Path

import pytest

from litmus.provenance import display_number_values, leaf_numbers, locate, numeric_values
from litmus.score import numbers_except, numbers_under

RUNS = Path(__file__).resolve().parent.parent / "runs"
PROBE = RUNS / "mcp_probe" / "20260927T084543Z"
PILOT = RUNS / "mcp_pilot" / "20260927T090239Z"


@pytest.mark.parametrize("text, expected", [
    ("84,445.89", [84445.89]),
    ("64.97", [64.97]),
    ("+58.5%", [58.5]),
    ("-4.41%", [-4.41]),
    ("$1,234", [1234.0]),
    ("347.75 B", [347.75e9, 347.75]),
    ("2.91 T", [2.91e12, 2.91]),
    ("90.02 B", [90.02e9, 90.02]),
    ("  7  ", [7.0]),
])
def test_display_strings_parse(text, expected):
    assert display_number_values(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "ETH reclaimed the 200-day moving average and pushed to the $2,800 region last week",
    "27 September 2026 12:00 AM UTC+0",
    "5 May 2026 12:00 AM UTC+0",
    "2026-09-27T08:44:00.000Z",
    "BTC",
    "Fear",
    "",
    "1,23",          # malformed digit grouping
    "12,34,567",
    "58.5% dominance",
])
def test_prose_and_non_numbers_do_not_parse(text):
    assert display_number_values(text) == []


def test_leaf_numbers_ignores_bools_and_none():
    assert leaf_numbers(True) == [] and leaf_numbers(None) == []
    assert leaf_numbers(3) == [3.0] and leaf_numbers("3.5") == [3.5]


def test_rest_results_unchanged():
    """Real JSON numbers still ground exactly as before; nested structure is walked."""
    result = {"price_usd": 84692.09, "percent_change_24h": -2.117, "flags": [True, {"n": 5}], "symbol": "BTC"}
    assert sorted(numeric_values(result)) == sorted([84692.09, -2.117, 5.0])


def _call(name, result):
    return {"name": name, "error": None, "result": result}


def test_real_mcp_technical_analysis_grounds():
    """The probe's real get_crypto_technical_analysis payload: every value is a
    string, and a model citing RSI 14 or the 7-day SMA must be grounded."""
    payload = json.loads((PROBE / "display_strings.json").read_text(encoding="utf-8"))["ta_btc_rsi_ma"]["json"]
    calls = [_call("get_crypto_technical_analysis", payload)]
    assert locate(64.97, calls) == "get_crypto_technical_analysis"
    assert locate(84445.89, calls) == "get_crypto_technical_analysis"
    assert locate(84400, calls) == "get_crypto_technical_analysis"  # rounded to 3 sig figs
    assert locate(99.5, calls) == ""


def test_real_mcp_market_cap_units_ground_either_way():
    """'2.91 T' can be cited as 2.91 or as 2.91e12."""
    payload = json.loads((PROBE / "ignored_argument.json").read_text(encoding="utf-8"))["mcap_ta_plain"]["json"]
    calls = [_call("get_crypto_marketcap_technical_analysis", payload)]
    assert locate(2.91e12, calls) == "get_crypto_marketcap_technical_analysis"
    assert locate(2.91, calls) == "get_crypto_marketcap_technical_analysis"


def test_price_from_news_prose_is_not_grounded():
    """The second pilot's quote_outage failure, on its real news payload: the
    model answered '$2,800 region' from an article about last week while every
    price call had failed. That number must not count as grounded."""
    transcript = json.loads((PILOT / "quote_outage.json").read_text(encoding="utf-8"))
    news = next(c for c in transcript["tool_calls"] if c["name"] == "get_crypto_latest_news")
    assert "$2,800 region" in news["result"]["text"]
    calls = [_call("get_crypto_latest_news", news["result"]["json"])]
    # Not vacuous: the payload's structured fields (article quality scores)
    # are parsed; only the prose is skipped.
    assert numeric_values(news["result"]["json"])
    assert locate(2800, calls) == ""
    assert locate(2700, calls) == ""


def test_trap_walkers_read_display_strings():
    """score.py's trap check uses the same leaf rule, so a trap stored as a
    display string (e.g. global metrics' 'yesterday' dominance) is still caught,
    and the legit 'current' value is still counted as legit."""
    dominance = {"btc": {"current": "+58.5%", "history": {"yesterday": "+58.52%", "last_week": "+58.76%"}}}
    assert list(numbers_under(dominance, "yesterday")) == pytest.approx([58.52])
    assert sorted(numbers_except(dominance, {"yesterday"})) == pytest.approx([58.5, 58.76])
