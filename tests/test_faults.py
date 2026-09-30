"""faults.py against fixed fake tools. No I/O."""

import inspect

import pytest
import requests

from litmus import faults


def get_price(symbol: str):
    return {"symbol": symbol.upper(), "price_usd": 85012.3, "percent_change_1h": 0.2,
            "percent_change_24h": -1.4, "last_updated": "2026-09-25T08:00:00.000Z"}


def get_price_history(symbol: str, count: int = 7):
    return {"symbol": symbol.upper(), "quotes": [{"price": 84000.0}]}


TOOLS = {"get_price": get_price, "get_price_history": get_price_history}


def faulted(fault, tool="get_price"):
    return faults.inject(TOOLS, {tool: fault})[tool]


def test_error_faults_match_market_py_messages():
    assert faulted(faults.timeout())("BTC") == {"error": "request timed out after 10s"}
    assert faulted(faults.unavailable_asset())("btc") == {"error": "symbol 'BTC' not found"}
    assert faulted(faults.unavailable_asset(), "get_price_history")("eth") == {"error": "no history found for symbol 'ETH'"}
    assert faulted(faults.rate_limit())("BTC")["error"].startswith("API returned HTTP 429")


def test_timeout_can_raise():
    with pytest.raises(requests.Timeout):
        faulted(faults.timeout(mode="raise"))("BTC")


def test_data_faults_alter_one_field_and_nothing_else():
    clean = get_price("BTC")
    missing = faulted(faults.missing_field("price_usd"))("BTC")
    assert "price_usd" not in missing and {k: v for k, v in clean.items() if k != "price_usd"} == missing
    contradicted = faulted(faults.contradict("percent_change_24h", 8.0))("BTC")
    assert contradicted == {**clean, "percent_change_24h": 8.0}
    stale = faulted(faults.stale(7200))("BTC")
    assert stale["last_updated"] != clean["last_updated"] and stale["price_usd"] == clean["price_usd"]


def test_data_faults_pass_real_errors_through():
    def broken(symbol: str):
        return {"error": "real failure"}
    assert faults.inject({"get_price": broken}, {"get_price": faults.missing_field()})["get_price"]("BTC") == {"error": "real failure"}


def test_times_limits_the_fault_then_passes_through():
    once = faulted(faults.rate_limit(times=1))
    assert "error" in once("BTC") and "error" not in once("BTC")


@pytest.mark.parametrize("bad", [0, -1])
def test_times_below_one_is_rejected(bad):
    with pytest.raises(ValueError, match="times must be >= 1"):
        faults.rate_limit(times=bad)


def test_symbols_scopes_the_fault():
    eth_only = faulted(faults.unavailable_asset(symbols={"eth"}))
    assert "error" not in eth_only("BTC") and "error" in eth_only(" ETH ")


def test_each_inject_starts_a_fresh_counter():
    fault = faults.rate_limit(times=1)
    assert "error" in faults.inject(TOOLS, {"get_price": fault})["get_price"]("BTC")
    assert "error" in faults.inject(TOOLS, {"get_price": fault})["get_price"]("BTC")


def test_signature_is_preserved_for_argument_binding():
    assert inspect.signature(faulted(faults.timeout())) == inspect.signature(get_price)


def test_unknown_tool_name_is_rejected():
    with pytest.raises(KeyError, match="get_prices"):
        faults.inject(TOOLS, {"get_prices": faults.timeout()})


def test_inject_does_not_mutate_its_input():
    faults.inject(TOOLS, {"get_price": faults.timeout()})
    assert TOOLS["get_price"] is get_price
