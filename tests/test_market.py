"""market.py's live tools against a fake HTTP layer: response shaping, and
the contract that every failure comes back as {"error": ...}, never raises."""

from datetime import datetime, timedelta, timezone

import pytest
import requests

from litmus import market


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def serve(monkeypatch, status=200, data=None, error_message=None, raise_exc=None):
    calls = []

    def get(url, headers=None, params=None, timeout=None):
        calls.append((url, params))
        if raise_exc:
            raise raise_exc
        return FakeResponse(status, {"status": {"error_message": error_message}, "data": data})
    monkeypatch.setattr(requests, "get", get)
    return calls


GLOBAL = {"btc_dominance": 58.5, "eth_dominance": 11.4, "btc_dominance_yesterday": 58.9,
          "quote": {"USD": {"total_market_cap": 2.86e12, "total_volume_24h": 1.06e11,
                            "total_market_cap_yesterday_percentage_change": -0.36,
                            "last_updated": "2026-09-25T15:44:59.999Z"}}}


def test_market_overview_shape(monkeypatch):
    serve(monkeypatch, data=GLOBAL)
    assert market.get_market_overview() == {
        "total_market_cap_usd": 2.86e12, "total_market_cap_change_24h_pct": -0.36, "total_volume_24h_usd": 1.06e11,
        "btc_dominance_pct": 58.5, "btc_dominance_yesterday_pct": 58.9, "eth_dominance_pct": 11.4,
        "last_updated": "2026-09-25T15:44:59.999Z"}


def test_fear_greed_shape(monkeypatch):
    serve(monkeypatch, data={"value": 72, "value_classification": "Greed", "update_time": "2026-09-25T15:38:10.033Z"})
    assert market.get_fear_greed() == {"value": 72, "classification": "Greed", "last_updated": "2026-09-25T15:38:10.033Z"}


def candle(day, closed=True):
    open_ = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day)
    close = open_ + timedelta(days=1, milliseconds=-1)
    return {"time_open": open_.isoformat().replace("+00:00", "Z"), "time_close": close.isoformat().replace("+00:00", "Z"),
            "quote": {"USD": {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 9.0}}}


def test_ohlcv_returns_only_completed_candles(monkeypatch):
    # day 0 = today's candle, still open: must be dropped.
    calls = serve(monkeypatch, data={"BTC": [{"symbol": "BTC", "quotes": [candle(d) for d in (3, 2, 1, 0)]}]})
    out = market.get_ohlcv(" btc ", count=3)
    assert out["symbol"] == "BTC" and len(out["candles"]) == 3
    assert all(datetime.fromisoformat(c["time_close"].replace("Z", "+00:00")) <= datetime.now(timezone.utc) for c in out["candles"])
    assert set(out["candles"][0]) == {"time_open", "time_close", "open", "high", "low", "close", "volume"}
    assert calls[0][1]["count"] == 4  # one extra, to cover the open candle


@pytest.mark.parametrize("tool", [market.get_market_overview, market.get_fear_greed, lambda: market.get_ohlcv("BTC")],
                         ids=["overview", "fear_greed", "ohlcv"])
def test_failures_are_error_dicts(monkeypatch, tool):
    serve(monkeypatch, status=429, error_message="rate limit")
    assert tool() == {"error": "API returned HTTP 429: rate limit"}
    serve(monkeypatch, raise_exc=requests.Timeout())
    assert tool() == {"error": f"request timed out after {market.REQUEST_TIMEOUT_SECONDS}s"}
    serve(monkeypatch, data={"unexpected": True})
    assert "error" in tool()


def test_ohlcv_unknown_symbol(monkeypatch):
    serve(monkeypatch, data={})
    assert market.get_ohlcv("nope") == {"error": "no OHLCV found for symbol 'NOPE'"}
