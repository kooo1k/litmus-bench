"""
market.py

The live CoinMarketCap REST tools the agent calls by default: current price,
price history, OHLCV candles, market overview (total cap, BTC dominance) and
Fear & Greed. Each returns a small dict shaped for the model, or an error
dict it can read, never an exception. The recorded replays (snapshots.py)
return the same shapes.
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CMC_API_KEY")

# CMC normally answers in well under a second. 10s bounds a hung connection
# so a live demo can't freeze, while still tolerating a slow network.
REQUEST_TIMEOUT_SECONDS = 10


def _cmc_get(url, params):
    """GET a CMC endpoint. Returns (data, None) or (None, error_dict).
    Every failure becomes a short error dict the agent can read, never an
    exception or a raw HTML/JSON dump."""
    try:
        response = requests.get(
            url,
            headers={"x-cmc_pro_api_key": API_KEY},
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, {"error": f"request timed out after {REQUEST_TIMEOUT_SECONDS}s"}
    except requests.RequestException as e:
        return None, {"error": f"request failed: {type(e).__name__}"}

    try:
        body = response.json()
    except ValueError:
        return None, {"error": f"API returned HTTP {response.status_code} with a non-JSON body"}

    if response.status_code != 200:
        message = (body.get("status") or {}).get("error_message") if isinstance(body, dict) else None
        return None, {"error": f"API returned HTTP {response.status_code}: {message or 'no error message'}"}
    return body.get("data"), None


def get_price(symbol: str):
    symbol = symbol.strip().upper()
    data, error = _cmc_get(
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
        {"symbol": symbol},
    )
    if error:
        return error

    coin = (data or {}).get(symbol)
    if not coin:
        return {"error": f"symbol '{symbol}' not found"}
    try:
        quote = coin["quote"]["USD"]
        return {
            "symbol": coin["symbol"],
            "price_usd": quote["price"],
            "percent_change_1h": quote["percent_change_1h"],
            "percent_change_24h": quote["percent_change_24h"],
            "last_updated": quote["last_updated"],
        }
    except (KeyError, TypeError) as e:
        return {"error": f"unexpected response shape for '{symbol}': missing {e}"}


def get_price_history(symbol: str, time_start: int = None, time_end: int = None, interval: str = "daily", count: int = 7):
    symbol = symbol.strip().upper()
    data, error = _cmc_get(
        "https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical",
        {"symbol": symbol, "time_start": time_start, "time_end": time_end, "interval": interval, "count": count},
    )
    if error:
        return error

    entries = (data or {}).get(symbol)
    if not entries:
        return {"error": f"no history found for symbol '{symbol}'"}
    return entries[0]


# --------------------------------------------------------------------------
# Market-wide and range tools. Same contract as above: a trimmed dict, or
# {"error": ...}, never an exception. Every result carries a timestamp
# ("last_updated", or time_open/time_close per candle) so replay can age it
# and faults.stale can target it.
# --------------------------------------------------------------------------

def _shape_market_overview(data, yesterday_change_pct, btc_dominance_yesterday):
    usd = data["quote"]["USD"]
    return {
        "total_market_cap_usd": usd["total_market_cap"],
        "total_market_cap_change_24h_pct": yesterday_change_pct,
        "total_volume_24h_usd": usd["total_volume_24h"],
        "btc_dominance_pct": data["btc_dominance"],
        "btc_dominance_yesterday_pct": btc_dominance_yesterday,
        "eth_dominance_pct": data["eth_dominance"],
        "last_updated": usd.get("last_updated") or data.get("last_updated") or data.get("timestamp"),
    }


def get_market_overview():
    data, error = _cmc_get("https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest", {})
    if error:
        return error
    try:
        usd = data["quote"]["USD"]
        return _shape_market_overview(data, usd["total_market_cap_yesterday_percentage_change"],
                                      data["btc_dominance_yesterday"])
    except (KeyError, TypeError) as e:
        return {"error": f"unexpected response shape for market overview: missing {e}"}


def _shape_fear_greed(value, classification, last_updated):
    return {"value": value, "classification": classification, "last_updated": last_updated}


def get_fear_greed():
    data, error = _cmc_get("https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest", {})
    if error:
        return error
    try:
        return _shape_fear_greed(data["value"], data["value_classification"], data["update_time"])
    except (KeyError, TypeError) as e:
        return {"error": f"unexpected response shape for fear and greed: missing {e}"}


def _shape_candle(q):
    usd = q["quote"]["USD"]
    return {"time_open": q["time_open"], "time_close": q["time_close"], "open": usd["open"],
            "high": usd["high"], "low": usd["low"], "close": usd["close"], "volume": usd["volume"]}


def get_ohlcv(symbol: str, count: int = 7):
    """The last `count` COMPLETED daily candles. Today's partial candle is
    left out, so the range and close of every candle are final."""
    symbol = symbol.strip().upper()
    data, error = _cmc_get(
        "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical",
        {"symbol": symbol, "time_period": "daily", "count": count + 1},
    )
    if error:
        return error
    entries = (data or {}).get(symbol)
    if not entries:
        return {"error": f"no OHLCV found for symbol '{symbol}'"}
    try:
        entry = entries[0] if isinstance(entries, list) else entries
        now = datetime.now(timezone.utc)
        closed = [q for q in entry["quotes"]
                  if datetime.fromisoformat(q["time_close"].replace("Z", "+00:00")) <= now]
        return {"symbol": entry["symbol"], "candles": [_shape_candle(q) for q in closed[-count:]]}
    except (KeyError, TypeError, IndexError) as e:
        return {"error": f"unexpected response shape for '{symbol}' OHLCV: missing {e}"}
