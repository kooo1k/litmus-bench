"""
snapshots.py

Replay recorded CMC responses instead of calling the live API, so every model
and every trial sees identical data, runs cost no API credits, and results are
reproducible.

A snapshot stores the data's AGE at recording time, not just its absolute
timestamps. At replay, every timestamp in the response is shifted by the same
offset so the data is exactly as old as it was when recorded:

    replayed last_updated = now - recorded_at_age_seconds

So a quote that was 42s old when recorded is 42s old on every replay, today or
in six months. Freshness rules (governor B1) behave the same on every run.
Deliberately stale data comes from faults.stale(), never from wall-clock drift.

Why this lives here and not in market.py: market.py is the live API layer the
real agent uses. Keeping replay out of it means the live CLI cannot serve
canned data by accident — replay only happens when a caller explicitly wires
in replay_tools().
"""

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from litmus import paths

# One directory per market scenario (see record_snapshots.py). SNAPSHOT_DIR
# is the default scenario: the live recording the original tasks were
# written against.
SNAPSHOT_ROOT = paths.SNAPSHOT_ROOT
SNAPSHOT_DIR = SNAPSHOT_ROOT / "2026-09-25_live"

# Keys whose values are ISO timestamps in CMC responses. An explicit list,
# not "any string that parses as a date": shifting a field that only happened
# to look like a date would silently corrupt data.
TIMESTAMP_KEYS = {"last_updated", "timestamp", "time_open", "time_close"}

FILE_SUFFIX = {"get_price": "price", "get_price_history": "history", "get_ohlcv": "ohlcv",
               "get_market_overview": "market_overview", "get_fear_greed": "fear_greed"}

# Tools that describe the whole market rather than one symbol: one file per
# scenario (market_overview.json), not one per symbol (btc_price.json).
MARKET_WIDE_TOOLS = {"get_market_overview", "get_fear_greed"}


def parse_ts(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def format_ts(dt):
    """CMC's format, e.g. 2026-09-25T08:12:02.000Z."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def anchor_timestamp(tool, response):
    """The timestamp whose age a snapshot records.

    get_price: last_updated, the field governor B1 judges freshness by.
    get_price_history: the newest daily candle (history has no last_updated).
    get_ohlcv: the newest candle's close.
    get_market_overview, get_fear_greed: last_updated.
    """
    if tool in ("get_price", "get_market_overview", "get_fear_greed"):
        return parse_ts(response["last_updated"])
    if tool == "get_price_history":
        return max(parse_ts(q["timestamp"]) for q in response["quotes"])
    if tool == "get_ohlcv":
        return max(parse_ts(c["time_close"]) for c in response["candles"])
    raise ValueError(f"no snapshot anchor defined for tool '{tool}'")


def snapshot_path(tool, symbol, directory=SNAPSHOT_DIR):
    """symbol is ignored (pass None) for market-wide tools."""
    if tool in MARKET_WIDE_TOOLS:
        return Path(directory) / f"{FILE_SUFFIX[tool]}.json"
    return Path(directory) / f"{symbol.strip().lower()}_{FILE_SUFFIX[tool]}.json"


def make_snapshot(tool, symbol, response, fetched_at):
    """Build a snapshot record from a live response.

    The age is fetched_at - anchor, measured when the response arrived — NOT
    assumed to be 0. CMC refreshes quotes periodically, so a just-fetched
    quote is typically tens of seconds old already. Recording 0 would make
    every replay look fresher than the real data was, which matters against
    the 60s scalp limit.
    """
    if not isinstance(response, dict) or "error" in response:
        # A recorded error would silently become a permanent fault in every
        # benchmark run that uses this snapshot.
        raise ValueError(f"refusing to snapshot an error response for {tool}({symbol}): {response}")
    age = (fetched_at - anchor_timestamp(tool, response)).total_seconds()
    if age < 0:
        # Only possible if the local clock is behind CMC's. Future-dated data
        # is meaningless to replay, so clamp — but say so.
        print(f"warning: {tool}({symbol}) age {age:.1f}s is negative (local clock behind CMC?); recording 0")
        age = 0.0
    return {
        "tool": tool,
        "symbol": None if symbol is None else symbol.strip().upper(),
        "recorded_at_age_seconds": round(age, 3),
        "response": response,
    }


def _shift_timestamps(obj, offset):
    if isinstance(obj, dict):
        return {k: format_ts(parse_ts(v) + offset) if k in TIMESTAMP_KEYS and isinstance(v, str)
                else _shift_timestamps(v, offset) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shift_timestamps(x, offset) for x in obj]
    return obj


def load_snapshot(path, now=None):
    """Return the recorded tool response with its timestamps re-issued so its
    anchor is exactly recorded_at_age_seconds old at `now`.

    Every timestamp in the response moves by the same offset, not only the
    anchor. Otherwise a quote re-dated to today would sit next to price
    history still ending on the recording date, and a model replayed months
    later would see a gap that never existed in the real data.
    """
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    now = now or datetime.now(timezone.utc)
    anchor = anchor_timestamp(snapshot["tool"], snapshot["response"])
    offset = (now - timedelta(seconds=snapshot["recorded_at_age_seconds"])) - anchor
    return _shift_timestamps(copy.deepcopy(snapshot["response"]), offset)


def replay_tools(directory=SNAPSHOT_DIR):
    """Drop-in replacements for market.get_price / get_price_history, for
    run_agent(tool_functions=...). Signatures match market.py exactly, because
    agent.py binds arguments against them to classify argument errors, and
    faults.py can wrap these just like the live tools.

    A symbol with no snapshot gets market.py's exact "not found" error: from
    the model's side, an unrecorded asset looks like one the API doesn't list.
    """
    def get_price(symbol: str):
        path = snapshot_path("get_price", symbol, directory)
        if not path.exists():
            return {"error": f"symbol '{symbol.strip().upper()}' not found"}
        return load_snapshot(path)

    def get_price_history(symbol: str, time_start: int = None, time_end: int = None,
                          interval: str = "daily", count: int = 7):
        path = snapshot_path("get_price_history", symbol, directory)
        if not path.exists():
            return {"error": f"no history found for symbol '{symbol.strip().upper()}'"}
        response = load_snapshot(path)
        # Only the recorded window exists. A smaller count gets the newest
        # `count` days; a larger one gets everything recorded. time_start,
        # time_end and interval are accepted (for signature parity) but ignored.
        if isinstance(count, int) and 0 < count < len(response["quotes"]):
            response["quotes"] = response["quotes"][-count:]
        return response

    def get_ohlcv(symbol: str, count: int = 7):
        path = snapshot_path("get_ohlcv", symbol, directory)
        if not path.exists():
            return {"error": f"no OHLCV found for symbol '{symbol.strip().upper()}'"}
        response = load_snapshot(path)
        if isinstance(count, int) and 0 < count < len(response["candles"]):
            response["candles"] = response["candles"][-count:]
        return response

    def market_wide(tool):
        # A scenario without this file is a recording gap, not a market
        # event, so the message says so rather than imitating an API error.
        def replay():
            path = snapshot_path(tool, None, directory)
            if not path.exists():
                return {"error": f"{tool} is not recorded for this scenario"}
            return load_snapshot(path)
        replay.__name__ = tool
        return replay

    return {"get_price": get_price, "get_price_history": get_price_history, "get_ohlcv": get_ohlcv,
            "get_market_overview": market_wide("get_market_overview"),
            "get_fear_greed": market_wide("get_fear_greed")}
