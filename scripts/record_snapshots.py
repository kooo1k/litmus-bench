"""
record_snapshots.py

Fetch real CMC responses (no faults) and save them as replay snapshots, one
directory per market scenario under tasks/snapshots/. Rerun only to
deliberately refresh the data, since any change to the snapshots changes what
the benchmark measures.

    python -m scripts.record_snapshots live                  # now -> tasks/snapshots/<today>_live/
    python -m scripts.record_snapshots historical            # every scenario in HISTORICAL
    python -m scripts.record_snapshots historical 2026-02-06_crash
    python -m scripts.record_snapshots add-tools             # market-wide + OHLCV tools for existing scenarios

All-or-nothing per scenario: every response is fetched first and nothing is
written unless all of them succeed, so a scenario never mixes data from two
moments.

Historical scenarios. The Startup plan serves intraday history (hourly, 5m)
for the last month only, but daily history back at least a year. So a past
moment is rebuilt from daily candles alone: the "quote" is the daily data
point at the scenario date (00:00 UTC), which carries exactly the fields
get_price returns (price, 1h and 24h change, timestamp), and the history is
the 7 daily candles before it. The model sees the same shapes as a live run.

Market-wide and OHLCV tools (get_market_overview, get_fear_greed, get_ohlcv)
are rebuilt for any past instant T with one rule: the latest data that
EXISTED at T. Global metrics: the point at or before T (5-minute data inside
the plan's one-month intraday window, daily points before it) and the point
24h earlier, for the 24h change. Fear & Greed: the live endpoint serves an
intraday value, but history keeps only each day's FINAL value (stamped 00:00,
known only once the day ends), and the intraday value at T was never
recorded. So a past scenario gets the final value of the day BEFORE T's:
the newest value certainly known at T. Using T's own day would hand the
agent its future. It is a day staler than a live reading would be. OHLCV: the
daily candles that had closed by T. add-tools applies this to scenarios that
were recorded before these tools existed, at each scenario's own instant,
without touching its price/history files.
"""

import json
import sys
from datetime import datetime, timedelta, timezone

from litmus.market import (_cmc_get, _shape_candle, _shape_fear_greed, _shape_market_overview, get_fear_greed,
                    get_market_overview, get_ohlcv, get_price, get_price_history)
from litmus.snapshots import SNAPSHOT_ROOT, format_ts, make_snapshot, parse_ts, snapshot_path

SYMBOLS = ["BTC", "ETH"]
HISTORY_DAYS = 7  # matches market.get_price_history's default and the tool schema's stated default

HISTORY_URL = "https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical"
GLOBAL_HISTORY_URL = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/historical"
FEAR_GREED_HISTORY_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
OHLCV_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"

# The plan serves intraday history (5m, hourly) for the last month only; a
# day of margin keeps a request near the boundary from being refused.
INTRADAY_WINDOW = timedelta(days=29)

# A historical quote has no real "age at fetch": it is a data point, not a
# fetch. 30s is pinned deliberately. It sits inside every freshness limit
# (scalp is 60s), so no scenario fails B1 because of when it happened to be
# recorded, which is the problem the live snapshot has (recorded at ~61s).
# Staleness is a fault (faults.stale), never an accident of recording.
HISTORICAL_QUOTE_AGE_SECONDS = 30

# Hand-picked from a year of daily BTC/ETH data (2025-09-25 .. 2026-09-24),
# one date per distinct market condition. Each is the most extreme date for
# its criterion, with near-duplicate dates from the same episode dropped.
HISTORICAL = {
    "2025-10-11_drop_from_highs": ("2025-10-11", "Sharp drop near the highs: BTC -7.2%, ETH -12.2% in 24h, BTC ~113k."),
    "2025-10-19_divergence": ("2025-10-19", "BTC and ETH disagree: 7d BTC -3.3%, ETH +3.7%."),
    "2025-12-31_quiet": ("2025-12-31", "Quietest week of the year: BTC 7d range 1.5%."),
    "2026-02-06_crash": ("2026-02-06", "Crash: BTC -14.0% in 24h, -25.7% over 7d."),
    "2026-02-07_relief_rally": ("2026-02-07", "Relief rally inside a crash: BTC +12.3% in 24h, still -16.2% over 7d."),
    "2026-03-10_chop": ("2026-03-10", "Chop: BTC 7d range 10.2% but net -0.5%; +3.7% in 24h."),
    "2026-06-07_downtrend": ("2026-06-07", "Grinding downtrend: BTC -17.5% over 7d, flat day."),
    "2026-08-22_uptrend": ("2026-08-22", "Strong uptrend: BTC +24.4% over 7d, +7.3% in 24h."),
}


def write_scenario(name, snapshots, meta):
    directory = SNAPSHOT_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    for snap in snapshots:
        path = snapshot_path(snap["tool"], snap["symbol"], directory)
        path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    (directory / "scenario.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {directory.relative_to(SNAPSHOT_ROOT.parent.parent)} ({len(snapshots)} snapshots)")


def record_live():
    # Refuse before fetching: a second live recording on the same day would
    # land in the same directory and silently replace the data the tasks
    # were written against. Historical scenarios may be re-recorded in
    # place, since the same date returns the same candles; live data never
    # comes back the same.
    name = f"{datetime.now(timezone.utc):%Y-%m-%d}_live"
    if (SNAPSHOT_ROOT / name).exists():
        print(f"{SNAPSHOT_ROOT / name} already exists; refusing to overwrite a live recording. "
              "Move or delete it first if you really mean to replace it.")
        return 1
    calls = [(tool, symbol, fetch) for symbol in SYMBOLS for tool, fetch in (
        ("get_price", lambda s: get_price(s)),
        ("get_price_history", lambda s: get_price_history(s, count=HISTORY_DAYS)),
        ("get_ohlcv", lambda s: get_ohlcv(s, count=HISTORY_DAYS)),
    )] + [("get_market_overview", None, lambda s: get_market_overview()),
          ("get_fear_greed", None, lambda s: get_fear_greed())]
    snapshots = []
    for tool, symbol, fetch in calls:
        response = fetch(symbol)
        fetched_at = datetime.now(timezone.utc)  # as soon as the response arrived
        try:
            snapshots.append(make_snapshot(tool, symbol, response, fetched_at))
        except (ValueError, KeyError, TypeError) as e:
            print(f"failed: {tool}({symbol or ''}): {e}\nnothing written.")
            return 1
        print(f"fetched {tool}({symbol or ''}): age {snapshots[-1]['recorded_at_age_seconds']}s")
    write_scenario(name, snapshots, {
        "kind": "live", "as_of": datetime.now(timezone.utc).isoformat(),
        "description": "Live CMC responses; ages measured at fetch time.",
    })
    return 0


def historical_snapshots(date, symbol):
    """(get_price snapshot, get_price_history snapshot) for `symbol` as of
    `date` 00:00 UTC. Raises ValueError if CMC doesn't return the exact
    candles needed."""
    at = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    # The endpoint's window edges are loose (a time_start of D can return
    # D+1 first), so ask wide and select the exact days by timestamp.
    data, error = _cmc_get(HISTORY_URL, {
        "symbol": symbol, "interval": "daily",
        "time_start": (at - timedelta(days=HISTORY_DAYS + 2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_end": (at + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    if error:
        raise ValueError(error["error"])
    entry = data[symbol][0]
    by_time = {parse_ts(q["timestamp"]): q for q in entry["quotes"]}
    wanted = [at - timedelta(days=d) for d in range(HISTORY_DAYS, 0, -1)]
    missing = [t for t in wanted + [at] if t not in by_time]
    if missing:
        raise ValueError(f"CMC returned no daily candle for {[t.date().isoformat() for t in missing]}")

    point = by_time[at]["quote"]["USD"]
    # Same keys, same order as market.get_price.
    quote = {
        "symbol": entry["symbol"],
        "price_usd": point["price"],
        "percent_change_1h": point["percent_change_1h"],
        "percent_change_24h": point["percent_change_24h"],
        "last_updated": by_time[at]["timestamp"],
    }
    history = {**entry, "quotes": [by_time[t] for t in wanted]}
    # Both files share one "fetch instant", so the quote is 30s old and the
    # newest history candle is one day + 30s old, as they were relative to
    # each other on the day.
    fetched_at = at + timedelta(seconds=HISTORICAL_QUOTE_AGE_SECONDS)
    return (make_snapshot("get_price", symbol, quote, fetched_at),
            make_snapshot("get_price_history", symbol, history, fetched_at))


def _latest_at(points, t, key):
    """The point with the latest timestamp at or before t (None if none)."""
    eligible = [p for p in points if parse_ts(key(p)) <= t]
    return max(eligible, key=lambda p: parse_ts(key(p))) if eligible else None


def market_overview_at(t):
    """get_market_overview's response as it would have read at t."""
    def point(at):
        intraday = datetime.now(timezone.utc) - at < INTRADAY_WINDOW
        if intraday:
            params = {"interval": "5m", "time_start": format_ts(at - timedelta(minutes=20)), "time_end": format_ts(at)}
        else:
            params = {"interval": "daily", "time_start": format_ts(at - timedelta(days=2)), "time_end": format_ts(at)}
        data, error = _cmc_get(GLOBAL_HISTORY_URL, params)
        if error:
            raise ValueError(error["error"])
        found = _latest_at((data or {}).get("quotes", []), at, lambda q: q["timestamp"])
        if found is None:
            raise ValueError(f"no global metrics point at or before {format_ts(at)}")
        return found

    now, before = point(t), point(t - timedelta(days=1))
    gap_h = (parse_ts(now["timestamp"]) - parse_ts(before["timestamp"])).total_seconds() / 3600
    if not 23.5 <= gap_h <= 24.5:
        raise ValueError(f"global metrics points are {gap_h:.2f}h apart, not 24h")
    change = (now["quote"]["USD"]["total_market_cap"] / before["quote"]["USD"]["total_market_cap"] - 1) * 100
    return _shape_market_overview(now, change, before["btc_dominance"])


_fear_greed_days = None


def fear_greed_at(t):
    """The newest FINAL daily value known at t. History stamps each day's
    final value at 00:00 of that day, but it is only final once the day
    ends, so the newest one known at t is stamped at or before t minus one
    day. (The live endpoint also serves an intraday value; history doesn't
    keep those, so a reconstruction can't reproduce it.)"""
    global _fear_greed_days
    if _fear_greed_days is None:
        data, error = _cmc_get(FEAR_GREED_HISTORY_URL, {"start": 1, "limit": 500})
        if error:
            raise ValueError(error["error"])
        _fear_greed_days = [{**d, "timestamp": format_ts(datetime.fromtimestamp(int(d["timestamp"]), timezone.utc))}
                            for d in data]
    found = _latest_at(_fear_greed_days, t - timedelta(days=1), lambda d: d["timestamp"])
    if found is None or t - parse_ts(found["timestamp"]) > timedelta(days=2):
        raise ValueError(f"no fear and greed value published by {format_ts(t)}")
    return _shape_fear_greed(found["value"], found["value_classification"], found["timestamp"])


def ohlcv_at(symbol, t, count=HISTORY_DAYS):
    """get_ohlcv's response at t: the last `count` daily candles closed by t."""
    data, error = _cmc_get(OHLCV_URL, {"symbol": symbol, "time_period": "daily",
                                       "time_start": format_ts(t - timedelta(days=count + 2)),
                                       "time_end": format_ts(t)})
    if error:
        raise ValueError(error["error"])
    entries = (data or {}).get(symbol)
    entry = entries[0] if isinstance(entries, list) else entries
    closed = [q for q in (entry or {}).get("quotes", []) if parse_ts(q["time_close"]) <= t][-count:]
    opens = [parse_ts(q["time_open"]) for q in closed]
    if len(closed) != count or any((b - a) != timedelta(days=1) for a, b in zip(opens, opens[1:])):
        raise ValueError(f"CMC returned {len(closed)} consecutive closed daily candles for {symbol}, need {count}")
    return {"symbol": symbol, "candles": [_shape_candle(q) for q in closed]}


def market_tool_snapshots(fetched_at):
    """Snapshots of the market-wide and OHLCV tools as of fetched_at, each
    aged relative to that instant like every other snapshot."""
    snaps = [make_snapshot("get_market_overview", None, market_overview_at(fetched_at), fetched_at),
             make_snapshot("get_fear_greed", None, fear_greed_at(fetched_at), fetched_at)]
    snaps += [make_snapshot("get_ohlcv", s, ohlcv_at(s, fetched_at), fetched_at) for s in SYMBOLS]
    return snaps


def scenario_instant(directory):
    """The moment a scenario's quote was fetched (or, for a rebuilt one, its
    pinned 'fetch'): the instant every other tool must be consistent with."""
    snap = json.loads((directory / "btc_price.json").read_text(encoding="utf-8"))
    return parse_ts(snap["response"]["last_updated"]) + timedelta(seconds=snap["recorded_at_age_seconds"])


def add_market_tools(names):
    """Add the market-wide and OHLCV tools to existing scenarios, leaving
    their price and history files untouched. All-or-nothing per scenario."""
    dirs = [SNAPSHOT_ROOT / n for n in names] if names else sorted(d for d in SNAPSHOT_ROOT.iterdir() if d.is_dir())
    missing = [d.name for d in dirs if not (d / "btc_price.json").is_file()]
    if missing:
        print(f"not scenarios (no btc_price.json): {missing}")
        return 1
    failed = 0
    for d in dirs:
        at = scenario_instant(d)
        try:
            snaps = market_tool_snapshots(at)
        except (ValueError, KeyError, TypeError) as e:
            print(f"failed: {d.name}: {e}; nothing written for this scenario.")
            failed += 1
            continue
        for snap in snaps:
            path = snapshot_path(snap["tool"], snap["symbol"], d)
            path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        meta_path = d / "scenario.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["market_tools"] = {"added": datetime.now(timezone.utc).isoformat(), "as_of": format_ts(at),
                                "rule": "latest data that existed at as_of (see record_snapshots.py)"}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        ages = {f"{x['tool']}({x['symbol'] or ''})": round(x["recorded_at_age_seconds"]) for x in snaps}
        print(f"{d.name}: {ages}")
    return 1 if failed else 0


def record_historical(names):
    unknown = [n for n in names if n not in HISTORICAL]
    if unknown:
        print(f"unknown scenarios: {unknown}; known: {sorted(HISTORICAL)}")
        return 1
    failed = 0
    for name in names or HISTORICAL:
        date, description = HISTORICAL[name]
        try:
            snapshots = [s for symbol in SYMBOLS for s in historical_snapshots(date, symbol)]
            at = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            snapshots += market_tool_snapshots(at + timedelta(seconds=HISTORICAL_QUOTE_AGE_SECONDS))
        except (ValueError, KeyError, TypeError) as e:
            print(f"failed: {name}: {e}; nothing written for this scenario.")
            failed += 1
            continue
        write_scenario(name, snapshots, {
            "kind": "historical", "as_of": f"{date}T00:00:00Z", "description": description,
            "quote_age_seconds": HISTORICAL_QUOTE_AGE_SECONDS,
            "source": "CMC v3 daily historical quotes; the quote is the daily data point at as_of.",
        })
    return 1 if failed else 0


if __name__ == "__main__":
    mode, *rest = sys.argv[1:] or [""]
    if mode == "live":
        sys.exit(record_live())
    if mode == "historical":
        sys.exit(record_historical(rest))
    if mode == "add-tools":
        sys.exit(add_market_tools(rest))
    print(__doc__)
    sys.exit(1)
