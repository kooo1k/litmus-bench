"""snapshots.py replay and record_snapshots.py recording. The CMC API is
replaced by a fake; recordings go to tmp_path, never tasks/snapshots/."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from litmus import agent
from scripts import record_snapshots
from litmus.snapshots import (SNAPSHOT_ROOT, anchor_timestamp, format_ts, load_snapshot, make_snapshot, parse_ts,
                       replay_tools)

SCENARIOS = sorted(p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir())
FILES = ["btc_history.json", "btc_ohlcv.json", "btc_price.json", "eth_history.json", "eth_ohlcv.json",
         "eth_price.json", "fear_greed.json", "market_overview.json", "scenario.json"]


def instant(scenario):
    return record_snapshots.scenario_instant(scenario)


# --------------------------------------------------------------------------
# Replay: every scenario serves every tool
# --------------------------------------------------------------------------

def test_nine_scenarios_each_with_every_tool():
    assert len(SCENARIOS) == 9
    for d in SCENARIOS:
        assert sorted(p.name for p in d.iterdir()) == FILES, d


def test_replay_serves_every_agent_tool_with_the_same_signature():
    import inspect
    replay = replay_tools(SCENARIOS[0])
    assert set(replay) == set(agent.TOOL_FUNCTIONS)
    for name, fn in agent.TOOL_FUNCTIONS.items():
        assert inspect.signature(replay[name]) == inspect.signature(fn), name


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[d.name for d in SCENARIOS])
def test_replay_serves_the_recorded_age(scenario):
    tools = replay_tools(scenario)
    now = datetime.now(timezone.utc)
    for path in sorted(scenario.glob("*.json")):
        if path.name == "scenario.json":
            continue
        snap = json.loads(path.read_text())
        name, sym = snap["tool"], snap["symbol"]
        result = tools[name](sym) if sym else tools[name]()
        age = (now - anchor_timestamp(name, result)).total_seconds()
        assert age == pytest.approx(snap["recorded_at_age_seconds"], abs=2), path.name
    for sym in ("BTC", "ETH"):
        stamps = [parse_ts(q["timestamp"]) for q in tools["get_price_history"](sym)["quotes"]]
        assert len(stamps) == 7 and {(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])} == {86400}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[d.name for d in SCENARIOS])
def test_no_snapshot_holds_data_from_after_its_scenario_instant(scenario):
    """Look-ahead guard: everything a scenario serves existed at its instant."""
    t = instant(scenario)
    for path in scenario.glob("*.json"):
        if path.name == "scenario.json":
            continue
        snap = json.loads(path.read_text())
        assert anchor_timestamp(snap["tool"], snap["response"]) <= t, path.name
    fg = json.loads((scenario / "fear_greed.json").read_text())["response"]
    assert parse_ts(fg["last_updated"]) <= t - timedelta(days=1)  # published only after its day ends


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[d.name for d in SCENARIOS])
def test_tools_agree_on_price_within_the_grounding_tolerance(scenario):
    """A top-level disagreement between tools would be an accidental
    contradiction in every task that uses the scenario."""
    tools = replay_tools(scenario)
    for sym in ("BTC", "ETH"):
        quote = tools["get_price"](sym)["price_usd"]
        candles = tools["get_ohlcv"](sym)["candles"]
        history = [q["quote"]["USD"]["price"] for q in tools["get_price_history"](sym)["quotes"]]
        assert abs(candles[-1]["close"] / quote - 1) < 0.005
        assert all(abs(c["open"] / h - 1) < 0.005 for c, h in zip(candles, history))


def test_historical_scenarios_pin_quote_age_at_30s():
    for d in SCENARIOS:
        meta = json.loads((d / "scenario.json").read_text())
        if meta["kind"] == "historical":
            assert json.loads((d / "btc_price.json").read_text())["recorded_at_age_seconds"] == 30


def test_replay_at_the_recording_instant_is_an_exact_round_trip():
    path = SNAPSHOT_ROOT / "2026-09-25_live" / "btc_price.json"
    snap = json.loads(path.read_text())
    t = parse_ts(snap["response"]["last_updated"]) + timedelta(seconds=snap["recorded_at_age_seconds"])
    assert load_snapshot(path, now=t) == snap["response"]


def test_replay_does_not_drift_with_wall_clock():
    path = SNAPSHOT_ROOT / "2026-09-25_live" / "btc_price.json"
    now = datetime.now(timezone.utc)
    ages = [(n - parse_ts(load_snapshot(path, now=n)["last_updated"])).total_seconds()
            for n in (now, now + timedelta(days=180))]
    assert ages[0] == pytest.approx(ages[1])


def test_unrecorded_symbol_looks_like_an_unlisted_one():
    tools = replay_tools(SCENARIOS[0])
    assert tools["get_price"]("sol") == {"error": "symbol 'SOL' not found"}
    assert tools["get_price_history"]("SOL") == {"error": "no history found for symbol 'SOL'"}
    assert tools["get_ohlcv"]("SOL") == {"error": "no OHLCV found for symbol 'SOL'"}


def test_count_trims_to_the_newest_days():
    tools = replay_tools(SCENARIOS[0])
    assert len(tools["get_price_history"]("BTC", count=2)["quotes"]) == 2
    assert len(tools["get_ohlcv"]("BTC", count=3)["candles"]) == 3


def test_make_snapshot_refuses_error_responses():
    with pytest.raises(ValueError, match="refusing to snapshot an error"):
        make_snapshot("get_price", "BTC", {"error": "x"}, datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Recording against a fake CMC. One fake serves every endpoint the recorder
# uses; `drop` removes dates, `error` fails every call.
# --------------------------------------------------------------------------

def fake_cmc(drop=(), error=None, drop_symbol=None, fear_greed_days=None):
    def day_points(start, end):
        t = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while t <= end:
            yield t
            t += timedelta(days=1)

    def _cmc_get(url, params):
        if error:
            return None, {"error": error}
        end = parse_ts(params["time_end"]) if "time_end" in params else None
        start = parse_ts(params["time_start"]) if "time_start" in params else None
        if url == record_snapshots.HISTORY_URL:
            sym = params["symbol"]
            quotes = []
            for t in day_points(end - timedelta(days=11), end):
                if t.date().isoformat() in drop and drop_symbol in (None, sym):
                    continue
                ts = format_ts(t)
                quotes.append({"timestamp": ts, "quote": {"USD": {"price": 100.0 + t.day, "percent_change_1h": 0.1,
                                                                   "percent_change_24h": 1.0, "timestamp": ts}}})
            return {sym: [{"id": 1, "name": sym, "symbol": sym, "quotes": quotes}]}, None
        if url == record_snapshots.GLOBAL_HISTORY_URL:
            daily = params["interval"] == "daily"
            step = timedelta(days=1) if daily else timedelta(minutes=5)
            # Real CMC points sit on the interval grid: midnight for daily, :00/:05 for 5m.
            grid = start.replace(hour=0, minute=0, second=0, microsecond=0) if daily else \
                start.replace(minute=start.minute - start.minute % 5, second=0, microsecond=0)
            t = grid if grid >= start else grid + step
            pts = []
            while t <= end:
                if t.date().isoformat() not in drop:
                    pts.append({"timestamp": format_ts(t), "btc_dominance": 58.0 + t.day / 100, "eth_dominance": 12.0,
                                "quote": {"USD": {"total_market_cap": 1e12 * (1 + t.day / 100), "total_volume_24h": 5e10,
                                                  "timestamp": format_ts(t)}}})
                t += step
            return {"quotes": pts}, None
        if url == record_snapshots.FEAR_GREED_HISTORY_URL:
            days = fear_greed_days or [datetime(2026, 2, d, tzinfo=timezone.utc) for d in range(1, 10)]
            return [{"timestamp": str(int(d.timestamp())), "value": d.day * 10, "value_classification": f"label{d.day}"}
                    for d in days if d.date().isoformat() not in drop], None
        if url == record_snapshots.OHLCV_URL:
            sym = params["symbol"]
            candles = []
            for t in day_points(start, end):
                if t.date().isoformat() in drop and drop_symbol in (None, sym):
                    continue
                candles.append({"time_open": format_ts(t), "time_close": format_ts(t + timedelta(days=1, milliseconds=-1)),
                                "quote": {"USD": {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 9.0}}})
            return {sym: [{"symbol": sym, "quotes": candles}]}, None
        raise AssertionError(f"unexpected CMC endpoint {url}")
    return _cmc_get


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    monkeypatch.setattr(record_snapshots, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    monkeypatch.setattr(record_snapshots, "_fear_greed_days", None)  # module-level cache
    return tmp_path / "snapshots"


def test_historical_scenario_shape(monkeypatch, recorder):
    monkeypatch.setattr(record_snapshots, "_cmc_get", fake_cmc())
    assert record_snapshots.record_historical(["2026-02-06_crash"]) == 0
    d = recorder / "2026-02-06_crash"
    assert sorted(p.name for p in d.iterdir()) == FILES
    quote = json.loads((d / "btc_price.json").read_text())
    history = json.loads((d / "btc_history.json").read_text())
    assert quote["response"]["last_updated"] == "2026-02-06T00:00:00.000Z" and quote["recorded_at_age_seconds"] == 30
    assert [q["timestamp"][:10] for q in history["response"]["quotes"]] == [
        f"2026-0{m}-{day:02d}" for m, day in ((1, 30), (1, 31), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5))]
    fg = json.loads((d / "fear_greed.json").read_text())["response"]
    assert fg["last_updated"] == "2026-02-05T00:00:00.000Z" and fg["value"] == 50  # the day BEFORE: no look-ahead
    candles = json.loads((d / "btc_ohlcv.json").read_text())["response"]["candles"]
    assert len(candles) == 7 and candles[-1]["time_open"] == "2026-02-05T00:00:00.000Z"  # closed by the instant
    ov = json.loads((d / "market_overview.json").read_text())["response"]
    assert ov["last_updated"] == "2026-02-06T00:00:00.000Z"
    assert ov["total_market_cap_change_24h_pct"] == pytest.approx((1.06 / 1.05 - 1) * 100)
    assert ov["btc_dominance_yesterday_pct"] == pytest.approx(58.05)


@pytest.mark.parametrize("fake", [
    fake_cmc(drop={"2026-02-03"}),                      # a history/OHLCV day missing
    fake_cmc(drop={"2026-02-06"}),                      # the quote day itself missing
    fake_cmc(drop={"2026-02-01"}, drop_symbol="ETH"),   # only one symbol short
    fake_cmc(drop={"2026-02-05"}),                      # no fear & greed value published in time
    fake_cmc(error="API returned HTTP 400: plan limit"),
], ids=["history-gap", "quote-gap", "eth-only-gap", "fear-greed-gap", "api-error"])
def test_incomplete_data_writes_nothing(monkeypatch, recorder, fake):
    monkeypatch.setattr(record_snapshots, "_cmc_get", fake)
    assert record_snapshots.record_historical(["2026-02-06_crash"]) == 1
    assert not (recorder / "2026-02-06_crash").exists()


def test_market_overview_uses_intraday_points_inside_the_window(monkeypatch):
    seen = []
    fake = fake_cmc()
    monkeypatch.setattr(record_snapshots, "_cmc_get", lambda url, p: (seen.append(p.get("interval")), fake(url, p))[1])
    t = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(days=3, minutes=2)
    ov = record_snapshots.market_overview_at(t)
    assert seen == ["5m", "5m"] and parse_ts(ov["last_updated"]) <= t
    seen.clear()
    record_snapshots.market_overview_at(datetime(2026, 2, 6, 0, 0, 30, tzinfo=timezone.utc) - timedelta(days=0))
    assert seen == ["daily", "daily"]


def test_market_overview_refuses_points_not_24h_apart(monkeypatch):
    monkeypatch.setattr(record_snapshots, "_cmc_get", fake_cmc(drop={"2026-02-05"}))
    with pytest.raises(ValueError, match="apart, not 24h"):
        record_snapshots.market_overview_at(datetime(2026, 2, 6, 0, 0, 30, tzinfo=timezone.utc))


def test_add_tools_leaves_existing_files_untouched(monkeypatch, recorder):
    d = recorder / "2026-02-06_crash"
    d.mkdir(parents=True)
    for name in ("btc_price.json", "eth_price.json", "btc_history.json", "eth_history.json", "scenario.json"):
        (d / name).write_bytes((SNAPSHOT_ROOT / "2026-02-06_crash" / name).read_bytes())
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.name != "scenario.json"}
    monkeypatch.setattr(record_snapshots, "_cmc_get", fake_cmc())
    assert record_snapshots.add_market_tools(["2026-02-06_crash"]) == 0
    assert {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in before} == before
    assert sorted(p.name for p in d.iterdir()) == FILES
    assert json.loads((d / "scenario.json").read_text())["market_tools"]["as_of"] == "2026-02-06T00:00:30.000Z"


def test_add_tools_is_all_or_nothing_per_scenario(monkeypatch, recorder):
    d = recorder / "2026-02-06_crash"
    d.mkdir(parents=True)
    for name in ("btc_price.json", "scenario.json"):
        (d / name).write_bytes((SNAPSHOT_ROOT / "2026-02-06_crash" / name).read_bytes())
    monkeypatch.setattr(record_snapshots, "_cmc_get", fake_cmc(drop={"2026-02-03"}))
    assert record_snapshots.add_market_tools(["2026-02-06_crash"]) == 1
    assert sorted(p.name for p in d.iterdir()) == ["btc_price.json", "scenario.json"]


def test_unknown_scenario_name(recorder):
    assert record_snapshots.record_historical(["nope"]) == 1


def test_live_recording_refuses_to_overwrite_todays_directory(monkeypatch, recorder):
    today = recorder / f"{datetime.now(timezone.utc):%Y-%m-%d}_live"
    today.mkdir(parents=True)
    called = []
    monkeypatch.setattr(record_snapshots, "get_price", lambda s: called.append(s))
    assert record_snapshots.record_live() == 1 and called == []  # refused before any fetch
