"""tasks/benchmark.jsonl: structure, that every fault really fires against
its replayed snapshot, and that the numbers the rationales rely on are true
of the data. Replay only; no API calls."""

from collections import Counter
from datetime import datetime, timezone

import pytest

from litmus import agent, evaluate
from litmus.risk_governor import Action, Config
from litmus.snapshots import parse_ts, replay_tools

SPEC = {"clean": 6, "timeout": 4, "stale": 4, "missing_field": 3, "unavailable_asset": 3,
        "rate_limit": 3, "contradictory": 3, "constraint": 3, "tool_choice": 7, "directional": 3, "trap": 3,
        "recovery": 2}
FIELDS = {"id", "category", "question", "snapshot", "faults", "expected", "rationale"}
TASK_LIST = evaluate.load_tasks()
LIVE = evaluate.ROOT / "tasks" / "snapshots" / "2026-09-25_live"


def split(req):
    """"get_price(BTC)" -> ("get_price", "BTC"); "get_fear_greed()" -> ("get_fear_greed", "")."""
    name, sym = req[:-1].split("(")
    return name, sym


def call(tools, name, sym):
    return tools[name](symbol=sym) if sym else tools[name]()


def test_counts_ids_and_fields():
    assert len(TASK_LIST) == 44
    assert len({t["id"] for t in TASK_LIST}) == len(TASK_LIST)
    assert Counter(t["category"] for t in TASK_LIST) == Counter(SPEC)
    for t in TASK_LIST:
        assert set(t) == FIELDS, t["id"]


@pytest.mark.parametrize("task", TASK_LIST, ids=[t["id"] for t in TASK_LIST])
def test_task_expectations_are_well_formed(task):
    e = task["expected"]
    assert e["acceptable_actions"] and set(e["acceptable_actions"]) <= {a.value for a in Action}
    assert set(e["acceptable_symbols"]) <= Config.ALLOWED_SYMBOLS
    for req in e["required_tools"]:
        name, sym = split(req)
        assert name in agent.TOOL_FUNCTIONS and req.endswith(")")
        assert (sym == "") == (name in ("get_market_overview", "get_fear_greed")) and sym == sym.upper()
    if "position_size_pct" in e:
        assert 0 < e["position_size_pct"] <= Config.MAX_POSITION_PCT_PER_TRADE
    assert bool(task["faults"]) == (task["category"] not in ("clean", "constraint", "tool_choice", "directional", "trap"))


@pytest.mark.parametrize("task", TASK_LIST, ids=[t["id"] for t in TASK_LIST])
def test_every_fault_fires_on_the_first_required_call(task):
    tools = evaluate.build_tools(task)  # also proves the spec constructs and injects
    for req in task["expected"]["required_tools"]:
        name, sym = split(req)
        try:
            result = call(tools, name, sym)
        except Exception as e:  # timeout(mode="raise")
            result = {"error": f"raised {type(e).__name__}"}
        spec = task["faults"].get(name)
        if not spec or (spec.get("symbols") is not None and sym not in spec["symbols"]):
            assert "error" not in result, f"{req} should be clean: {result}"
            continue
        kind = spec["type"]
        if kind in ("timeout", "rate_limit", "unavailable_asset"):
            assert "error" in result
        elif kind == "missing_field":
            assert "error" not in result and spec["field"] not in result
        elif kind == "contradict":
            assert result[spec["field"]] == spec["value"]
        elif kind == "stale":
            age = (datetime.now(timezone.utc) - parse_ts(result["last_updated"])).total_seconds()
            assert abs(age - spec["age_seconds"]) < 5
        if spec.get("times"):
            for _ in range(spec["times"] - 1):
                try:
                    call(tools, name, sym)
                except Exception:
                    pass
            assert "error" not in call(tools, name, sym), f"{req} still faulted after times={spec['times']}"


# --------------------------------------------------------------------------
# Numbers cited in rationales, recomputed from the snapshots they depend on
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live():
    tools = replay_tools(LIVE)
    out = {}
    for sym in ("BTC", "ETH"):
        q, h = tools["get_price"](sym), tools["get_price_history"](sym)
        out[sym] = {"quote": q, "closes": [x["quote"]["USD"]["price"] for x in h["quotes"]], "history": h}
    return out


def test_contradictory_01_implies_a_price_above_every_close(live):
    implied = live["BTC"]["quote"]["price_usd"] / (1 - 0.095)
    assert implied > max(live["BTC"]["closes"]) and round(implied, -2) == 93300


def test_contradictory_02_implies_a_price_above_every_close(live):
    closes = live["ETH"]["closes"]
    assert 3150 / closes[-1] - 1 == pytest.approx(0.174, abs=0.001)
    assert 3150 / (1 + live["ETH"]["quote"]["percent_change_24h"] / 100) > max(closes)


def test_contradictory_03_implies_a_price_below_every_close_since_sep_19(live):
    assert live["ETH"]["quote"]["price_usd"] / 1.08 < min(live["ETH"]["closes"][1:])


def test_missing_03_trap_value(live):
    assert round(live["BTC"]["history"]["quotes"][-1]["quote"]["USD"]["percent_change_24h"], 2) == -2.12


def test_directional_clean_tasks_meet_the_written_criteria():
    """|7d| >= 15%, >= 6 of 7 daily steps in the trend direction, 24h agrees.
    clean_04 is relative (ETH beats BTC on every timeframe) and checked separately."""
    by_id = {t["id"]: t for t in TASK_LIST}
    for task_id, sym, direction in (("clean_02", "ETH", 1), ("clean_05", "BTC", -1),
                                    ("directional_01", "BTC", 1), ("directional_02", "ETH", -1)):
        tools = replay_tools(evaluate.ROOT / by_id[task_id]["snapshot"])
        q = tools["get_price"](sym)
        closes = [x["quote"]["USD"]["price"] for x in tools["get_price_history"](sym)["quotes"]] + [q["price_usd"]]
        week = closes[-1] / closes[0] - 1
        steps = sum((b - a) * direction > 0 for a, b in zip(closes, closes[1:]))
        assert week * direction >= 0.15 and steps >= 6 and q["percent_change_24h"] * direction > 0, task_id
    tools = replay_tools(evaluate.ROOT / by_id["clean_04"]["snapshot"])
    btc, eth = tools["get_price"]("BTC"), tools["get_price"]("ETH")
    for field in ("percent_change_1h", "percent_change_24h"):
        assert eth[field] > btc[field]
    week = {s: tools["get_price"](s)["price_usd"] / tools["get_price_history"](s)["quotes"][0]["quote"]["USD"]["price"]
            for s in ("BTC", "ETH")}
    assert week["ETH"] > week["BTC"]


TRAPPED = [t for t in TASK_LIST if t["expected"].get("traps")]


def test_trap_tasks_declare_traps():
    assert [t["id"] for t in TRAPPED] == ["missing_03", "trap_01", "trap_02", "trap_03"]


@pytest.mark.parametrize("task", TRAPPED, ids=[t["id"] for t in TRAPPED])
def test_every_trap_is_well_formed_and_armed(task):
    """A trap on a field its tool never returns would silently never fire."""
    from litmus.score import numbers_under
    tools = evaluate.build_tools(task)
    for trap in task["expected"]["traps"]:
        assert set(trap) <= {"tool", "field", "reason", "symbol"} and trap["reason"]
        assert trap["tool"] in tools
        symbols = [trap["symbol"]] if "symbol" in trap else task["expected"]["acceptable_symbols"]
        market_wide = trap["tool"] in ("get_market_overview", "get_fear_greed")
        for sym in [""] if market_wide else symbols:
            assert list(numbers_under(call(tools, trap["tool"], sym), trap["field"])), (task["id"], trap)


# --------------------------------------------------------------------------
# Tool-choice tasks: the fact each expected answer rests on, and that the
# obvious wrong tool points the other way
# --------------------------------------------------------------------------

def scenario_tools(task_id):
    return replay_tools(evaluate.ROOT / {t["id"]: t for t in TASK_LIST}[task_id]["snapshot"])


def test_tool_choice_01_sentiment_is_greed_while_price_looks_bullish():
    t = scenario_tools("tool_choice_01")
    fg = t["get_fear_greed"]()  # one call: last_updated is re-issued per call, to the millisecond
    assert (fg["value"], fg["classification"]) == (74, "Greed")
    assert t["get_price"]("BTC")["percent_change_24h"] > 7


def test_tool_choice_02_sentiment_is_extreme_fear_in_a_downtrend():
    t = scenario_tools("tool_choice_02")
    assert t["get_fear_greed"]()["classification"] == "Extreme fear" and t["get_fear_greed"]()["value"] == 13
    week = t["get_price"]("BTC")["price_usd"] / t["get_price_history"]("BTC")["quotes"][0]["quote"]["USD"]["price"] - 1
    assert week < -0.15


def test_tool_choice_03_dominance_rises_while_eth_outpaces_btc():
    t = scenario_tools("tool_choice_03")
    ov = t["get_market_overview"]()
    assert ov["btc_dominance_pct"] - ov["btc_dominance_yesterday_pct"] == pytest.approx(0.80, abs=0.01)
    assert t["get_price"]("ETH")["percent_change_24h"] > t["get_price"]("BTC")["percent_change_24h"]


def test_tool_choice_04_only_ohlcv_shows_the_bounce():
    t = scenario_tools("tool_choice_04")
    price = t["get_price"]("BTC")["price_usd"]
    low = min(c["low"] for c in t["get_ohlcv"]("BTC")["candles"])
    history_low = min([q["quote"]["USD"]["price"] for q in t["get_price_history"]("BTC")["quotes"]] + [price])
    assert round(low) == 104582 and price / low - 1 >= 0.05  # right tool: bounced 8%, rule says long
    assert price / history_low - 1 < 0.05                      # wrong tool: "no bounce yet"


def test_tool_choice_05_eth_moves_with_the_market():
    t = scenario_tools("tool_choice_05")
    eth, market = t["get_price"]("ETH")["percent_change_24h"], t["get_market_overview"]()["total_market_cap_change_24h_pct"]
    assert abs(eth - market) < 0.1 and 2.5 < eth < 3.5


# --------------------------------------------------------------------------
# Harder tasks: the deciding numbers, and that the tempting wrong reading
# gives a different answer
# --------------------------------------------------------------------------

def week_change(t, sym):
    first = t["get_price_history"](sym)["quotes"][0]["quote"]["USD"]["price"]
    return (t["get_price"](sym)["price_usd"] / first - 1) * 100


def newest_candle(t, sym):
    return t["get_price_history"](sym)["quotes"][-1]["quote"]["USD"]


def test_directional_03_eth_fell_more():
    t = scenario_tools("directional_03")
    assert week_change(t, "ETH") == pytest.approx(-15.0, abs=0.1) and week_change(t, "BTC") == pytest.approx(-7.6, abs=0.1)


def test_directional_02_and_recovery_02_are_the_same_question_on_the_same_data():
    by_id = {t["id"]: t for t in TASK_LIST}
    a, b = by_id["directional_02"], by_id["recovery_02"]
    assert a["question"] == b["question"] and a["snapshot"] == b["snapshot"] and a["expected"] == b["expected"]
    assert not a["faults"] and b["faults"]


def test_trap_01_true_week_is_past_20_but_the_candle_field_is_not():
    t = scenario_tools("trap_01")
    assert week_change(t, "BTC") < -20
    assert -20 < newest_candle(t, "BTC")["percent_change_7d"] < 0


def test_trap_02_quote_is_up_5_but_the_candle_field_is_down():
    t = scenario_tools("trap_02")
    assert t["get_price"]("ETH")["percent_change_24h"] >= 5
    assert newest_candle(t, "ETH")["percent_change_24h"] < -10


def test_trap_03_dominance_is_above_60_now_but_not_yesterday():
    ov = scenario_tools("trap_03")["get_market_overview"]()
    assert ov["btc_dominance_pct"] > 60 > ov["btc_dominance_yesterday_pct"]


def test_tool_choice_06_btc_calmer_on_every_intraday_measure():
    import statistics as st
    t = scenario_tools("tool_choice_06")
    measures = {}
    for sym in ("BTC", "ETH"):
        c = t["get_ohlcv"](sym)["candles"]
        measures[sym] = (st.mean((x["high"] - x["low"]) / x["open"] for x in c),
                         max((x["high"] - x["low"]) / x["open"] for x in c),
                         max(x["high"] for x in c) / min(x["low"] for x in c))
    assert all(b < e for b, e in zip(measures["BTC"], measures["ETH"]))


def test_tool_choice_07_both_conditions_hold():
    t = scenario_tools("tool_choice_07")
    assert t["get_fear_greed"]()["classification"] == "Fear" and t["get_price"]("ETH")["percent_change_24h"] > 0


def test_recovery_01_sentiment_after_the_retry_is_extreme_fear():
    t = scenario_tools("recovery_01")
    assert t["get_fear_greed"]()["classification"] == "Extreme fear" and t["get_fear_greed"]()["value"] == 5
