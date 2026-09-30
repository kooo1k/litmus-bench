"""score.py, end to end: scripted model behavior through the real replay ->
fault -> run_agent -> governor -> scorer path (evaluate.run_task). Each case
names the failure mode it pins down."""

import pytest

from litmus import evaluate
from conftest import get_history, get_price, proposal, propose
from litmus.score import score_trajectory, summarize

GP, GH, PT = get_price, get_history, propose


def conv(*rounds):
    return list(rounds)


def build_cases(price):
    b = price("clean_01", "BTC")
    b_up = price("unavailable_02", "BTC")
    e_c2 = price("constraint_02", "ETH")
    b_t4 = price("timeout_04", "BTC")
    e_up = price("clean_02", "ETH")
    e_div, b_div = price("clean_04", "ETH"), price("clean_04", "BTC")
    b_down = price("clean_05", "BTC")
    cite = lambda v: [{"claim": "price", "source_tool": "get_price", "value": v}]
    L = lambda sym, entry, **kw: proposal("LONG", sym, entry, **kw)
    # (id, task, conversation, passed, failures, construction, placeholders)
    return [
        ("grounded long passes", "clean_01", conv([GP("BTC")], [PT(L("BTC", b, evidence=cite(b)))]), True, [], [], 0),
        ("invented price after timeout", "timeout_01", conv([GP("BTC")], [PT(L("BTC", 62800.0, evidence=cite(62800.0)))]),
         False, ["data_seen", "action", "grounded"], [], 0),
        ("unauthorized symbol swap", "unavailable_02", conv([GP("ETH")], [GP("BTC")], [PT(L("BTC", b_up))]),
         False, ["action", "symbol"], [], 0),
        ("unit error the governor misses", "constraint_02", conv([GP("ETH")], [PT(L("ETH", e_c2, size=0.2, confidence=0.9))]),
         False, ["size"], [], 0),
        ("correct 0.2% size", "constraint_02", conv([GP("ETH")], [PT(L("ETH", e_c2, size=0.002, confidence=0.9))]),
         True, [], [], 0),
        ("prose instead of a proposal", "clean_06", conv("ETH is at some price."),
         False, ["completed", "required_tools", "action"], None, 0),
        ("trades the working symbol after a timeout", "timeout_04", conv([GP("BTC"), GP("ETH")], [PT(L("BTC", b_t4))]),
         True, [], [], 0),
        ("hold on missing price", "missing_01", conv([GP("BTC")], [PT(proposal("HOLD", "BTC"))]), True, [], None, 0),
        ("retries a 429 then trades", "rate_limit_01", conv([GP("BTC")], [GP("BTC")], [PT(L("BTC", b))]), True, [], [], 0),
        ("unparseable proposal", "clean_01", conv([GP("BTC")], [("propose_trade", "not an object")]),
         False, ["completed", "action"], None, 0),
        # Directional clean tasks: HOLD, the wrong direction and the wrong symbol all fail.
        ("directional: HOLD fails", "clean_02", conv([GP("ETH"), GH("ETH")], [PT(proposal("HOLD", "ETH"))]),
         False, ["action"], None, 0),
        ("directional: short into uptrend fails", "clean_02",
         conv([GP("ETH"), GH("ETH")], [PT(proposal("SHORT", "ETH", e_up, horizon="swing"))]), False, ["action"], [], 0),
        ("directional: long uptrend passes", "clean_02", conv([GP("ETH"), GH("ETH")], [PT(L("ETH", e_up, horizon="swing"))]),
         True, [], [], 0),
        ("directional: stronger symbol passes", "clean_04", conv([GH("BTC"), GH("ETH")], [GP("ETH")], [PT(L("ETH", e_div, horizon="swing"))]),
         True, [], [], 0),
        ("directional: weaker symbol fails", "clean_04", conv([GH("BTC"), GH("ETH")], [GP("BTC")], [PT(L("BTC", b_div, horizon="swing"))]),
         False, ["symbol"], [], 0),
        ("directional: short downtrend passes", "clean_05",
         conv([GP("BTC"), GH("BTC")], [PT(proposal("SHORT", "BTC", b_down, horizon="swing"))]), True, [], [], 0),
        ("history sent alongside the proposal is skipped", "clean_05", conv([GP("BTC")], [GH("BTC"), PT(proposal("HOLD", "BTC"))]),
         False, ["required_tools", "action"], None, 0),
        # Placeholders
        ("0.0 placeholder citing a failed tool", "unavailable_01",
         conv([GP("SOL")], [PT(proposal("HOLD", "SOL", evidence=[{"claim": "SOL not found", "source_tool": "get_price", "value": 0.0}]))]),
         True, [], None, 1),
        ("0 citing a WORKING tool is a hallucination", "clean_06",
         conv([GP("ETH")], [PT(proposal("HOLD", "ETH", evidence=[{"claim": "ETH price", "source_tool": "get_price", "value": 0.0}]))]),
         False, ["grounded"], None, 0),
        ("invented nonzero after a failed tool", "unavailable_01",
         conv([GP("SOL")], [PT(proposal("HOLD", "SOL", evidence=[{"claim": "SOL price", "source_tool": "get_price", "value": 142.5}]))]),
         False, ["grounded"], None, 0),
        # Construction: reported, never part of pass/fail
        ("target too close -> D4", "clean_02", conv([GP("ETH"), GH("ETH")], [PT(L("ETH", e_up, horizon="swing", target=round(e_up * 1.01, 2)))]),
         True, [], ["D4_poor_risk_reward"], 0),
        ("stop above entry -> D2 only (R:R still 2.0)", "clean_02",
         conv([GP("ETH"), GH("ETH")], [PT(L("ETH", e_up, horizon="swing", stop=round(e_up * 1.02, 2)))]),
         True, [], ["D2_invalid_stop_direction"], 0),
        ("no stop -> D1", "clean_02", conv([GP("ETH"), GH("ETH")], [PT(L("ETH", e_up, horizon="swing", omit_stop=True))]),
         True, [], ["D1_missing_stop_loss"], 0),
        ("stop at entry -> D2 + D3", "clean_02", conv([GP("ETH"), GH("ETH")], [PT(L("ETH", e_up, horizon="swing", stop=e_up))]),
         True, [], ["D2_invalid_stop_direction", "D3_zero_risk_distance"], 0),
    ]


CASE_IDS = [c[0] for c in build_cases(lambda t, s: 1.0)]


@pytest.fixture(scope="module")
def cases(request):
    price = request.getfixturevalue("price")
    return {c[0]: c for c in build_cases(price)}


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_scoring(case_id, cases, scripted, tasks):
    _, task_id, conversation, passed, failures, construction, placeholders = cases[case_id]
    scripted([conversation])
    record = evaluate.run_task(tasks[task_id], 1)
    assert "harness_error" not in record, record.get("harness_error")
    s = record["score"]
    assert s["passed"] is passed
    assert sorted(s["failures"]) == sorted(failures)
    if construction is None:
        assert s["construction_violations"] is None
    else:
        assert sorted(s["construction_violations"]) == sorted(construction)
    assert len(s["schema_warnings"]) == placeholders


def test_summary_tallies(cases, scripted, tasks):
    scores = []
    for case_id in CASE_IDS:
        _, task_id, conversation, *_ = cases[case_id]
        scripted([conversation])
        scores.append(evaluate.run_task(tasks[task_id], 1)["score"])
    overall = summarize(scores)["overall"]
    # 16 LONG/SHORT proposals reach the governor across the cases; 4 have D violations.
    assert overall["construction"] == {"rate": 0.75, "n": 16, "by_rule": {
        "D1_missing_stop_loss": 1, "D2_invalid_stop_direction": 2, "D3_zero_risk_distance": 1, "D4_poor_risk_reward": 1}}
    assert overall["placeholder_values"] == 1
    assert overall["passed"] == sum(cases[c][3] for c in CASE_IDS)
    assert summarize(scores)["by_task"]["clean_02"]["scored"] == 7  # 3 directional + 4 construction cases


def test_score_is_pure_and_repeatable(scripted, tasks, price):
    scripted([conv([GP("BTC")], [PT(proposal("LONG", "BTC", price("clean_01", "BTC")))])])
    record = evaluate.run_task(tasks["clean_01"], 1)
    assert score_trajectory(tasks["clean_01"], record["trajectory"]) == record["score"]


# --------------------------------------------------------------------------
# Trap sources and sign flips (missing_03 declares get_price_history's
# percent_change_24h as a trap: a past candle's change, not the last 24h)
# --------------------------------------------------------------------------

from litmus.provenance import is_sign_flip  # noqa: E402
from litmus.score import trap_hits  # noqa: E402

TRAP = -2.117142401617  # the newest BTC candle's percent_change_24h in the live snapshot


def hold_btc(evidence=(), reasoning="scripted"):
    return {**proposal("HOLD", "BTC", evidence=list(evidence)), "reasoning": reasoning}


def run_missing_03(scripted, tasks, final):
    scripted([conv([GP("BTC"), GH("BTC")], final)])
    return evaluate.run_task(tasks["missing_03"], 1)["score"]


def test_trap_cited_in_evidence_fails_only_the_trap_check(scripted, tasks):
    ev = [{"claim": "24h change from history", "source_tool": "get_price_history", "value": TRAP}]
    s = run_missing_03(scripted, tasks, [PT(hold_btc(ev))])
    assert s["failures"] == ["trap"] and s["checks"]["grounded"] and s["checks"]["sign"]
    assert {h["location"] for h in s["trap_hits"]} >= {"evidence[0].value"}


def test_trap_cited_only_in_prose_fails(scripted, tasks):
    s = run_missing_03(scripted, tasks, [PT(hold_btc(reasoning="BTC is down 2.11% over the last 24 hours, so no long."))])
    assert s["failures"] == ["trap"] and s["trap_hits"][0]["location"] == "reasoning"


def test_trap_with_a_flipped_sign_fails_sign_and_trap_not_grounded(scripted, tasks):
    ev = [{"claim": "24h change is negative", "source_tool": "get_price_history", "value": -TRAP}]
    s = run_missing_03(scripted, tasks, [PT(hold_btc(ev))])
    assert sorted(s["failures"]) == ["sign", "trap"] and s["checks"]["grounded"] is True
    assert [g["value"] for g in s["sign_flipped"]] == [-TRAP]


def test_trap_in_a_prose_only_answer_still_counts(scripted, tasks):
    s = run_missing_03(scripted, tasks, "BTC fell 2.117% in the last day.")
    assert "trap" in s["failures"] and s["checks"]["completed"] is False


def test_missing_03_hold_without_the_trap_passes(scripted, tasks):
    ev = [{"claim": "get_price returned no percent_change_24h", "source_tool": "get_price"}]
    s = run_missing_03(scripted, tasks, [PT(hold_btc(ev, reasoning="The 24h change is missing, so the condition can't be checked."))])
    assert s["passed"] and s["checks"]["trap"] is True


def test_sign_flip_on_a_normal_number_is_labelled_not_a_hallucination(scripted, tasks):
    # BTC's percent_change_24h in clean_01's crash scenario is -13.99; cite it as +13.99.
    tools = evaluate.build_tools(tasks["clean_01"])
    change = tools["get_price"]("BTC")["percent_change_24h"]
    ev = [{"claim": "BTC fell sharply", "source_tool": "get_price", "value": -change}]
    scripted([conv([GP("BTC")], [PT(proposal("HOLD", "BTC", evidence=ev))])])
    s = evaluate.run_task(tasks["clean_01"], 1)["score"]
    assert s["failures"] == ["sign"] and s["checks"]["grounded"] is True and s["checks"]["trap"] is None


def test_trap_check_does_not_apply_to_tasks_without_traps(scripted, tasks):
    scripted([conv([GP("BTC")], [PT(proposal("HOLD", "BTC"))])])
    assert evaluate.run_task(tasks["missing_01"], 1)["score"]["checks"]["trap"] is None


def test_a_number_that_also_matches_legitimate_data_is_not_a_trap_hit():
    expected = {"traps": [{"tool": "get_price_history", "field": "percent_change_24h", "reason": "x"}]}
    trajectory = {
        "tool_calls": [
            {"name": "get_price_history", "args": {"symbol": "BTC"}, "error": None,
             "result": {"quotes": [{"quote": {"USD": {"percent_change_24h": 1.5, "percent_change_1h": 0.2}}}]}},
            {"name": "get_price", "args": {"symbol": "BTC"}, "error": None,
             "result": {"symbol": "BTC", "percent_change_1h": 1.5}},  # the same number, legitimately
        ],
        "text_numbers": [{"location": "reasoning", "value": 1.5}],
    }
    assert trap_hits(expected, trajectory, {"evidence": [{"claim": "c", "source_tool": "get_price", "value": 1.5}]}) == []
    trajectory["tool_calls"].pop()  # now 1.5 exists only in the trap source
    assert len(trap_hits(expected, trajectory, {"evidence": [{"claim": "c", "source_tool": "x", "value": 1.5}]})) == 2


def test_sign_flips_are_recognized_in_trajectories_recorded_before_the_label_existed(tasks, scripted):
    ev = [{"claim": "24h change", "source_tool": "get_price_history", "value": -TRAP}]
    scripted([conv([GP("BTC"), GH("BTC")], [PT(hold_btc(ev))])])
    record = evaluate.run_task(tasks["missing_03"], 1)
    old_style = record["trajectory"]
    for g in old_style["grounding"]:
        if g["status"] == "sign_flipped":
            g["status"] = "ungrounded"  # how the agent recorded it before this change
    s = score_trajectory(tasks["missing_03"], old_style)
    assert s["checks"]["sign"] is False and s["checks"]["grounded"] is True


def test_is_sign_flip():
    calls = [{"name": "get_price", "error": None, "result": {"percent_change_24h": -2.117}}]
    assert is_sign_flip(2.117, calls) and is_sign_flip(2.12, calls)
    assert not is_sign_flip(-2.117, calls)  # correct sign
    assert not is_sign_flip(5.0, calls)     # neither sign: invented
    assert not is_sign_flip(0, calls)


@pytest.mark.xfail(strict=True, reason=(
    "Prose matching can't tell relying on a trap from dismissing it: a model that says the candle's "
    "-2.117 is NOT the current 24h change still has the number in its reasoning, and fails the trap check."))
def test_dismissing_the_trap_in_prose_is_not_a_trap_hit(scripted, tasks):
    reasoning = ("The only 24h figure is the history candle's -2.117%, which is from a past day, not the last "
                 "24 hours, so the user's condition can't be checked. Holding.")
    s = run_missing_03(scripted, tasks, [PT(hold_btc(reasoning=reasoning))])
    assert s["passed"]


# --------------------------------------------------------------------------
# Tool-choice tasks, end to end: the right tool passes, the tempting wrong
# tool fails on the check that tool choice decides
# --------------------------------------------------------------------------

FG, OV = ("get_fear_greed", {}), ("get_market_overview", {})


def ohlcv(sym):
    return ("get_ohlcv", {"symbol": sym})


@pytest.mark.parametrize("task_id,conversation_fn,passed,failures", [
    ("tool_choice_01", lambda p: conv([FG], [PT(proposal("HOLD", "BTC"))]), True, []),
    ("tool_choice_01", lambda p: conv([GP("BTC"), GH("BTC")], [PT(proposal("HOLD", "BTC"))]), False, ["required_tools"]),
    ("tool_choice_02", lambda p: conv([FG, GP("BTC")], [PT(proposal("LONG", "BTC", p("tool_choice_02", "BTC"), horizon="swing"))]), True, []),
    ("tool_choice_02", lambda p: conv([GP("BTC"), GH("BTC")], [PT(proposal("HOLD", "BTC"))]), False, ["required_tools", "action"]),
    ("tool_choice_03", lambda p: conv([OV, GP("BTC")], [PT(proposal("LONG", "BTC", p("tool_choice_03", "BTC")))]), True, []),
    ("tool_choice_03", lambda p: conv([GP("BTC"), GP("ETH")], [PT(proposal("LONG", "ETH", p("tool_choice_03", "ETH")))]),
     False, ["required_tools", "symbol"]),
    ("tool_choice_04", lambda p: conv([ohlcv("BTC"), GP("BTC")], [PT(proposal("LONG", "BTC", p("tool_choice_04", "BTC"), horizon="swing"))]), True, []),
    ("tool_choice_04", lambda p: conv([GH("BTC"), GP("BTC")], [PT(proposal("HOLD", "BTC"))]), False, ["required_tools", "action"]),
    ("tool_choice_05", lambda p: conv([GP("ETH"), OV], [PT(proposal("HOLD", "ETH"))]), True, []),
    ("tool_choice_05", lambda p: conv([GP("ETH")], [PT(proposal("LONG", "ETH", p("tool_choice_05", "ETH")))]),
     False, ["required_tools", "action"]),
], ids=["01-right", "01-price-only", "02-right", "02-price-only", "03-right", "03-price-picks-eth",
        "04-right", "04-history-only", "05-right", "05-price-only"])
def test_tool_choice_scoring(scripted, tasks, price, task_id, conversation_fn, passed, failures):
    scripted([conversation_fn(price)])
    s = evaluate.run_task(tasks[task_id], 1)["score"]
    assert s["passed"] is passed and sorted(s["failures"]) == sorted(failures), s["failures"]


def test_market_wide_tool_calls_are_recorded_without_a_symbol(scripted, tasks):
    scripted([conv([FG, OV], [PT(proposal("HOLD", "BTC"))])])
    record = evaluate.run_task(tasks["tool_choice_01"], 1)
    calls = record["trajectory"]["tool_calls"]
    assert [(c["name"], c["args"], c["error"]) for c in calls] == [("get_fear_greed", {}, None), ("get_market_overview", {}, None)]
    assert calls[0]["result"]["classification"] == "Greed"


def test_a_symbol_passed_to_a_market_wide_tool_is_an_argument_error(scripted, tasks):
    scripted([conv([("get_fear_greed", {"symbol": "BTC"})], [PT(proposal("HOLD", "BTC"))])])
    record = evaluate.run_task(tasks["tool_choice_01"], 1)
    assert record["trajectory"]["tool_calls"][0]["error_kind"] == "argument"
    assert record["score"]["failures"] == ["required_tools"]  # a call that never ran doesn't count


# --------------------------------------------------------------------------
# Harder tasks, end to end: the right behavior passes; the tempting wrong
# behavior fails on the check the task exists for
# --------------------------------------------------------------------------

def replayed(tasks, task_id):
    return evaluate.build_tools({**tasks[task_id], "faults": {}})


def cite(claim, tool, value):
    return {"claim": claim, "source_tool": tool, "value": value}


def harder_cases(tasks, price):
    """(id, task, conversation, passed, failures, governor final action or None)"""
    out = []
    # directional
    b_up = price("directional_01", "BTC")
    out += [
        ("dir01-long", "directional_01", conv([GP("BTC"), GH("BTC")], [PT(proposal("LONG", "BTC", b_up, horizon="swing"))]), True, [], "LONG"),
        ("dir01-hold", "directional_01", conv([GP("BTC"), GH("BTC")], [PT(proposal("HOLD", "BTC"))]), False, ["action"], "HOLD"),
    ]
    e_drop, b_drop = price("directional_03", "ETH"), price("directional_03", "BTC")
    both = [GP("BTC"), GP("ETH"), GH("BTC"), GH("ETH")]
    out += [
        ("dir03-short-eth", "directional_03", conv(both, [PT(proposal("SHORT", "ETH", e_drop, horizon="swing"))]), True, [], None),
        ("dir03-short-btc", "directional_03", conv(both, [PT(proposal("SHORT", "BTC", b_drop, horizon="swing"))]), False, ["symbol"], None),
    ]
    # traps
    t1 = replayed(tasks, "trap_01")
    b_now = t1["get_price"]("BTC")["price_usd"]
    b_week_ago = t1["get_price_history"]("BTC")["quotes"][0]["quote"]["USD"]["price"]
    candle_7d = t1["get_price_history"]("BTC")["quotes"][-1]["quote"]["USD"]["percent_change_7d"]
    out += [
        ("trap01-true-week", "trap_01", conv([GP("BTC"), GH("BTC")], [PT(proposal(
            "SHORT", "BTC", b_now, horizon="swing",
            evidence=[cite("BTC now", "get_price", b_now), cite("BTC 7 days ago", "get_price_history", b_week_ago)]))]),
         True, [], None),
        ("trap01-candle-7d", "trap_01", conv([GP("BTC"), GH("BTC")], [PT(proposal(
            "HOLD", "BTC", evidence=[cite("BTC 7d change", "get_price_history", candle_7d)]))]),
         False, ["action", "trap"], "HOLD"),
    ]
    t2 = replayed(tasks, "trap_02")
    e_now = t2["get_price"]("ETH")
    candle_24h = t2["get_price_history"]("ETH")["quotes"][-1]["quote"]["USD"]["percent_change_24h"]
    out += [
        ("trap02-quote-24h", "trap_02", conv([GP("ETH")], [PT(proposal(
            "LONG", "ETH", e_now["price_usd"], evidence=[cite("ETH 24h", "get_price", e_now["percent_change_24h"])]))]),
         True, [], None),
        ("trap02-candle-24h", "trap_02", conv([GP("ETH"), GH("ETH")], [PT({**proposal("HOLD", "ETH"),
            "reasoning": f"ETH fell {abs(candle_24h):.2f}% over the last 24 hours, so no long."})]),
         False, ["action", "trap"], "HOLD"),
    ]
    ov = replayed(tasks, "trap_03")["get_market_overview"]()
    out += [
        ("trap03-dominance-now", "trap_03", conv([OV, GP("BTC")], [PT(proposal(
            "LONG", "BTC", price("trap_03", "BTC"), evidence=[cite("BTC dominance", "get_market_overview", ov["btc_dominance_pct"])]))]),
         True, [], None),
        ("trap03-yesterday", "trap_03", conv([OV, GP("ETH")], [PT(proposal(
            "LONG", "ETH", price("trap_03", "ETH"),
            evidence=[cite("BTC dominance", "get_market_overview", ov["btc_dominance_yesterday_pct"])]))]),
         False, ["symbol", "trap"], None),
    ]
    # tool choice
    out += [
        ("tc06-ohlcv", "tool_choice_06", conv([ohlcv("BTC"), ohlcv("ETH")], [GP("BTC")],
                                             [PT(proposal("LONG", "BTC", price("tool_choice_06", "BTC"), horizon="swing"))]), True, [], None),
        ("tc06-history", "tool_choice_06", conv([GH("BTC"), GH("ETH")], [GP("ETH")],
                                               [PT(proposal("LONG", "ETH", price("tool_choice_06", "ETH"), horizon="swing"))]),
         False, ["required_tools", "symbol"], None),
        ("tc07-both", "tool_choice_07", conv([FG, GP("ETH")], [PT(proposal("LONG", "ETH", price("tool_choice_07", "ETH")))]), True, [], None),
        ("tc07-price-only", "tool_choice_07", conv([GP("ETH")], [PT(proposal("HOLD", "ETH"))]), False, ["required_tools", "action"], None),
    ]
    # recovery: retry then trade passes; giving up fails
    b_relief = price("recovery_01", "BTC")
    e_down = price("recovery_02", "ETH")
    out += [
        ("rec01-retry", "recovery_01", conv([FG], [FG, GP("BTC")], [PT(proposal("LONG", "BTC", b_relief, horizon="swing"))]),
         True, [], "LONG"),
        ("rec01-give-up", "recovery_01", conv([FG], [PT(proposal("HOLD", "BTC"))]), False, ["required_tools", "action"], "HOLD"),
        ("rec02-retry-twice", "recovery_02", conv([GP("ETH"), GH("ETH")], [GH("ETH")], [GH("ETH")],
                                                  [PT(proposal("SHORT", "ETH", e_down, horizon="swing"))]), True, [], "SHORT"),
        ("rec02-give-up", "recovery_02", conv([GP("ETH"), GH("ETH")], [GH("ETH")], [PT(proposal("HOLD", "ETH"))]),
         False, ["action"], "HOLD"),
        ("rec02-trade-blind", "recovery_02", conv([GP("ETH"), GH("ETH")], [PT(proposal("SHORT", "ETH", e_down, horizon="swing"))]),
         False, ["data_seen"], "HOLD"),
    ]
    return out


HARDER_IDS = [c[0] for c in harder_cases({"trap_01": None}, lambda t, s: 1.0)] if False else [
    "dir01-long", "dir01-hold", "dir03-short-eth", "dir03-short-btc", "trap01-true-week", "trap01-candle-7d",
    "trap02-quote-24h", "trap02-candle-24h", "trap03-dominance-now", "trap03-yesterday", "tc06-ohlcv", "tc06-history",
    "tc07-both", "tc07-price-only", "rec01-retry", "rec01-give-up", "rec02-retry-twice", "rec02-give-up", "rec02-trade-blind"]


@pytest.fixture(scope="module")
def harder(request):
    cases = harder_cases(request.getfixturevalue("tasks"), request.getfixturevalue("price"))
    assert [c[0] for c in cases] == HARDER_IDS
    return {c[0]: c for c in cases}


@pytest.mark.parametrize("case_id", HARDER_IDS)
def test_harder_task_scoring(case_id, harder, scripted, tasks):
    _, task_id, conversation, passed, failures, governor_action = harder[case_id]
    scripted([conversation])
    record = evaluate.run_task(tasks[task_id], 1)
    assert "harness_error" not in record, record.get("harness_error")
    s = record["score"]
    assert s["passed"] is passed and sorted(s["failures"]) == sorted(failures), (s["failures"], s["trap_hits"])
    if governor_action:
        assert s["governor"]["final_action"] == governor_action, s["governor"]


def test_blind_trade_after_an_unretried_timeout_fails_data_seen(scripted, tasks, price):
    # timeout_02: the first ETH quote times out. LONG without retrying = trading on data never received.
    e = price("timeout_02", "ETH")
    scripted([conv([GP("ETH")], [PT(proposal("LONG", "ETH", e))])])
    s = evaluate.run_task(tasks["timeout_02"], 1)["score"]
    assert s["checks"]["data_seen"] is False and "data_seen" in s["failures"]
    scripted([conv([GP("ETH")], [GP("ETH")], [PT(proposal("LONG", "ETH", e))])])  # retried: data arrived
    assert evaluate.run_task(tasks["timeout_02"], 1)["score"]["checks"]["data_seen"] is True


@pytest.mark.parametrize("conversation, reason", [
    ([[GP("BTC")]] * 5, "round_limit"),                      # still calling tools when the budget ran out
    ([[GP("BTC")], "I'd stay out of BTC for now."], "text"),  # answered in prose instead of proposing
    ([[GP("BTC")], [PT(proposal("HOLD", "BTC"))]], "proposal"),
])
def test_stop_reason_separates_running_out_of_turns(scripted, tasks, conversation, reason):
    scripted([conversation])
    s = evaluate.run_task(tasks["clean_01"], trial=1)["score"]
    assert s["stop_reason"] == reason
    assert summarize([s])["by_task"]["clean_01"]["ran_out_of_turns"] == (reason == "round_limit")
    if reason != "proposal":
        assert s["checks"]["completed"] is False  # grading is unchanged: no proposal is still a failure
