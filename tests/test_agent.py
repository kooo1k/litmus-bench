"""agent.py: the loop, tool-error classification, provenance and schema
checks. The model is scripted; tools are replayed or fake."""

import json
from datetime import datetime
from types import SimpleNamespace as NS

import pytest

from litmus import agent
from conftest import get_price, proposal, propose
from litmus.snapshots import SNAPSHOT_DIR, replay_tools


def call(name, arguments):
    return NS(id="c", function=NS(name=name, arguments=arguments if isinstance(arguments, str) else json.dumps(arguments)))


TOOLS = {"get_price": lambda symbol: {"symbol": symbol, "price_usd": 100.0, "last_updated": "2026-01-01T00:00:00Z"}}


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------

def test_system_prompt_carries_the_run_start_time(scripted):
    model = scripted([[[get_price("BTC")], [propose(proposal("HOLD", "BTC"))]]])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    system = model.sent[0][0]["content"]
    stamp = system.rsplit("The current time is ", 1)[1].split(" ")[0]
    assert datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(microsecond=0) == \
        datetime.fromisoformat(t["timestamp"]).replace(microsecond=0)


def test_prose_answer_is_no_proposal(scripted):
    scripted([["BTC looks fine."]])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    assert t["outcome"] == "no_proposal" and t["final_action"] == "HOLD"


def test_round_cap_ends_a_model_that_never_proposes(scripted):
    scripted([[[get_price("BTC")]] * agent.MAX_ROUNDS])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    assert t["outcome"] == "no_proposal" and len(t["tool_calls"]) == agent.MAX_ROUNDS


def test_data_calls_alongside_the_proposal_are_skipped(scripted):
    scripted([[[get_price("BTC"), propose(proposal("HOLD", "BTC"))]]])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    assert t["tool_calls"][0]["error_kind"] == "skipped" and t["tool_calls"][0]["result"] is None


def test_long_without_entry_is_an_invalid_proposal(scripted):
    p = proposal("LONG", "BTC")
    scripted([[[get_price("BTC")], [propose(p)]]])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    assert t["outcome"] == "invalid_proposal" and "entry_price" in t["proposal_error"]


def test_hold_without_evidence_is_valid_but_flagged(scripted):
    p = proposal("HOLD", "BTC")
    del p["evidence"]
    scripted([[[propose(p)]]])
    t = agent.run_agent("q", tool_functions=replay_tools(), save=False, verbose=False)
    assert t["outcome"] == "proposal" and any("evidence" in w for w in t["schema_warnings"])


# --------------------------------------------------------------------------
# Tool-call classification
# --------------------------------------------------------------------------

def test_tool_error_kinds():
    assert agent.execute_tool_call(call("get_price", "{not json"), TOOLS)["error_kind"] == "argument"
    assert agent.execute_tool_call(call("get_price", {"days": 7}), TOOLS)["error_kind"] == "argument"
    assert agent.execute_tool_call(call("get_quote", {"symbol": "BTC"}), TOOLS)["error_kind"] == "unknown_tool"
    failing = {"get_price": lambda symbol: {"error": "down"}}
    assert agent.execute_tool_call(call("get_price", {"symbol": "BTC"}), failing)["error_kind"] == "tool"

    def raises(symbol):
        raise TimeoutError("slow")
    rec = agent.execute_tool_call(call("get_price", {"symbol": "BTC"}), {"get_price": raises})
    assert rec["error_kind"] == "tool" and rec["result"] == {"error": "TimeoutError: slow"}


def rec(name, symbol, error_kind=None):
    return {"name": name, "args": {"symbol": symbol}, "error_kind": error_kind}


def test_unresolved_failures_clear_on_a_successful_retry():
    calls = [rec("get_price", "BTC", "tool"), rec("get_price", "btc"), rec("get_price_history", "ETH", "tool")]
    assert agent.unresolved_tool_failures(calls) == ["get_price_history(ETH)"]
    assert agent.unresolved_tool_failures([rec("get_price", "BTC", "argument")]) == []


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cited,actual,ok", [
    (85000, 85012.3, True),      # 3 significant figures
    (90000, 86000, False),       # 4.6% off
    (3.2, 3.2417, True),         # rounded to one decimal
    (3.3, 3.2417, False),
    (-2.117, 2.117142, False),   # a dropped sign is not a rounding
])
def test_values_match(cited, actual, ok):
    assert agent.values_match(cited, actual) is ok


def test_find_source_distinguishes_misattribution_from_invention():
    calls = [{"name": "get_price_history", "error": None, "result": {"quotes": [{"price": 84370.4}]}}]
    assert agent.find_source(84370.4, "get_price_history", calls) == ("get_price_history", "grounded")
    assert agent.find_source(84370.4, "get_price", calls) == ("get_price_history", "misattributed")
    assert agent.find_source(99999.0, "get_price", calls) == ("", "ungrounded")


def test_entry_price_must_come_from_a_quote_for_the_same_symbol():
    calls = [{"name": "get_price", "error": None, "result": {"symbol": "BTC", "price_usd": 85000.0,
                                                              "last_updated": "2026-01-01T00:00:00Z"}}]
    p = proposal("LONG", "ETH", 85000.0)
    _, grounding = agent.build_decision(p, calls, datetime.fromisoformat("2026-01-01T00:00:30+00:00"))
    assert grounding[-1]["claim"] == "[harness] entry_price" and grounding[-1]["status"] == "ungrounded"


def test_extract_numbers_skips_durations_counts_and_timestamps():
    found = [v for _, v in agent.extract_numbers("BTC at $85,012 is up 3.2% over 24h, 7 days, 3x, 2026-09-25T08:00Z, $1.5k")]
    assert found == [85012.0, 3.2, 1500.0]


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_schema_violations():
    schema = agent.PROPOSE_TRADE_SCHEMA
    assert agent.schema_violations(proposal("HOLD", "BTC"), schema) == []
    assert any("not one of" in v for v in agent.schema_violations({**proposal("HOLD", "BTC"), "action": "BUY"}, schema))
    assert any("unexpected field" in v for v in agent.schema_violations({**proposal("HOLD", "BTC"), "extra": 1}, schema))
    assert any("expected number" in v for v in agent.schema_violations({**proposal("HOLD", "BTC"), "confidence": True}, schema))
    ev_null = {**proposal("HOLD", "BTC"), "evidence": [{"claim": "x", "source_tool": "get_price", "value": None}]}
    assert agent.schema_violations(ev_null, schema) == []  # optional null == absent


def test_evidence_value_description_tells_the_model_it_may_omit():
    desc = agent.PROPOSE_TRADE_SCHEMA["properties"]["evidence"]["items"]["properties"]["value"]["description"]
    assert "Omit this field" in desc


def test_default_snapshot_dir_is_the_live_scenario():
    assert SNAPSHOT_DIR.name == "2026-09-25_live"



def test_prompt_allows_arithmetic_but_not_invention_and_leaks_no_task_numbers():
    """v2.1 prompt (2026-09-28). Its worked example must be neutral: no number
    a task's answer turns on (a decimal, or 1,000 and up, in any rationale)
    may appear in it, or the prompt would teach to the test."""
    import re
    from litmus import evaluate
    assert agent.PROMPT_VERSION == "v2.1"
    assert "Never invent a number" in agent.SYSTEM_PROMPT and "calculated from such numbers" in agent.SYSTEM_PROMPT
    assert "calculated numbers never go in evidence" in agent.SYSTEM_PROMPT  # evidence = tool values (grounding)
    assert set(re.findall(r"\d+(?:\.\d+)?", agent.SYSTEM_PROMPT)) == {"100", "4", "25"}  # the neutral example
    tasks = evaluate.load_tasks() + evaluate.load_tasks(evaluate.MCP_TASKS_PATH)
    key_numbers = {n for task in tasks for n in re.findall(r"\d+(?:\.\d+)?", task["rationale"].replace(",", ""))
                   if "." in n or float(n) >= 1000}
    assert not any(n in agent.SYSTEM_PROMPT for n in key_numbers)
