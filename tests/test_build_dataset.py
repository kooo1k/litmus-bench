"""build_dataset.py: which trajectories become SFT examples, and what a
corrected example looks like. Runs are generated through evaluate.main with
a scripted model, into tmp_path."""

import json

import pytest

from litmus import agent, evaluate
from scripts import build_dataset
from conftest import get_history, get_price, proposal, propose


def run(scripted, task_id, conversation):
    """One scripted trial of task_id through the real runner; returns its run dir."""
    before = set(evaluate.OUT_ROOT.iterdir()) if evaluate.OUT_ROOT.exists() else set()
    scripted([conversation])
    evaluate.main(["--ids", task_id])
    (run_dir,) = set(evaluate.OUT_ROOT.iterdir()) - before
    return run_dir


def build_one(scripted, tasks, task_id, conversation):
    examples, skipped = build_dataset.build([run(scripted, task_id, conversation)], tasks)
    return examples, skipped


def original_messages(run_dir):
    (record,) = evaluate.load_trials(run_dir).values()
    return record["trajectory"]["messages"]


def decision(example):
    last = example["messages"][-1]
    assert last["role"] == "assistant" and last["tool_calls"][0]["function"]["name"] == "propose_trade"
    return json.loads(last["tool_calls"][0]["function"]["arguments"])


# --------------------------------------------------------------------------
# Passing runs are exported exactly as they happened
# --------------------------------------------------------------------------

def test_passing_trajectory_is_exported_unchanged(scripted, tasks):
    run_dir = run(scripted, "missing_01", [[get_price("BTC")], [propose(proposal("HOLD", "BTC"))]])
    (ex,), skipped = build_dataset.build([run_dir], tasks)
    assert ex["metadata"]["source"] == "passing" and not skipped
    assert ex["messages"] == original_messages(run_dir)
    assert [t["function"]["name"] for t in ex["tools"]] == [
        "get_price", "get_price_history", "get_ohlcv", "get_market_overview", "get_fear_greed", "propose_trade"]


# --------------------------------------------------------------------------
# Corrected: failed fault runs whose only right answer is HOLD
# --------------------------------------------------------------------------

def assert_corrected(ex, run_dir, tasks):
    """Prefix unchanged, final decision a HOLD that passes the real scorer."""
    original = original_messages(run_dir)
    assert ex["messages"][:-1] == original[:-1]
    p = decision(ex)
    assert p["action"] == "HOLD" and agent.schema_violations(p, agent.PROPOSE_TRADE_SCHEMA) == []
    (record,) = evaluate.load_trials(run_dir).values()
    score, _ = build_dataset.rescore_with(tasks[ex["metadata"]["task_id"]], record["trajectory"], p)
    assert score["passed"], score["failures"]
    return p


def test_contradiction_long_is_corrected_to_hold_citing_the_conflict(scripted, tasks, price):
    e = price("contradictory_03", "ETH")
    run_dir = run(scripted, "contradictory_03",
                  [[get_price("ETH")], [get_history("ETH")], [propose(proposal("LONG", "ETH", e))]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    p = assert_corrected(ex, run_dir, tasks)
    assert ex["metadata"]["original_action"] == "LONG" and ex["metadata"]["original_failures"] == ["action"]
    assert p["evidence"] == [{"claim": "get_price(ETH) reports percent_change_24h = 8.0, which conflicts with the price history",
                              "source_tool": "get_price", "value": 8.0}]


def test_invented_price_after_timeout_is_corrected_to_cite_the_error(scripted, tasks):
    run_dir = run(scripted, "timeout_01", [[get_price("BTC")], [propose(proposal("LONG", "BTC", 62800.0))]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    p = assert_corrected(ex, run_dir, tasks)
    assert p["evidence"][0]["claim"] == "get_price(BTC) returned an error: request timed out after 10s"
    assert "value" not in p["evidence"][0]  # no number where there is none
    assert any(m["role"] == "tool" and "timed out" in m["content"] for m in ex["messages"])  # the model saw it


def test_stale_quote_is_cited_with_its_age(scripted, tasks, price):
    b = price("stale_01", "BTC")
    run_dir = run(scripted, "stale_01",
                  [[get_price("BTC"), get_history("BTC")], [propose(proposal("LONG", "BTC", b))]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    p = assert_corrected(ex, run_dir, tasks)
    assert "72.0 hours before the current time" in p["evidence"][0]["claim"]


def test_missing_field_is_cited(scripted, tasks, price):
    e = price("missing_02", "ETH")
    run_dir = run(scripted, "missing_02", [[get_price("ETH")], [propose(proposal("LONG", "ETH", e))]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    p = assert_corrected(ex, run_dir, tasks)
    assert p["evidence"][0]["claim"] == "get_price(ETH) returned no 'last_updated' field"


def test_round_cap_run_keeps_every_message_and_appends_the_decision(scripted, tasks):
    run_dir = run(scripted, "unavailable_02", [[get_price("ETH")]] * agent.MAX_ROUNDS)
    (ex,), _ = build_dataset.build([run_dir], tasks)
    assert ex["metadata"]["original_failures"][0] == "completed"
    assert ex["messages"][:-1] == original_messages(run_dir)  # nothing dropped: it ended on a tool result
    assert decision(ex)["action"] == "HOLD"


def test_only_the_faults_the_model_saw_are_cited(scripted, tasks):
    # unavailable_01 faults both SOL tools; the model only called get_price.
    run_dir = run(scripted, "unavailable_01", [[get_price("SOL")], [propose(proposal("LONG", "SOL", 150.0))]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    assert [e["source_tool"] for e in decision(ex)["evidence"]] == ["get_price"]


# --------------------------------------------------------------------------
# Excluded, with a reason
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task_id,conversation,reason", [
    ("clean_02", [[get_price("ETH"), get_history("ETH")], [propose(proposal("HOLD", "ETH"))]],
     "failed: not a fault task"),
    ("timeout_02", [[get_price("ETH")], [propose(proposal("LONG", "ETH", 999.0))]],
     "failed: HOLD is not the only acceptable action"),
    ("timeout_01", [[propose(proposal("LONG", "BTC", 62800.0))]],
     "failed: required tools never called (fault not in the conversation)"),
])
def test_uncorrectable_failures_are_excluded(scripted, tasks, task_id, conversation, reason):
    examples, skipped = build_one(scripted, tasks, task_id, conversation)
    assert examples == [] and skipped == {reason: 1}


def test_harness_errors_and_changed_questions_are_excluded(scripted, tasks, tmp_path):
    from conftest import PER_DAY_429, QuotaError
    scripted([QuotaError(PER_DAY_429)])
    evaluate.main(["--ids", "clean_01"])
    run_dir = run(scripted, "clean_06", [[get_price("ETH")], [propose(proposal("HOLD", "ETH"))]])
    edited = dict(tasks)
    edited["clean_06"] = {**tasks["clean_06"], "question": "a different question"}
    examples, skipped = build_dataset.build(sorted(evaluate.OUT_ROOT.iterdir()), edited)
    assert examples == []
    assert skipped == {"harness error (no trajectory)": 1, "question changed since this run": 1}


# --------------------------------------------------------------------------
# Format and determinism
# --------------------------------------------------------------------------

def test_reads_older_per_trial_json_runs(scripted, tasks, tmp_path):
    run_dir = run(scripted, "missing_01", [[get_price("BTC")], [propose(proposal("HOLD", "BTC"))]])
    (record,) = evaluate.load_trials(run_dir).values()
    legacy = tmp_path / "legacy_run"
    legacy.mkdir()
    (legacy / "missing_01__t1.json").write_text(json.dumps(record), encoding="utf-8")
    (ex,), _ = build_dataset.build([legacy], tasks)
    assert ex["metadata"]["run_id"] == "legacy_run" and ex["metadata"]["source"] == "passing"


def test_main_writes_jsonl_and_is_deterministic(scripted, tasks, tmp_path, price):
    e = price("contradictory_03", "ETH")
    run_dir = run(scripted, "contradictory_03",
                  [[get_price("ETH")], [get_history("ETH")], [propose(proposal("LONG", "ETH", e))]])
    outs = []
    for name in ("a.jsonl", "b.jsonl"):
        assert build_dataset.main([str(run_dir), "--out", str(tmp_path / name)]) == 0
        outs.append((tmp_path / name).read_text(encoding="utf-8"))
    assert outs[0] == outs[1]
    (line,) = outs[0].splitlines()
    ex = json.loads(line)
    assert set(ex) == {"messages", "tools", "metadata"}
    assert all(m["role"] in ("system", "user", "assistant", "tool") for m in ex["messages"])


def test_hold_that_relied_on_a_trap_is_corrected_to_a_trap_free_hold(scripted, tasks):
    trap = {"claim": "history shows the 24h change", "source_tool": "get_price_history", "value": -2.117142401617}
    run_dir = run(scripted, "missing_03",
                  [[get_price("BTC"), get_history("BTC")],
                   [propose({**proposal("HOLD", "BTC", evidence=[trap]), "reasoning": "BTC is down 2.117% today."})]])
    (ex,), _ = build_dataset.build([run_dir], tasks)
    assert ex["metadata"]["source"] == "corrected" and ex["metadata"]["original_failures"] == ["trap"]
    p = assert_corrected(ex, run_dir, tasks)
    assert p["evidence"] == [{"claim": "get_price(BTC) returned no 'percent_change_24h' field", "source_tool": "get_price"}]
