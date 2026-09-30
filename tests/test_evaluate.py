"""evaluate.py: trial loop, rate limits, persistence and resume. Every model
call is scripted; nothing here reaches a real API."""

import json
import subprocess
import sys

import pytest

from litmus import evaluate
from conftest import (PER_DAY_429, PER_MINUTE_429, QuotaError, get_history, get_price, proposal,
                      propose)

ROOT = evaluate.ROOT
CHILD = ROOT / "tests" / "_child_run.py"
IDS = ["clean_01", "clean_03", "clean_06"]  # three tasks, 2 trials each = 6 trials


def hold(symbol):
    return [[propose(proposal("HOLD", symbol))]]


def only_run_dir():
    dirs = list(evaluate.OUT_ROOT.iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def lines(run_dir):
    return [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]


def pairs(records):
    return [(r["benchmark"]["task_id"], r["benchmark"]["trial"]) for r in records]


def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(evaluate.run_task, "__defaults__", (evaluate.QUOTA_ATTEMPTS, slept.append))
    return slept


# --------------------------------------------------------------------------
# Kill mid-run, then resume: the test the resumable runner exists for
# --------------------------------------------------------------------------

def run_child(tmp_path, kill_at, *evaluate_args):
    log = tmp_path / "conversations.log"
    return subprocess.run(
        [sys.executable, str(CHILD), str(tmp_path / "out"), str(log), str(kill_at), "--", *evaluate_args],
        capture_output=True, text=True, cwd=ROOT, timeout=120)


def started(tmp_path):
    return [json.loads(line) for line in (tmp_path / "conversations.log").read_text().splitlines()]


def test_killed_run_resumes_without_repeating_or_skipping_a_trial(tmp_path):
    # Process 1 is killed (os._exit, no cleanup) as its 4th conversation starts.
    first = run_child(tmp_path, 4, "--ids", *IDS, "--trials", "2")
    assert first.returncode == 9, first.stdout + first.stderr
    run_dir = next((tmp_path / "out").iterdir())
    after_kill = lines(run_dir)
    assert len(after_kill) == 3  # trials 1-3 finished and were fsynced; the 4th died mid-conversation
    assert not (run_dir / "summary.json").exists()  # the old runner would have kept nothing usable

    second = run_child(tmp_path, 0, "--resume", run_dir.name)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "3 trial(s) already scored, 3 to go" in second.stdout

    final = lines(run_dir)
    plan = [(task, trial) for trial in (1, 2) for task in IDS]
    assert sorted(pairs(final)) == sorted(plan)       # none skipped...
    assert len(pairs(final)) == len(set(pairs(final)))  # ...and none run twice
    assert pairs(final)[:3] == pairs(after_kill)        # the resume only appended

    by_pid = {}
    for s in started(tmp_path):
        by_pid.setdefault(s["pid"], []).append(s)
    counts = sorted(len(v) for v in by_pid.values())
    assert counts == [3, 4]  # process 1 began 4 (one killed), process 2 began exactly the 3 missing

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["complete"] and summary["scored_trials"] == 6


def test_half_written_last_line_is_dropped_before_resuming(tmp_path, scripted):
    scripted([hold("BTC")] * 3)
    evaluate.main(["--ids", *IDS])
    run_dir = only_run_dir()
    good = lines(run_dir)
    # Simulate a kill in the middle of writing a 4th record.
    with open(run_dir / "trials.jsonl", "a", encoding="utf-8") as f:
        f.write('{"benchmark": {"task_id": "clean_01", "tri')
    assert evaluate.repair_trials(run_dir) > 0
    assert lines(run_dir) == good
    assert evaluate.repair_trials(run_dir) == 0  # idempotent


def test_corrupt_line_in_the_middle_is_an_error_not_a_guess(tmp_path, scripted):
    scripted([hold("BTC")] * 3)
    evaluate.main(["--ids", *IDS])
    path = only_run_dir() / "trials.jsonl"
    text = path.read_text(encoding="utf-8").splitlines()
    text[1] = text[1][:50]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        evaluate.load_trials(path.parent)


# --------------------------------------------------------------------------
# Rate limits: per-minute retried within a trial, per-day stops the run
# --------------------------------------------------------------------------

def test_quota_kind():
    assert evaluate.quota_kind(QuotaError(PER_MINUTE_429)) == "minute"
    assert evaluate.quota_kind(QuotaError(PER_DAY_429)) == "daily"
    assert evaluate.quota_kind(QuotaError("429 with no quota named")) == "minute"
    assert evaluate.quota_kind(RuntimeError("500")) is None


def test_per_minute_429_is_retried_inside_the_same_trial(monkeypatch, scripted, tasks):
    slept = no_sleep(monkeypatch)
    model = scripted([QuotaError(PER_MINUTE_429), QuotaError(PER_MINUTE_429), hold("BTC")])
    record = evaluate.run_task(tasks["clean_01"], 1)
    assert "score" in record and record["benchmark"]["attempts"] == 3
    assert slept == [6.0, 6.0] and model.started == 3


def test_per_minute_429_on_every_attempt_becomes_a_harness_error(monkeypatch, scripted, tasks):
    no_sleep(monkeypatch)
    scripted([QuotaError(PER_MINUTE_429)] * 3)
    record = evaluate.run_task(tasks["clean_01"], 1)
    assert "harness_error" in record and record["benchmark"]["quota_exhausted"] is True


def test_non_quota_error_is_not_retried(monkeypatch, scripted, tasks):
    slept = no_sleep(monkeypatch)
    err = RuntimeError("500 internal")
    scripted([err])
    record = evaluate.run_task(tasks["clean_01"], 1)
    assert "harness_error" in record and record["benchmark"]["attempts"] == 1 and slept == []
    assert record["benchmark"]["quota_exhausted"] is False


def test_daily_quota_stops_the_whole_run_without_retrying(monkeypatch, scripted):
    slept = no_sleep(monkeypatch)
    model = scripted([hold("BTC"), QuotaError(PER_DAY_429)])  # trial 2 hits the daily cap
    code = evaluate.main(["--ids", *IDS, "--trials", "2"])
    assert code == 2
    assert model.started == 2 and slept == []  # no retry, no further trials attempted
    run_dir = only_run_dir()
    recs = lines(run_dir)
    assert "score" in recs[0] and recs[1]["benchmark"]["daily_quota"] is True
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scored_trials"] == 1 and not summary["complete"]

    # After the "reset": resume re-attempts the trial that hit the cap and finishes the rest.
    model = scripted([hold("BTC")] * 5)
    assert evaluate.main(["--resume", run_dir.name]) == 0
    assert model.started == 5
    final = evaluate.load_trials(run_dir)
    assert len(final) == 6 and all("score" in r for r in final.values())


def test_two_trials_exhausting_retries_in_a_row_stops_the_run(monkeypatch, scripted):
    # A daily cap that doesn't say "PerDay": the fallback must still stop the run.
    no_sleep(monkeypatch)
    model = scripted([QuotaError("429 quota exceeded")] * 6)
    assert evaluate.main(["--ids", *IDS]) == 2
    assert model.started == 6  # 2 trials x 3 attempts, then stop; the 3rd task never starts


def test_one_exhausted_trial_does_not_stop_the_run(monkeypatch, scripted):
    no_sleep(monkeypatch)
    model = scripted([QuotaError(PER_MINUTE_429)] * 3 + [hold("BTC")] * 2)
    assert evaluate.main(["--ids", *IDS]) == 1  # incomplete: one harness error
    assert model.started == 5


# --------------------------------------------------------------------------
# Resume guards
# --------------------------------------------------------------------------

def test_resume_refuses_extra_selection_flags(scripted):
    scripted([hold("BTC")] * 3)
    evaluate.main(["--ids", *IDS])
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", only_run_dir().name, "--trials", "5"])


def test_resume_refuses_if_a_task_definition_changed(tmp_path, scripted):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text((ROOT / "tasks" / "benchmark.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    scripted([QuotaError(PER_DAY_429)])
    assert evaluate.main(["--tasks", str(task_file), "--ids", "clean_01"]) == 2
    edited = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines()]
    next(t for t in edited if t["id"] == "clean_01")["expected"]["acceptable_actions"] = ["HOLD"]
    task_file.write_text("\n".join(json.dumps(t) for t in edited) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", only_run_dir().name])


def test_resume_refuses_a_different_model(monkeypatch, scripted):
    scripted([QuotaError(PER_DAY_429)])
    evaluate.main(["--ids", "clean_01"])
    monkeypatch.setattr(evaluate, "MODEL", "some-other-model")
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", only_run_dir().name])


def test_resume_refuses_if_the_code_changed(monkeypatch, scripted):
    """A new system prompt or scorer fix between sessions must not mix into
    one run: the code fingerprint is recorded and re-checked on resume."""
    scripted([QuotaError(PER_DAY_429)])
    evaluate.main(["--ids", "clean_01"])
    config = json.loads((only_run_dir() / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    assert config["code_sha256"] == evaluate.code_fingerprint()
    monkeypatch.setattr(evaluate, "code_fingerprint", lambda: "0" * 64)
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", only_run_dir().name])


def test_resume_of_a_run_without_a_code_fingerprint_warns_but_proceeds(scripted, capsys):
    """Runs from before fingerprinting can still be resumed, with a warning."""
    scripted([QuotaError(PER_DAY_429), hold("BTC")])
    evaluate.main(["--ids", "clean_01"])
    path = only_run_dir() / evaluate.RUN_FILE
    config = json.loads(path.read_text(encoding="utf-8"))
    del config["code_sha256"]
    path.write_text(json.dumps(config), encoding="utf-8")
    evaluate.main(["--resume", only_run_dir().name])
    assert "predates code fingerprinting" in capsys.readouterr().out


def test_resume_of_a_missing_run_is_an_error():
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", "19990101T000000Z"])


# --------------------------------------------------------------------------
# Trial order, per-task counts, fresh faults per trial
# --------------------------------------------------------------------------

def test_trials_run_round_robin(scripted, tasks):
    model = scripted([hold("BTC")] * 6)
    evaluate.main(["--ids", *IDS, "--trials", "2"])
    order = [m[1]["content"] for m in model.sent if len(m) == 2]
    questions = [tasks[i]["question"] for i in IDS]
    assert order == questions + questions


def test_per_task_counts_and_harness_errors_excluded(monkeypatch, scripted, price):
    no_sleep(monkeypatch)
    up = price("clean_02", "ETH")
    long_eth = [[get_price("ETH"), get_history("ETH")], [propose(proposal("LONG", "ETH", up, horizon="swing"))]]
    hold_eth = [[get_price("ETH"), get_history("ETH")], [propose(proposal("HOLD", "ETH"))]]
    scripted([long_eth, hold_eth, QuotaError(PER_MINUTE_429), QuotaError(PER_MINUTE_429),
              QuotaError(PER_MINUTE_429)])
    evaluate.main(["--ids", "clean_02", "--trials", "3"])
    summary = json.loads((only_run_dir() / "summary.json").read_text())
    t = summary["by_task"]["clean_02"]
    assert (t["passed"], t["scored"]) == (1, 2)  # the rate-limited 3rd trial is not a failure
    assert t["actions"] == {"HOLD ETH": 1, "LONG ETH": 1}
    assert [(e["task_id"], e["trial"]) for e in summary["harness_errors"]] == [("clean_02", 3)]


def test_times_fault_fires_again_in_every_trial(scripted):
    # timeout_02: get_price times out once per trial. The counter must reset per trial.
    conv = [[get_price("ETH")], [propose(proposal("HOLD", "ETH"))]]
    scripted([conv, conv])
    evaluate.main(["--ids", "timeout_02", "--trials", "2"])
    first_calls = [r["trajectory"]["tool_calls"][0]["error"] for r in lines(only_run_dir())]
    assert first_calls == ["request timed out after 10s"] * 2


def test_bad_task_config_fails_before_any_model_call(tmp_path, tasks):
    bad = {**tasks["clean_01"], "snapshot": "tasks/snapshots/nope"}
    with pytest.raises(ValueError, match="no scenario"):
        evaluate.build_tools(bad)
    bad = {**tasks["clean_01"], "faults": {"get_price": {"type": "requests"}}}
    with pytest.raises(ValueError, match="unknown fault type"):
        evaluate.build_tools(bad)
    bad = {**tasks["clean_01"], "faults": {"get_price": {"type": "rate_limit", "times": 0}}}
    with pytest.raises(ValueError, match="times must be >= 1"):
        evaluate.build_tools(bad)


def test_two_runs_in_the_same_second_get_distinct_ids(monkeypatch, scripted):
    from datetime import datetime as real_datetime, timezone

    class FrozenClock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 26, 9, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(evaluate, "datetime", FrozenClock)
    scripted([hold("BTC")] * 2)
    evaluate.main(["--ids", "clean_01"])
    evaluate.main(["--ids", "clean_01"])
    assert sorted(p.name for p in evaluate.OUT_ROOT.iterdir()) == ["20260926T090000Z", "20260926T090000Z-2"]
