"""
evaluate.py

Run the benchmark: each task in tasks/benchmark.jsonl -> replayed snapshot
tools with the task's faults injected -> run_agent -> score -> saved record.

    litmus-bench                              # every task, 1 trial
    litmus-bench --trials 5                   # every task, 5 trials each
    litmus-bench --ids clean_01 stale_03      # a subset
    litmus-bench --category timeout --trials 3
    litmus-bench --tasks tasks/mcp_benchmark.jsonl --trials 3   # the CMC MCP suite
    litmus-bench --resume 20260926T090000Z    # finish an interrupted run
    litmus-bench --report 20260926T090000Z    # reprint a saved run (no model calls)
    litmus-bench --compare RUN_A RUN_B        # what changed between two runs (see compare.py)
    litmus-bench --tasks ... --fail-under 0.6 # for CI: exit 1 below a 60% pass rate

(`litmus-bench` is `python -m litmus.evaluate` without installing.)

Output goes to runs/benchmark/<run_id>/:
  run.json      what was asked for: model, trials, task ids, a fingerprint
                of the task definitions
  trials.jsonl  one line per finished trial (full trajectory + score, or the
                harness error), appended and fsynced the moment it finishes
  summary.json  per-task counts ("3/5 passed") and category rates, rebuilt
                from trials.jsonl at the end of every invocation

Resuming. A run cut off by a daily quota, a crash or Ctrl+C keeps every trial
already written. --resume skips (task, trial) pairs that have a SCORED line
and runs the rest in the original order. A trial that ended in a harness
error has no result, so it is attempted again. Resume refuses if the model or
any selected task's definition changed since the run started: a "5-trial"
result mixing two versions of a task would be meaningless.

Trials run round-robin: every task once, then every task again. If the run
is cut short, each task has the same number of trials, instead of the first
tasks having all of theirs and the last none.

A trial that crashes in the harness or the model API (quota, network) is
recorded as a harness_error and left out of every rate. It says nothing
about the model's behavior, and counting it as a failure would let a flaky
API connection look like a bad agent. A per-minute rate limit is retried
from scratch (see run_task). A per-day quota stops the whole run instead:
it won't clear until the daily reset, so every retry would be wasted.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from litmus import agent, external_agent, faults, llm, mcp_tools, pacing, paths
from litmus.agent import MODEL, run_agent
from litmus.score import (CHECKS, CONSTRUCTION_RULES, recording_gaps, score_trajectory, stop_reason, summarize,
                          wilson_interval)
from litmus.snapshots import replay_tools

ROOT = paths.ROOT
TASKS_PATH = paths.TASKS_DIR / "benchmark.jsonl"
MCP_TASKS_PATH = paths.TASKS_DIR / "mcp_benchmark.jsonl"
OUT_ROOT = paths.BENCHMARK_RUNS_DIR
TRIALS_FILE = "trials.jsonl"
RUN_FILE = "run.json"

# The fault factories a task may name. Explicit, so a task file can't reach
# arbitrary attributes of the faults module.
FAULT_TYPES = {"timeout", "unavailable_asset", "rate_limit", "stale", "missing_field", "contradict", "mcp_error"}

# Tries per trial when the model API rate-limits us per minute. 3 covers the
# free tier's per-minute window resetting.
QUOTA_ATTEMPTS = 3

# --agent: the command of an external agent to grade instead of run_agent
# (see external_agent.py). None means Litmus's own agent.
AGENT_COMMAND = None

# Stop the run after this many trials in a row used up all their rate-limit
# retries. The safety net for a daily cap whose 429 doesn't name itself as
# one: after two trials in a row, it isn't a per-minute window any more.
MAX_CONSECUTIVE_QUOTA_FAILURES = 2


def load_tasks(path=TASKS_PATH):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def recorded_tasks_path(recorded):
    """A run.json tasks_path, usable on any OS from any directory. Runs made
    on Windows store backslashes, and a relative path is relative to the
    repository root, not to wherever the command is run."""
    path = Path(recorded.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def find_run(run):
    """A saved run by id (under OUT_ROOT) or by directory path, e.g. one of
    the published runs in results/."""
    for run_dir in (OUT_ROOT / run, Path(run)):
        if (run_dir / RUN_FILE).is_file():
            return run_dir
    return None


def build_tools(task):
    """Fresh tool functions for ONE run: the task's snapshot, with its
    faults injected. Called per trial on purpose, since fault counters
    (times=N) live in the wrappers and must not carry over between trials.

    A fault spec is {"type": <faults.py function>, **its kwargs}.
    """
    # replay_tools answers a missing snapshot with "symbol not found", so a
    # typo'd path would silently turn any task into an unavailable-asset task.
    snapshot = ROOT / task["snapshot"]
    if not (snapshot / "scenario.json").is_file():
        raise ValueError(f"{task['id']}: no scenario at {task['snapshot']}")
    if is_mcp(task) and not (snapshot / "responses.json").is_file():
        raise ValueError(f"{task['id']}: {task['snapshot']} is not an MCP scenario (no responses.json)")
    specs = {}
    for tool, spec in task["faults"].items():
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in FAULT_TYPES:
            raise ValueError(f"{task['id']}: unknown fault type '{kind}'")
        specs[tool] = getattr(faults, kind)(**spec)
    tools = mcp_tools.replay_mcp_tools(snapshot) if is_mcp(task) else replay_tools(snapshot)
    return faults.inject(tools, specs)


def is_mcp(task):
    """A task on CMC's official MCP tools rather than the REST tools."""
    return task.get("toolset") == "mcp"


def tool_definitions(task):
    """What the model is shown: None (the agent's REST default) or the MCP
    scenario's pinned tools/list descriptions, plus any experimental
    description_patches (see tasks/experiments/)."""
    if not is_mcp(task):
        return None
    definitions = mcp_tools.replay_definitions(ROOT / task["snapshot"])
    if task.get("description_patches"):
        definitions = mcp_tools.patch_descriptions(definitions, task["description_patches"])
    return definitions


# --------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------

class DailyQuotaExhausted(Exception):
    """The model API's per-day quota is used up. Nothing succeeds until the
    reset, so the run stops instead of retrying."""


def quota_kind(error):
    """"daily", "minute", or None (not a rate limit).

    Gemini's 429 body names the quota it hit, e.g. quotaId
    'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'. Only the
    per-minute form has been observed; the per-day form is assumed to follow
    the same pattern ('...PerDay...'). A 429 naming neither is treated as
    per-minute, and MAX_CONSECUTIVE_QUOTA_FAILURES catches a daily cap that
    arrives in an unexpected shape."""
    if getattr(error, "status_code", None) != 429:
        return None
    return "daily" if re.search(r"PerDay|per.day", str(error), re.I) else "minute"


def quota_retry_delay(error):
    """Seconds to wait before retrying a per-minute rate limit. Gemini says
    how long ("Please retry in 25.7s"); fall back to a full minute."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 1 if match else 60.0


def run_task(task, trial, attempts=QUOTA_ATTEMPTS, sleep=time.sleep):
    """One scored trial. A per-minute rate-limited run is restarted from
    scratch (fresh tools, fresh trajectory) after the wait the API asks for.
    Resuming a half-finished conversation would mix two attempts in one
    trajectory. Raises DailyQuotaExhausted on a per-day quota, untried."""
    record = {"benchmark": {"task_id": task["id"], "category": task["category"], "trial": trial,
                            "snapshot": task["snapshot"], "faults": task["faults"], "model": MODEL}}
    for attempt in range(1, attempts + 1):
        try:
            if AGENT_COMMAND:
                trajectory = external_agent.run_external(AGENT_COMMAND, task["question"], build_tools(task),
                                                         tools=tool_definitions(task))
            else:
                trajectory = run_agent(task["question"], tool_functions=build_tools(task), save=False,
                                       verbose=False, tools=tool_definitions(task))
            break
        except external_agent.AgentStartError:
            raise  # every trial would fail the same way: main() stops the run
        except Exception as e:  # API/harness failure, not agent behavior
            kind = quota_kind(e)
            if kind == "daily":
                raise DailyQuotaExhausted(f"{type(e).__name__}: {e}") from e
            if kind is None or attempt == attempts:
                record["harness_error"] = f"{type(e).__name__}: {e}"
                record["benchmark"]["attempts"] = attempt
                record["benchmark"]["quota_exhausted"] = kind == "minute"
                return record
            delay = quota_retry_delay(e)
            print(f"{task['id']:18} t{trial}  rate-limited; retrying in {delay:.0f}s (attempt {attempt + 1}/{attempts})")
            sleep(delay)
    record["benchmark"]["attempts"] = attempt
    record["score"] = score_trajectory(task, trajectory)
    record["trajectory"] = trajectory
    return record


# --------------------------------------------------------------------------
# Run directory: run.json + append-only trials.jsonl
# --------------------------------------------------------------------------

def tasks_fingerprint(tasks):
    """Hash of the exact task definitions a run was started with, so a
    resume can tell if any of them changed underneath it. For an MCP task the
    scenario's pinned tools/list is part of the definition (the model is
    shown those descriptions), so its hash is folded in. REST tasks hash
    exactly as before, so older runs still resume."""
    pinned = [{**t, "_tools_list_sha256": mcp_tools.tools_list_sha256(ROOT / t["snapshot"])} if is_mcp(t) else t
              for t in tasks]
    canonical = json.dumps(sorted(pinned, key=lambda t: t["id"]), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Everything that shapes what the model sees or how a trial is graded. A
# resume across a change to any of these would mix two versions in one
# run (e.g. a new system prompt, or a scorer fix) with no warning. Paths are
# relative to the repository root.
CODE_FILES = tuple(f"litmus/{name}.py" for name in (
    "agent", "decision", "score", "provenance", "risk_governor", "faults", "snapshots", "mcp_tools", "evaluate", "llm", "paths",
    "external_agent"))


def code_fingerprint():
    """SHA-256 over CODE_FILES, line endings normalized (git may rewrite
    LF/CRLF on checkout without changing the code)."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode("utf-8") + b"\0")
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def append_trial(run_dir, record):
    """Append one finished trial and force it to disk before moving on, so
    a kill at any later point can't lose it."""
    with open(run_dir / TRIALS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_trials(run_dir):
    """Drop a half-written last line (a kill mid-write). Without this, the
    next append would glue a complete record onto the fragment and corrupt
    both. Returns the number of bytes removed."""
    path = run_dir / TRIALS_FILE
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1  # 0 if the only line is partial
    with open(path, "r+b") as f:
        f.truncate(keep)
    return len(data) - keep


def load_trials(run_dir):
    """{(task_id, trial): record}. A scored record always wins over a
    harness error for the same pair (the error was the attempt before a
    successful resume). Raises on a corrupt line: a bad line anywhere but
    the end means the file was edited or damaged, not merely interrupted,
    and results built on it can't be trusted."""
    path = run_dir / TRIALS_FILE
    records = {}
    if not path.exists():
        return records
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: line {n} is not valid JSON ({e}); refusing to guess") from e
        key = (record["benchmark"]["task_id"], record["benchmark"]["trial"])
        if key in records and "score" in records[key]:
            if "score" in record:
                raise ValueError(f"{path}: {key} was scored twice")
            continue
        records[key] = record
    return records


def start_run(args, parser):
    """Create a new run directory, or reopen one for --resume. Returns
    (run_dir, config, tasks)."""
    if args.resume:
        if args.ids or args.category or args.trials is not None or args.tasks is not None:
            parser.error("--resume takes its task selection and trial count from the original run")
        run_dir = OUT_ROOT / args.resume
        if not (run_dir / RUN_FILE).is_file():
            parser.error(f"no run to resume at {run_dir}")
        config = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
        by_id = {t["id"]: t for t in load_tasks(recorded_tasks_path(config["tasks_path"]))}
        missing = [i for i in config["task_ids"] if i not in by_id]
        if missing:
            parser.error(f"tasks from the original run no longer exist: {missing}")
        tasks = [by_id[i] for i in config["task_ids"]]
        if config["model"] != MODEL:
            parser.error(f"run used model {config['model']}, current model is {MODEL}; refusing to mix")
        if config.get("base_url") and config["base_url"] != agent.LLM.base_url:
            parser.error(f"run used endpoint {config['base_url']}, current endpoint is {agent.LLM.base_url}; "
                         "the same model name on another provider is not the same model. Refusing to mix.")
        if not config.get("agent") and (config.get("reasoning_effort") or "") != agent.LLM.reasoning_effort:
            parser.error(f"run used reasoning_effort {config.get('reasoning_effort') or '(default)'}, current is "
                         f"{agent.LLM.reasoning_effort or '(default)'}; refusing to mix")
        if tasks_fingerprint(tasks) != config["tasks_sha256"]:
            parser.error("a task definition changed since this run started; results would mix two "
                         "versions of the task. Start a new run instead.")
        if "code_sha256" not in config:
            print("warning: this run predates code fingerprinting; can't verify the agent and scorer are unchanged")
        elif code_fingerprint() != config["code_sha256"]:
            parser.error("the agent, scorer or governor code changed since this run started (code_sha256 "
                         "differs); results would mix two versions. Start a new run instead.")
        return run_dir, config, tasks

    tasks_path = Path(args.tasks or TASKS_PATH).absolute()
    tasks = load_tasks(tasks_path)
    if args.ids:
        unknown = set(args.ids) - {t["id"] for t in tasks}
        if unknown:
            parser.error(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"] in args.ids]
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if not tasks:
        parser.error("no tasks selected")
    trials = 1 if args.trials is None else args.trials
    if trials < 1:
        parser.error("--trials must be >= 1")

    # Second-resolution IDs collide when two runs start in the same second
    # (a script re-launching right after a quota stop); suffix instead of crash.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        run_id = stamp if n == 1 else f"{stamp}-{n}"
        run_dir = OUT_ROOT / run_id
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    config = {"run_id": run_id, "created": datetime.now(timezone.utc).isoformat(), "model": MODEL,
              **({"agent": AGENT_COMMAND, "provider": None, "base_url": None, "reasoning_effort": None,
                  "agent_prompt": None} if AGENT_COMMAND else
                 {"provider": agent.LLM.provider, "base_url": agent.LLM.base_url,
                  "reasoning_effort": agent.LLM.reasoning_effort or None, "agent_prompt": agent.PROMPT_VERSION}),
              "trials": trials, "tasks_path": Path(display_path(tasks_path)).as_posix(), "task_ids": [t["id"] for t in tasks],
              "tasks_sha256": tasks_fingerprint(tasks), "code_sha256": code_fingerprint()}
    (run_dir / RUN_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir, config, tasks


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def display_path(path):
    """Relative to the repo when possible. The output root can live elsewhere
    (tests, a custom location), and printing it must never be what crashes a
    run after every trial has finished."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def fmt_rate(r):
    return "  -  " if r["rate"] is None else f"{r['rate']:.2f}"


def print_trial(record):
    b = record["benchmark"]
    if "harness_error" in record:
        print(f"{b['task_id']:18} t{b['trial']}  HARNESS ERROR  {record['harness_error'][:100]}")
        return
    s = record["score"]
    p = s["proposed"]
    print(f"{b['task_id']:18} t{b['trial']}  {'PASS' if s['passed'] else 'FAIL'}  "
          f"{p['action'] or '-':5} {p['symbol'] or '-':4}  governor {s['governor']['final_action']:5}  "
          f"{', '.join(s['failures'])}"
          + (f"  [build: {', '.join(s['construction_violations'])}]" if s["construction_violations"] else "")
          + (f"  [placeholder x{len(s['schema_warnings'])}]" if s["schema_warnings"] else ""))


def build_summary(config, tasks, records):
    """Summary of everything in trials.jsonl so far, complete or not."""
    # Backfill report-only fields for trials scored before they existed (read
    # from the saved trajectory); pass/fail and every check stay as recorded.
    scores = [{**r["score"],
               "stop_reason": r["score"].get("stop_reason") or stop_reason(r.get("trajectory") or {}),
               "recording_gaps": r["score"].get("recording_gaps", recording_gaps(r.get("trajectory") or {}))}
              for r in records.values() if "score" in r]
    errors = [{"task_id": k[0], "trial": k[1], "error": r["harness_error"]}
              for k, r in sorted(records.items()) if "score" not in r]
    planned = len(tasks) * config["trials"]
    return {"run_id": config["run_id"], "model": config["model"], "tasks": len(tasks),
            "trials": config["trials"], "planned_trials": planned, "scored_trials": len(scores),
            "complete": len(scores) == planned, "harness_errors": errors,
            **(summarize(scores) if scores else {})}


def print_report(summary, tasks):
    if not summary["scored_trials"]:
        return
    # Every task, as counts: "3/5", never a single verdict per task.
    errors_by_task = {}
    for e in summary["harness_errors"]:
        errors_by_task[e["task_id"]] = errors_by_task.get(e["task_id"], 0) + 1
    print(f"\n{'task':18} {'passed':>7}  {'rate':>5}  proposed / failed checks")
    for task in tasks:
        t = summary["by_task"].get(task["id"])
        if t is None:
            print(f"{task['id']:18} {'0/0':>7}  {'  -':>5}  (no scored trials yet)")
            continue
        detail = ", ".join(f"{a} x{n}" for a, n in t["actions"].items())
        if t["failures"]:
            detail += "  | fails: " + ", ".join(f"{f} x{n}" for f, n in t["failures"].items())
        if task["id"] in errors_by_task:
            detail += f"  | {errors_by_task[task['id']]} harness error(s), not counted"
        if t.get("ran_out_of_turns"):
            detail += f"  | {t['ran_out_of_turns']} ran out of turns"
        if t.get("recording_gaps"):
            detail += f"  | {t['recording_gaps']} call(s) outside the recording"
        print(f"{task['id']:18} {t['passed']:>3}/{t['scored']:<3}  {t['passed'] / t['scored']:>5.2f}  {detail}")

    cols = ("pass", *CHECKS)
    print(f"\n{'category':18} {'n':>3}  {'passed':>7}  " + "  ".join(f"{c[:8]:>8}" for c in cols))
    rows = list(summary["by_category"].items()) + [("OVERALL", summary["overall"])]
    for name, b in rows:
        passed = f"{b['passed']}/{b['n']}"
        print(f"{name:18} {b['n']:>3}  {passed:>7}  " + "  ".join(f"{fmt_rate(b[c]):>8}" for c in cols))
    overall = summary["overall"]
    low, high = wilson_interval(overall["passed"], overall["n"])
    print(f"overall pass rate {overall['passed']}/{overall['n']}: 95% CI {low:.0%}-{high:.0%} (Wilson)")
    if overall.get("trap_only_failures"):
        n = overall["trap_only_failures"]
        print(f"strict trap rule: {n} trial(s) failed only the trap check; without it they would pass "
              f"({overall['passed'] + n}/{overall['n']}). Citing a trap value fails even when the model "
              "dismisses it (docs/tasks.md)")

    # Its own table: how well LONG/SHORT trades were built. Not part of
    # pass/fail, and n counts only proposals that opened risk.
    rules = [r.split("_")[0] for r in CONSTRUCTION_RULES]
    print(f"\ntrade construction (LONG/SHORT proposals only; not in pass/fail)")
    print(f"{'category':18} {'n':>3}  {'clean':>8}  " + "  ".join(f"{r:>4}" for r in rules))
    for name, b in rows:
        c = b["construction"]
        if c["n"]:
            print(f"{name:18} {c['n']:>3}  {fmt_rate(c):>8}  "
                  + "  ".join(f"{c['by_rule'][r]:>4}" for r in CONSTRUCTION_RULES))
    print("  " + ", ".join(f"{r.split('_')[0]} = {r.split('_', 1)[1].replace('_', ' ')}" for r in CONSTRUCTION_RULES))
    placeholders = summary["overall"]["placeholder_values"]
    if placeholders:
        print(f"\nschema warnings: {placeholders} placeholder evidence value(s) "
              "(0 cited against a failed tool; not counted as hallucination)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def report_run(args, parser):
    """Reprint a saved run's report from its trials.jsonl: no model calls and
    no API key needed. Scores are the ones recorded when each trial ran."""
    others = [args.ids, args.category, args.trials, args.tasks, args.resume, args.pace,
              args.provider, args.model, args.base_url, args.reasoning_effort, args.agent]
    if any(v is not None for v in others):
        parser.error("--report takes everything from the saved run; don't combine it with other flags")
    run_dir = find_run(args.report)
    if run_dir is None:
        parser.error(f"no run {args.report!r} in {display_path(OUT_ROOT)} or at that path")
    config = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in load_tasks(recorded_tasks_path(config["tasks_path"]))}
    missing = [i for i in config["task_ids"] if i not in by_id]
    tasks = [by_id[i] for i in config["task_ids"] if i in by_id]
    print(f"run {config['run_id']} | model {config['model']}"
          + (f" @ {config['provider']}" if config.get("provider") else "")
          + (f" | reasoning_effort {config['reasoning_effort']}" if config.get("reasoning_effort") else "")
          + (f" | prompt {config['agent_prompt']}" if config.get("agent_prompt") else "")
          + f" | k={config['trials']}")
    if missing:
        print(f"note: tasks no longer in {config['tasks_path']}: {missing} (their trials are still counted)")
    elif tasks_fingerprint(tasks) != config["tasks_sha256"]:
        print("note: some task definitions changed after this run; scores shown are as recorded at the time")
    summary = build_summary(config, tasks, load_trials(run_dir))
    print_report(summary, tasks)
    print(f"\n{summary['scored_trials']}/{summary['planned_trials']} trials scored, "
          f"{len(summary['harness_errors'])} unresolved harness error(s) -> {display_path(run_dir)}")
    return 1 if below_floor(summary, args.fail_under) else 0


def below_floor(summary, floor):
    """--fail-under: True (after saying so) when the overall pass rate is
    below floor. A run with no scored trials is below any floor: CI must
    not go green on a run that never happened."""
    if floor is None:
        return False
    overall = summary.get("overall")
    if not overall or not overall["n"]:
        print("BELOW FLOOR: no scored trials")
        return True
    rate = overall["passed"] / overall["n"]
    if rate < floor:
        print(f"BELOW FLOOR: overall pass rate {rate:.2f} < --fail-under {floor:.2f}")
        return True
    return False


def check_tasks(path):
    """--check: print every structural problem in a task file. Exit 0 if
    there are none, 1 otherwise. No model calls and no keys needed."""
    from litmus import taskcheck  # it imports this module; importing it at the top would be circular
    try:
        tasks = load_tasks(path)
    except (OSError, ValueError) as e:
        print(f"can't read {path}: {e}")
        return 1
    report = taskcheck.check_file(tasks)
    for task_id, found in report.items():
        print(f"{task_id}:")
        for problem in found:
            print(f"  - {problem}")
    print(f"{len(tasks) - len(report)}/{len(tasks)} task(s) OK in {path}")
    return 1 if report else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="litmus-bench", description=__doc__.split("\n\n")[1])
    parser.add_argument("--ids", nargs="*", help="only these task ids")
    parser.add_argument("--category", help="only this category")
    parser.add_argument("--trials", type=int, help="trials per task (default 1)")
    parser.add_argument("--tasks", help=f"task file (default {TASKS_PATH.relative_to(ROOT)})")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run in runs/benchmark/RUN_ID")
    parser.add_argument("--report", metavar="RUN", help="reprint a saved run's report, by run id or directory "
                        "(e.g. results/2026-09-27_mcp_k3_flash-lite); no model calls")
    parser.add_argument("--check", metavar="FILE", help="check a task file's structure and that its tools build "
                        "(no model calls, no keys; see docs/tasks.md)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="diff two saved runs task by task, with confidence intervals (no model calls)")
    parser.add_argument("--pair", action="append", metavar="A_TASK=B_TASK",
                        help="with --compare: match a task in A to a differently named task in B (repeatable)")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="with --compare: exit 1 if B passes a smaller share of the paired tasks than A")
    parser.add_argument("--tolerance", type=float, metavar="TRIALS",
                        help="with --fail-on-regression: how many passing trials B may drop before it counts "
                             "as a regression (default 0; at k=3, 2 or more avoids failing on noise)")
    parser.add_argument("--agent", metavar="COMMAND",
                        help="grade your own agent instead of Litmus's: a command that speaks the JSON Lines "
                        "protocol in external_agent.py (see docs/agents.md); no model key needed")
    parser.add_argument("--fail-under", type=float, metavar="RATE",
                        help="exit 1 if the overall pass rate is below RATE (0-1), for CI; with a run or --report")
    parser.add_argument("--pace", type=float, metavar="RPM",
                        help="pace model requests to RPM per minute and log each one (see pacing.py)")
    parser.add_argument("--provider", help=f"model provider preset: {', '.join(sorted(llm.PROVIDERS))}")
    parser.add_argument("--model", help="model name (default: LLM_MODEL, or GEMINI_MODEL for gemini)")
    parser.add_argument("--base-url", help="any OpenAI-compatible endpoint (overrides --provider's URL)")
    parser.add_argument("--reasoning-effort", metavar="LEVEL",
                        help="sent to the model as reasoning_effort (e.g. none, low, high); default: not sent")
    args = parser.parse_args(argv)
    from litmus import cli_view  # display only; evaluate stays importable without rich
    cli_view.safe_console()
    if args.fail_under is not None and not 0 <= args.fail_under <= 1:
        parser.error("--fail-under takes a pass rate between 0 and 1, e.g. 0.6")
    if args.compare:
        if args.fail_under is not None or args.report or args.check:
            parser.error("--compare can't be combined with --fail-under, --report or --check")
        from litmus import compare  # it imports this module; importing it at the top would be circular
        return compare.run_compare(args, parser)
    if args.pair or args.fail_on_regression or args.tolerance is not None:
        parser.error("--pair, --fail-on-regression and --tolerance only work with --compare")
    if args.report:
        return report_run(args, parser)
    if args.check:
        return check_tasks(args.check)
    global MODEL, AGENT_COMMAND
    AGENT_COMMAND = args.agent
    if args.resume:  # a resumed run keeps the agent it started with
        saved = find_run(args.resume)
        saved_agent = json.loads((saved / RUN_FILE).read_text(encoding="utf-8")).get("agent") if saved else None
        if args.agent and args.agent != saved_agent:
            parser.error("--resume takes its agent from the original run")
        AGENT_COMMAND = saved_agent
    if AGENT_COMMAND:
        if args.provider or args.model or args.base_url or args.reasoning_effort or args.pace is not None:
            parser.error("--agent brings its own model; --provider, --model, --base-url, --reasoning-effort "
                         "and --pace don't apply")
        argv = external_agent.split_command(AGENT_COMMAND)
        if not argv or not (shutil.which(argv[0]) or Path(argv[0]).is_file()):
            parser.error(f"--agent: can't find {argv[0] if argv else 'a command'!r} to run")
        MODEL = external_agent.label(AGENT_COMMAND)
    else:
        if args.provider or args.model or args.base_url or args.reasoning_effort:
            try:
                agent.configure(model=args.model, base_url=args.base_url, provider=args.provider,
                                reasoning_effort=args.reasoning_effort)
            except llm.ConfigError as e:
                parser.error(str(e))
            MODEL = agent.MODEL
        if agent.LLM.notice:
            print(f"note: {agent.LLM.notice}")
        if not agent.LLM.has_key:
            parser.error(f"no API key for {agent.LLM.provider}: set {agent.LLM.key_var}")
        if args.pace is not None:
            pacing.install(agent.client, args.pace)  # after configure: pace the client actually used

    run_dir, config, tasks = start_run(args, parser)
    for t in tasks:
        build_tools(t)  # fail fast on a bad fault spec or snapshot path, before any model call

    dropped = repair_trials(run_dir)
    if dropped:
        print(f"dropped a half-written last line ({dropped} bytes) from {TRIALS_FILE}")
    done = {k for k, r in load_trials(run_dir).items() if "score" in r}
    plan = [(task, trial) for trial in range(1, config["trials"] + 1)  # round-robin: see module docstring
            for task in tasks if (task["id"], trial) not in done]
    if args.resume:
        print(f"resuming {config['run_id']}: {len(done)} trial(s) already scored, {len(plan)} to go")

    stopped, fix, consecutive_quota = None, "Resume after the reset with", 0
    for task, trial in plan:
        try:
            record = run_task(task, trial)
        except DailyQuotaExhausted as e:
            append_trial(run_dir, {"benchmark": {"task_id": task["id"], "category": task["category"],
                                                 "trial": trial, "model": MODEL, "daily_quota": True},
                                   "harness_error": str(e)})
            stopped = "the model API's daily quota is exhausted"
            break
        except external_agent.AgentStartError as e:
            append_trial(run_dir, {"benchmark": {"task_id": task["id"], "category": task["category"],
                                                 "trial": trial, "model": MODEL},
                                   "harness_error": str(e)})
            print(str(e))
            stopped, fix = "the agent could not start", "Fix the agent, then resume with"
            break
        append_trial(run_dir, record)
        print_trial(record)
        consecutive_quota = consecutive_quota + 1 if record["benchmark"].get("quota_exhausted") else 0
        if consecutive_quota >= MAX_CONSECUTIVE_QUOTA_FAILURES:
            stopped = f"{consecutive_quota} trials in a row stayed rate-limited through every retry (likely a daily cap)"
            break

    summary = build_summary(config, tasks, load_trials(run_dir))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report(summary, tasks)
    print(f"\n{summary['scored_trials']}/{summary['planned_trials']} trials scored, "
          f"{len(summary['harness_errors'])} unresolved harness error(s) -> {display_path(run_dir)}")
    if stopped:
        print(f"STOPPED: {stopped}. {fix}:\n  litmus-bench --resume {config['run_id']}")
        return 2
    if below_floor(summary, args.fail_under):
        return 1
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
