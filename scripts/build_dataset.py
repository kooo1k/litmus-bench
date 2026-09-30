"""
build_dataset.py

Turn benchmark trajectories into SFT training data (chat-messages JSONL):

    python -m scripts.build_dataset                          # every run in runs/benchmark/
    python -m scripts.build_dataset runs/benchmark/<run_id>  # specific runs
    python -m scripts.build_dataset --out runs/sft/dataset.jsonl

Two kinds of example, one per output line:

  passing    a trajectory that passes its task, messages exactly as they
             happened.
  corrected  a FAILED run of a fault task whose only acceptable action is
             HOLD: every message up to the model's final decision is kept
             as it happened, and the final decision is replaced by a
             deterministic HOLD that cites the fault the model had already
             been shown (the tool error, the missing field, the stale
             timestamp, the conflicting number).

Everything else is excluded, with a counted reason. Every trajectory is
re-scored with the CURRENT score.py and task definitions (stored scores can
be stale), and a trajectory whose question no longer matches its task is
skipped: it answered a different question.

A correction is only made when it is certainly right:
  - the task's only acceptable action is HOLD (so HOLD is THE answer, not
    one of several), and it is a fault task (so there is a fault to cite);
  - the model called every required tool (so the evidence is in the
    conversation it had, not invented by the correction);
  - the fault is visible in those tool results.
Each corrected example is then run through the real governor and scorer
and dropped unless it passes, so the dataset can never contain a
"correction" the benchmark itself would fail.

Output line: {"messages": [...], "tools": [...], "metadata": {...}}. The
messages are OpenAI chat format (system/user/assistant/tool, tool calls
with JSON-string arguments), the same format run_agent logs.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from litmus import agent, evaluate, paths
from litmus.risk_governor import MarketContext, PortfolioState, check_decision
from litmus.score import score_trajectory
from litmus.snapshots import parse_ts

DEFAULT_OUT = paths.RUNS_DIR / "sft" / "dataset.jsonl"
FAULT_CATEGORIES = {"timeout", "stale", "missing_field", "unavailable_asset", "rate_limit", "contradictory"}

# The corrected HOLD's confidence. propose_trade's own schema says a HOLD
# made because data is missing or unclear "should be low"; 0.1 is low
# without being a degenerate 0.
CORRECTED_CONFIDENCE = 0.1


# --------------------------------------------------------------------------
# Reading runs: trials.jsonl (current) or one JSON per trial (older runs)
# --------------------------------------------------------------------------

def read_run(run_dir):
    run_dir = Path(run_dir)
    if (run_dir / evaluate.TRIALS_FILE).exists():
        records = list(evaluate.load_trials(run_dir).values())
    else:
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*__t*.json"))]
    for r in records:
        r.setdefault("benchmark", {})["run_id"] = run_dir.name
    return records


# --------------------------------------------------------------------------
# Correction
# --------------------------------------------------------------------------

def calls_to(tool, symbol, tool_calls):
    return [c for c in tool_calls
            if c["name"] == tool and c.get("args") is not None
            and str(c["args"].get("symbol", "")).strip().upper() == symbol]


def fault_evidence(task, trajectory):
    """Evidence items for each of the task's faults the model actually ran
    into, or None if it ran into none. A faulted tool the model never
    called is skipped, not cited: the correction may only reference what is
    in the conversation. A fault it DID call but whose effect isn't visible
    (e.g. the field is present after all) makes the whole correction unsafe."""
    symbol = task["expected"]["acceptable_symbols"][0]
    tool_calls = trajectory["tool_calls"]
    evidence = []
    for tool, spec in task["faults"].items():
        symbols = spec.get("symbols") or [symbol]
        for sym in symbols:
            calls = calls_to(tool, sym, tool_calls)
            if not calls:
                continue
            kind = spec["type"]
            if kind in ("timeout", "rate_limit", "unavailable_asset"):
                failed = [c for c in calls if c["error_kind"] == "tool"]
                if not failed:
                    return None
                evidence.append({"claim": f"{tool}({sym}) returned an error: {failed[-1]['error']}", "source_tool": tool})
                continue
            seen = [c for c in calls if c["error"] is None and isinstance(c["result"], dict)]
            if not seen:
                return None
            result = seen[-1]["result"]
            if kind == "missing_field":
                if spec["field"] in result:
                    return None
                evidence.append({"claim": f"{tool}({sym}) returned no '{spec['field']}' field", "source_tool": tool})
            elif kind == "stale":
                if "last_updated" not in result:
                    return None
                age_h = (datetime.fromisoformat(trajectory["timestamp"]) - parse_ts(result["last_updated"])).total_seconds() / 3600
                evidence.append({"claim": f"{tool}({sym}) last_updated is {result['last_updated']}, "
                                          f"{age_h:.1f} hours before the current time", "source_tool": tool})
            elif kind == "contradict":
                if result.get(spec["field"]) != spec["value"]:
                    return None
                evidence.append({"claim": f"{tool}({sym}) reports {spec['field']} = {spec['value']}, which conflicts "
                                          f"with the price history", "source_tool": tool, "value": spec["value"]})
            else:
                return None
    return evidence or None


def corrected_proposal(task, trajectory, evidence):
    original = original_proposal(trajectory) or {}
    horizon = original.get("time_horizon")
    faults = "; ".join(e["claim"] for e in evidence)
    conflict = any(spec["type"] == "contradict" for spec in task["faults"].values())
    why = ("The sources conflict, so there is no reliable signal to trade on." if conflict
           else "Data needed for this decision is missing, failed or out of date, so I will not trade on it.")
    return {
        "action": "HOLD",
        "symbol": task["expected"]["acceptable_symbols"][0],
        "confidence": CORRECTED_CONFIDENCE,
        "position_size_pct": 0,
        "leverage": 1,
        "time_horizon": horizon if horizon in ("scalp", "intraday", "swing") else "intraday",
        "evidence": evidence,
        "reasoning": f"{faults}. {why}",
    }


def original_proposal(trajectory):
    try:
        p = json.loads(trajectory.get("raw_proposal") or "null")
    except json.JSONDecodeError:
        return None
    return p if isinstance(p, dict) else None


def strip_final_decision(messages):
    """Messages up to the model's final decision: drop the last assistant
    message if it is the proposal or a prose answer. A run that hit the
    round cap ends on tool results and keeps everything."""
    messages = copy.deepcopy(messages)
    last = messages[-1] if messages else None
    if last and last.get("role") == "assistant":
        calls = last.get("tool_calls") or []
        if not calls or any(c["function"]["name"] == agent.PROPOSE_TRADE for c in calls):
            messages.pop()
    return messages


def rescore_with(task, trajectory, proposal):
    """Score `proposal` as if the model had made it at the end of this
    trajectory, through the real build_decision -> governor -> scorer."""
    t = copy.deepcopy(trajectory)
    t["raw_proposal"] = json.dumps(proposal)
    t["schema_warnings"] = agent.schema_violations(proposal, agent.PROPOSE_TRADE_SCHEMA)
    t["schema_compliant"] = not t["schema_warnings"]
    decision, grounding = agent.build_decision(proposal, t["tool_calls"], datetime.fromisoformat(t["timestamp"]))
    market = MarketContext(unresolved_tool_failures=agent.unresolved_tool_failures(t["tool_calls"]))
    verdict = check_decision(decision, PortfolioState(equity_usd=10_000), market)
    # The decision replaced the model's last message, so its prose is gone
    # too: recompute the prose-number report from the corrected proposal.
    # Keeping the original report would grade the correction on reasoning
    # it doesn't contain (and fail the trap check on the model's words).
    t["final_text"] = None
    t["text_numbers"] = agent.text_numbers(None, proposal, t["tool_calls"])
    t["unverified_text_numbers"] = sum(1 for n in t["text_numbers"] if n["status"] == "unverified")
    t.update(outcome="proposal", grounding=grounding, risk_result=verdict.to_dict(), final_action=verdict.final_action)
    return score_trajectory(task, t), t


def correct(task, trajectory, score):
    """(messages, proposal) for a corrected example, or (None, reason)."""
    if task["category"] not in FAULT_CATEGORIES:
        return None, "failed: not a fault task"
    if task["expected"]["acceptable_actions"] != ["HOLD"]:
        return None, "failed: HOLD is not the only acceptable action"
    if not score["checks"]["required_tools"]:
        return None, "failed: required tools never called (fault not in the conversation)"
    evidence = fault_evidence(task, trajectory)
    if not evidence:
        return None, "failed: fault not visible in the tool results"
    proposal = corrected_proposal(task, trajectory, evidence)
    check, _ = score_with_guard(task, trajectory, proposal)
    if check is None or not check["passed"]:
        return None, f"failed: correction did not pass the scorer ({check and check['failures']})"
    messages = strip_final_decision(trajectory["messages"])
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": "corrected_decision", "type": "function",
        "function": {"name": agent.PROPOSE_TRADE, "arguments": json.dumps(proposal)}}]})
    return (messages, proposal), None


def score_with_guard(task, trajectory, proposal):
    try:
        return rescore_with(task, trajectory, proposal)
    except agent.InvalidProposal:
        return None, None


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def tools_of(trajectory):
    """The tool definitions the model was shown in this run: stored on the
    trajectory for non-default toolsets (MCP tasks), else the REST tools."""
    return trajectory.get("tool_definitions") or agent.TOOLS


def build(run_dirs, tasks):
    examples, skipped = [], Counter()
    for run_dir in run_dirs:
        for record in read_run(run_dir):
            meta = record["benchmark"]
            if "trajectory" not in record:
                skipped["harness error (no trajectory)"] += 1
                continue
            task = tasks.get(meta.get("task_id"))
            trajectory = record["trajectory"]
            if task is None:
                skipped["task no longer exists"] += 1
                continue
            if trajectory["prompt"] != task["question"]:
                skipped["question changed since this run"] += 1
                continue
            score = score_trajectory(task, trajectory)
            base = {"task_id": task["id"], "category": task["category"], "run_id": meta["run_id"],
                    "trial": meta.get("trial"), "model": trajectory.get("model"), "snapshot": task["snapshot"]}
            if score["passed"]:
                examples.append({"messages": trajectory["messages"], "tools": tools_of(trajectory), "metadata": {
                    **base, "source": "passing", "proposed_action": score["proposed"]["action"],
                    "construction_violations": score["construction_violations"]}})
                continue
            result, reason = correct(task, trajectory, score)
            if result is None:
                skipped[reason] += 1
                continue
            messages, proposal = result
            examples.append({"messages": messages, "tools": tools_of(trajectory), "metadata": {
                **base, "source": "corrected", "original_failures": score["failures"],
                "original_action": score["proposed"]["action"], "corrected_action": proposal["action"]}})
    return examples, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export benchmark trajectories as SFT chat-messages JSONL.")
    parser.add_argument("runs", nargs="*", help="run directories (default: every run in runs/benchmark/)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output file (default {DEFAULT_OUT.relative_to(paths.ROOT)})")
    args = parser.parse_args(argv)

    run_dirs = [Path(r) for r in args.runs] or sorted(p for p in evaluate.OUT_ROOT.iterdir() if p.is_dir())
    tasks = {t["id"]: t for t in evaluate.load_tasks()}
    examples, skipped = build(run_dirs, tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")

    sources = Counter(ex["metadata"]["source"] for ex in examples)
    print(f"{len(examples)} examples -> {evaluate.display_path(out)}")
    print(f"  passing: {sources['passing']}   corrected: {sources['corrected']}")
    print(f"  distinct tasks: {len({ex['metadata']['task_id'] for ex in examples})} of {len(tasks)}")
    if skipped:
        print("excluded:")
        for reason, n in skipped.most_common():
            print(f"  {n:4}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
