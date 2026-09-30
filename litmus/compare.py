"""
compare.py

Diff two saved benchmark runs task by task: which tasks got better or
worse, and which checks changed. This is what the benchmark is for. Change
one thing (the model, the agent prompt, a tool description), rerun, and
see what moved.

    litmus-bench --compare results/2026-09-28_mcp_k3_o4-mini results/2026-09-28_prompt_v2.1/o4-mini
    litmus-bench --compare BASELINE NEW --fail-on-regression       # for CI: exit 1 if NEW passes less
    litmus-bench --compare BASELINE NEW --fail-on-regression --tolerance 2   # ... by more than 2 trials
    litmus-bench --compare A B --pair mcp_eur_threshold=mcp_eur_threshold_eurc_hint

Tasks are paired by id. --pair matches a task in A to a differently named
task in B, e.g. a description-patch variant of the same question. Only
paired tasks count toward the overall line. No model calls and no keys:
only the runs' trials.jsonl files are read.

It is honest about sample size. At k=3 one trial is a third of a task's
result, so each side's overall rate gets a 95% Wilson interval and the
difference a Fisher exact p-value. "15/24 vs 17/24" is then read as the
noise it is, not as a regression.

The CI gate can't lean on the p-value: at k=3 almost no difference reaches
p < 0.05, so a significance gate would never fail. It counts trials
instead, and --tolerance says how many passes B may drop before that counts
as a regression. At k=3 a gap of 2 of 24 is well inside noise (o4-mini's
17 vs 15 has p = 0.76), so use a tolerance of 2 or more at k=3, or run more
trials, to avoid failing builds on noise.
"""

import json

from litmus import evaluate
from litmus.score import fisher_exact_p, wilson_interval


def load_run(run):
    """(config, summary) for a saved run, by id or directory. Raises
    ValueError when there is no run there."""
    run_dir = evaluate.find_run(run)
    if run_dir is None:
        raise ValueError(f"no run {run!r} in {evaluate.display_path(evaluate.OUT_ROOT)} or at that path")
    config = json.loads((run_dir / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    # build_summary only needs the task count; the ids come from run.json,
    # so a run still compares after its task file has moved or changed.
    tasks = [{"id": i} for i in config["task_ids"]]
    return config, evaluate.build_summary(config, tasks, evaluate.load_trials(run_dir))


def label(config):
    """One line naming what was run: the things a comparison can vary."""
    if config.get("agent"):  # an external agent: its model and prompt are its own
        parts = [f"{config['model']} ({config['agent']})"]
    else:
        parts = [config["model"] + (f" @ {config['provider']}" if config.get("provider") else "")]
        if config.get("reasoning_effort"):
            parts.append(f"reasoning_effort {config['reasoning_effort']}")
        parts.append(f"prompt {config.get('agent_prompt') or 'v1'}")  # runs before prompt versioning were v1
    parts.append(f"k={config['trials']}")
    parts.append(config["tasks_path"])
    return " | ".join(parts)


def pair_tasks(a_ids, b_ids, explicit=()):
    """[(a_id, b_id)], then the ids left unpaired on each side. Explicit
    pairs come first, then ids both runs share, in A's order."""
    pairs = list(explicit)
    used_a, used_b = {a for a, _ in pairs}, {b for _, b in pairs}
    pairs += [(i, i) for i in a_ids if i in b_ids and i not in used_a and i not in used_b]
    paired_a, paired_b = {a for a, _ in pairs}, {b for _, b in pairs}
    return (pairs, [i for i in a_ids if i not in paired_a], [i for i in b_ids if i not in paired_b])


def _checks_moved(a, b):
    """Failed checks whose count changed, e.g. ["trap 2->0", "action 0->1"]."""
    fa, fb = a["failures"], b["failures"]
    return [f"{c} {fa.get(c, 0)}->{fb.get(c, 0)}" for c in sorted(set(fa) | set(fb)) if fa.get(c, 0) != fb.get(c, 0)]


def compare(a_summary, b_summary, pairs, tolerance=0):
    """The comparison as data: one row per pair, and the overall line over
    paired tasks that have scored trials on both sides. B has regressed when
    it passes more than `tolerance` trials fewer than A's pass rate predicts
    for B's trial count (so runs of different sizes compare fairly)."""
    rows, totals = [], {"a_passed": 0, "a_n": 0, "b_passed": 0, "b_n": 0}
    for a_id, b_id in pairs:
        a, b = a_summary.get("by_task", {}).get(a_id), b_summary.get("by_task", {}).get(b_id)
        row = {"a_id": a_id, "b_id": b_id, "a": a, "b": b}
        if a and b:
            delta = b["passed"] / b["scored"] - a["passed"] / a["scored"]
            row["change"] = "better" if delta > 0 else "worse" if delta < 0 else "same"
            row["checks"] = _checks_moved(a, b)
            totals["a_passed"] += a["passed"]; totals["a_n"] += a["scored"]
            totals["b_passed"] += b["passed"]; totals["b_n"] += b["scored"]
        else:
            row["change"], row["checks"] = "not scored", []
        rows.append(row)
    overall = dict(totals)
    if totals["a_n"] and totals["b_n"]:
        a_rate, b_rate = totals["a_passed"] / totals["a_n"], totals["b_passed"] / totals["b_n"]
        shortfall = a_rate * totals["b_n"] - totals["b_passed"]  # B's passes below A's rate, in trials
        overall.update(a_rate=a_rate, b_rate=b_rate,
                       a_interval=wilson_interval(totals["a_passed"], totals["a_n"]),
                       b_interval=wilson_interval(totals["b_passed"], totals["b_n"]),
                       p_value=fisher_exact_p(totals["a_passed"], totals["a_n"], totals["b_passed"], totals["b_n"]),
                       shortfall=shortfall, tolerance=tolerance,
                       regressed=shortfall > tolerance + 1e-9)
    return {"rows": rows, "overall": overall}


def _count(t):
    return f"{t['passed']}/{t['scored']}" if t else "-"


def _interval(passed, n, interval):
    return f"{passed}/{n} = {passed / n:.0%} (95% CI {interval[0]:.0%}-{interval[1]:.0%})"


def print_comparison(a_config, b_config, a_run, b_run, result, only_a, only_b):
    print(f"A  {label(a_config)}\n   {a_run}")
    print(f"B  {label(b_config)}\n   {b_run}")
    if a_config.get("code_sha256") and b_config.get("code_sha256") and a_config["code_sha256"] != b_config["code_sha256"]:
        prompts = (a_config.get("agent_prompt") or "v1", b_config.get("agent_prompt") or "v1")
        why = (f"the agent prompt changed ({prompts[0]} -> {prompts[1]}), and any other harness change "
               "would show here too" if prompts[0] != prompts[1]
               else "part of any change may come from the harness, not from what you varied")
        print(f"note: the agent, grader or governor code differs between the runs: {why}")

    names = [r["a_id"] if r["a_id"] == r["b_id"] else f"{r['a_id']} -> {r['b_id']}" for r in result["rows"]]
    width = max([len(n) for n in names] + [len("task")])
    marks = {"better": "+ better", "worse": "- worse", "same": "  same", "not scored": "  not scored"}
    print(f"\n{'task':{width}}  {'A':>5}  {'B':>5}  {'change':12}  failed checks A->B")
    for name, row in zip(names, result["rows"]):
        print(f"{name:{width}}  {_count(row['a']):>5}  {_count(row['b']):>5}  {marks[row['change']]:12}  "
              f"{', '.join(row['checks'])}")

    counts = {c: sum(r["change"] == c for r in result["rows"]) for c in ("better", "worse", "same")}
    print(f"\n{counts['better']} better, {counts['worse']} worse, {counts['same']} same")
    for side, ids in (("A", only_a), ("B", only_b)):
        if ids:
            print(f"only in {side} (not compared): {', '.join(ids)}")

    o = result["overall"]
    if "p_value" not in o:
        print("\nno paired task has scored trials on both sides")
        return
    print(f"\noverall, paired tasks:  A {_interval(o['a_passed'], o['a_n'], o['a_interval'])}")
    print(f"                        B {_interval(o['b_passed'], o['b_n'], o['b_interval'])}")
    verdict = ("a real difference at this sample size" if o["p_value"] < 0.05
               else "not distinguishable from noise at this sample size; run more trials to tell")
    print(f"difference: {o['b_passed'] - o['a_passed']:+d} trial(s), Fisher exact p = {o['p_value']:.2f}: {verdict}")


def run_compare(args, parser):
    """--compare A B [--pair A_ID=B_ID ...] [--fail-on-regression [--tolerance N]].
    Exit 1 with --fail-on-regression when B passes more than N trials fewer
    than A's rate predicts on the paired tasks (N defaults to 0: any drop),
    else 0."""
    a_run, b_run = args.compare
    if args.tolerance is not None and (not args.fail_on_regression or args.tolerance < 0):
        parser.error("--tolerance takes a number of trials >= 0 and goes with --fail-on-regression")
    explicit = []
    for spec in args.pair or []:
        a_id, sep, b_id = spec.partition("=")
        if not sep or not a_id or not b_id:
            parser.error(f"--pair takes A_TASK=B_TASK, got {spec!r}")
        explicit.append((a_id, b_id))
    try:
        (a_config, a_summary), (b_config, b_summary) = load_run(a_run), load_run(b_run)
    except ValueError as e:
        parser.error(str(e))
    for a_id, b_id in explicit:
        if a_id not in a_config["task_ids"]:
            parser.error(f"--pair: {a_id!r} is not a task in run A")
        if b_id not in b_config["task_ids"]:
            parser.error(f"--pair: {b_id!r} is not a task in run B")
    pairs, only_a, only_b = pair_tasks(a_config["task_ids"], b_config["task_ids"], explicit)
    if not pairs:
        parser.error("the runs share no task ids; match tasks with --pair A_TASK=B_TASK")
    result = compare(a_summary, b_summary, pairs, tolerance=args.tolerance or 0)
    print_comparison(a_config, b_config, a_run, b_run, result, only_a, only_b)
    o = result["overall"]
    if args.fail_on_regression and o.get("regressed"):
        print(f"REGRESSION: B passed {o['shortfall']:.1f} trial(s) fewer than A's rate predicts "
              f"(--tolerance {o['tolerance']:g})")
        return 1
    if args.fail_on_regression and o.get("shortfall", 0) > 1e-9:
        print(f"B passed {o['shortfall']:.1f} trial(s) fewer than A's rate predicts: within --tolerance "
              f"{o['tolerance']:g}, not a regression")
    return 0
