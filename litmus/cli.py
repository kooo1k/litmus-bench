"""
cli.py

The `litmus` command (also `python -m litmus`).

    litmus                                  the interactive shell (see shell.py)
    litmus "Should I go long BTC?"          live CoinMarketCap REST tools
    litmus --mcp "Should I go long ETH?"    CoinMarketCap's official MCP server, live
    litmus --task mcp_eur_threshold         a benchmark task on its recorded replay, graded

plus `--list-tasks`. Any OpenAI-compatible model: --provider / --model /
--base-url (see llm.py). This module only wires modes together: the agent
loop is agent.py, the display is cli_view.py, the interactive shell is
shell.py, and both the shell and the flags run through run_once().

Exit codes: 0 done (or a graded task passed), 1 a graded task failed,
2 the model API failed.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import openai

from litmus import agent, cli_view, evaluate, live_tasks, llm, mcp_tools, pacing
from litmus.score import score_trajectory

# Live toolsets: name -> (label, CMC credits used)
MODES = {
    "mcp": "CoinMarketCap MCP · live",
    "all-tools": "CoinMarketCap MCP · live · every tool",
    "rest": "CoinMarketCap REST · live",
}

# The headline run: flash-lite assumed euro parity, went LONG, the risk
# governor approved it, the benchmark failed it.
DEFAULT_REPLAY = ("results/2026-09-27_mcp_k3_flash-lite", "mcp_eur_threshold", 2)

EXAMPLES = """examples:
  litmus                                          interactive shell (recommended)
  litmus --replay                                 watch a saved real run: no keys, no cost
  litmus --task mcp_eur_threshold                 the headline failure, graded, on recorded data
  litmus --mcp "Should I go long ETH right now?"  ask live on CMC's MCP server
  litmus --list-tasks                             every benchmark task
  litmus --provider openai --model o4-mini --task mcp_eur_threshold
"""


def _all_tasks(path=None):
    """The shipped REST, MCP and live tasks, or only the tasks in `path`."""
    if path:
        return evaluate.load_tasks(path)
    return evaluate.load_tasks() + evaluate.load_tasks(evaluate.MCP_TASKS_PATH) + list(live_tasks.LIVE_TASKS)


def prepare_task(task):
    """A task ready to run: a recorded task as it is, a live one (see
    live_tasks.py) set from Litmus's own fetch of today's prices. Raises
    live_tasks.LiveTaskUnavailable when today's market can't support it."""
    if not task.get("live"):
        return task
    return live_tasks.build(task, mcp_tools.live_mcp_tools(["get_crypto_quotes_latest"])["get_crypto_quotes_latest"])


def _live_mcp_setup(all_tools=False):
    """(tool definitions, tool functions, notes) for CoinMarketCap's MCP
    server, live: the benchmark's 9 tools, or with all_tools every tool the
    server lists. The model is shown the server's current tool descriptions,
    compared against the benchmark's pinned copy; if the live list can't be
    fetched, the pinned copy is used and the user is told."""
    pinned = sorted(d for d in mcp_tools.MCP_SNAPSHOT_ROOT.iterdir() if (d / "tools_list.json").is_file())[-1]
    names = None if all_tools else mcp_tools.BENCHMARK_TOOLS
    try:
        listing = mcp_tools.mcp_request("tools/list")["result"]
        definitions = mcp_tools.tool_definitions(listing["tools"], names)
        live = hashlib.sha256(json.dumps(listing, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        note = ("CMC's live tool descriptions match the benchmark's pinned copy"
                if live == mcp_tools.tools_list_sha256(pinned)
                else f"CMC's tool descriptions have changed since the benchmark recording ({pinned.name})")
    except Exception as e:  # network, auth, or a removed tool
        pinned_tools = json.loads((pinned / "tools_list.json").read_text(encoding="utf-8"))["tools"]
        definitions = mcp_tools.tool_definitions(pinned_tools, names)
        note = f"couldn't fetch CMC's live tool list ({type(e).__name__}); showing the model the pinned copy"
    functions = mcp_tools.live_mcp_tools([d["function"]["name"] for d in definitions])
    return definitions, functions, [note]


def missing_key(task=None):
    """What stops a run before any call, as a message naming the fix, or None:
    the model's API key, and for live modes CMC's."""
    config = agent.LLM
    if not config.has_key:
        return f"no API key for {config.provider}: set {config.key_var} (see .env.example)"
    if (task is None or task.get("live")) and not os.getenv("CMC_API_KEY"):
        return "set CMC_API_KEY for live CoinMarketCap data (or run a recorded task with --task ID)"
    return None


def toolset_for(task=None, mode="rest"):
    """(tool definitions, tool functions, header label, notes) for a task's
    recorded replay, or for a live mode in MODES."""
    if task is not None and task.get("live"):
        tools, functions, notes = _live_mcp_setup()
        ref = task["reference"]
        notes.append(f"graded on today's market: Litmus fetched ETH ${ref['eth_usd']:,.2f} and EURC "
                     f"${ref['eurc_usd']:.4f} at {ref['fetched_at']} and set the threshold from them")
        return tools, functions, f"task {task['id']} · CoinMarketCap MCP · live", notes
    if task is not None:
        notes = [f"injected faults: {', '.join(task['faults'])}"] if task["faults"] else []
        label = f"task {task['id']} · recorded replay {task['snapshot'].rstrip('/').split('/')[-1]}"
        return evaluate.tool_definitions(task), evaluate.build_tools(task), label, notes
    if mode in ("mcp", "all-tools"):
        tools, functions, notes = _live_mcp_setup(all_tools=mode == "all-tools")
        if mode == "all-tools":
            notes.append("all tools: execute_skill runs hosted skills that can take 30-300s each")
        return tools, functions, f"CoinMarketCap MCP · live · {len(tools)} tools", notes
    return None, None, MODES["rest"], []


def load_saved_trial(run, task_id=None, trial=None):
    """(record, task, run config) for one saved trial of a benchmark run
    (a run id or directory, e.g. results/2026-09-27_mcp_k3_flash-lite).
    Without a task id, the run's first trial; without a trial number, the
    task's first. Raises LookupError with the choices when nothing matches."""
    run_dir = evaluate.find_run(run)
    if run_dir is None:
        raise LookupError(f"no saved run {run!r}")
    config = json.loads((run_dir / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    records = evaluate.load_trials(run_dir)
    matches = [key for key in records if (task_id is None or key[0] == task_id)
               and (trial is None or key[1] == trial) and "score" in records[key]]
    if not matches:
        raise LookupError(f"no scored trial {task_id or ''} {trial or ''} in {run}; it has: "
                          + ", ".join(sorted({key[0] for key in records})))
    key = sorted(matches, key=lambda k: (k[1], k[0]) if task_id is None else k)[0]
    tasks = {t["id"]: t for t in evaluate.load_tasks(evaluate.recorded_tasks_path(config["tasks_path"]))}
    return records[key], tasks[key[0]], config


def replay(view, run=None, task_id=None, trial=None, delay=0.0):
    """Play a saved trial back through the live display: the header, each
    tool call as it happened, then the decision, governor and grade. No
    model call and no API key; `delay` seconds between tool lines makes it
    look like the live run it was."""
    if run is None:
        run, task_id, trial = DEFAULT_REPLAY
    record, task, config = load_saved_trial(run, task_id, trial)
    trajectory = record["trajectory"]
    name = str(run).replace("\\", "/")
    name = name[len("results/"):] if name.startswith("results/") else Path(name).name
    label = f"replay · {name} · trial {record['benchmark']['trial']}"
    provider = ("your own agent" if config.get("agent")
                else config.get("provider") or (llm._infer_provider(config["base_url"]) if config.get("base_url") else None)
                or ("gemini" if trajectory["model"].startswith("gemini") else "unrecorded provider"))
    view.header(trajectory["model"], provider, label, task["question"],
                [f"a saved run from {str(trajectory.get('timestamp', ''))[:16].replace('T', ' ')} UTC: "
                 "no model call, no API key"])
    for call in trajectory["tool_calls"]:
        if delay:
            time.sleep(delay)
        view.on_event("tool", call)
    if delay:
        time.sleep(delay)
    cli_view.render_result(view, trajectory, task, record["score"])
    return 0


def run_once(view, question, task=None, mode="rest", as_json=False):
    """Run the agent on one question (a task's, when `task` is given), show
    the result, and return (exit code, trajectory, grade). Keys are checked by
    the caller (missing_key); a model API failure is shown, not raised."""
    config = agent.LLM
    notes = [config.notice] if config.notice else []
    if config.reasoning_effort:
        notes.append(f"reasoning_effort: {config.reasoning_effort}")
    tools, functions, label, more = toolset_for(task, mode)
    notes += more
    if not as_json:
        view.header(agent.MODEL, config.provider, label, question, notes)
    try:
        with view.running():
            trajectory = agent.run_agent(question, tool_functions=functions, tools=tools, verbose=False,
                                         on_event=None if as_json else view.on_event)
    except openai.APIError as e:
        view.error(f"the model API failed: {type(e).__name__}: {str(e)[:300]}",
                   "Retry, choose another model (--provider/--model), or run a recorded task with --task ID.")
        return 2, None, None

    score = score_trajectory(task, trajectory) if task is not None else None
    if task is not None and task.get("live"):
        # The right answer was set before the run; make sure the market
        # didn't cross it while the agent worked.
        score["live_inconclusive"] = live_tasks.recheck(task, functions["get_crypto_quotes_latest"])
        # A live task exists only for this run: keep what it was, and its
        # grade, with the saved trajectory, or the run can't be re-checked.
        trajectory["live_task"] = {**{k: task[k] for k in ("id", "question", "expected", "rationale", "reference")},
                                   "grade": score}
        if trajectory.get("saved_to"):
            Path(trajectory["saved_to"]).write_text(json.dumps(trajectory, indent=2, default=str), encoding="utf-8")
    if as_json:
        print(json.dumps({"trajectory": trajectory, "grade": score}, indent=2, default=str))
    else:
        cli_view.render_result(view, trajectory, task, score)
        if score and score.get("live_inconclusive"):
            view.error(score["live_inconclusive"], "Run the task again; the market rarely moves this far in a minute.")
    return (0 if score is None or score["passed"] else 1), trajectory, score


def main(argv=None):
    raw = sys.argv[1:] if argv is None else argv
    if not raw:
        return _start_shell()

    parser = argparse.ArgumentParser(
        prog="litmus", epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Ask the trading agent one question on live CoinMarketCap data (REST tools, or CMC's official "
                    "MCP server with --mcp), or run a benchmark task on its recorded replay and see it graded (--task). "
                    "Run with no arguments for the interactive shell.")
    parser.add_argument("question", nargs="*", help="your question (not used with --task or --list-tasks)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mcp", action="store_true", help="use CoinMarketCap's official MCP server, live")
    mode.add_argument("--task", metavar="ID", help="run a benchmark task on its recorded replay and grade it")
    mode.add_argument("--list-tasks", action="store_true", help="list benchmark task ids and questions")
    mode.add_argument("--replay", nargs="*", metavar="RUN [TASK [TRIAL]]",
                      help="play a saved real run back, no keys needed (default: the headline EUR failure)")
    parser.add_argument("--tasks", metavar="FILE", help="with --task / --list-tasks: your own task file "
                        "(default: the shipped benchmark; see docs/tasks.md)")
    parser.add_argument("--all-tools", action="store_true",
                        help="with --mcp: every tool CMC's server lists, not just the benchmark's 9 "
                             "(adds find_skill and execute_skill; hosted skills take 30-300s)")
    parser.add_argument("--provider", help=f"model provider preset: {', '.join(sorted(llm.PROVIDERS))}")
    parser.add_argument("--model", help="model name (default: LLM_MODEL, or GEMINI_MODEL for gemini)")
    parser.add_argument("--base-url", help="any OpenAI-compatible endpoint")
    parser.add_argument("--reasoning-effort", metavar="LEVEL",
                        help="sent to the model as reasoning_effort (e.g. none, low, high); default: not sent")
    parser.add_argument("--pace", type=float, metavar="RPM", help="space model requests to RPM per minute")
    parser.add_argument("--json", action="store_true", help="print the trajectory (and grade) as JSON")
    parser.add_argument("--plain", action="store_true", help="plain text, no colors or panels")
    args = parser.parse_args(raw)
    cli_view.safe_console()

    view = cli_view.LiveView(plain=args.plain or args.json)
    if args.replay is not None:
        run, task_id, trial = (args.replay + [None, None, None])[:3]
        try:
            return replay(view, run, task_id, int(trial) if trial else None, delay=0 if args.plain else 0.35)
        except (LookupError, ValueError) as e:
            parser.error(str(e))
    if args.list_tasks:
        cli_view.render_task_list(view, _all_tasks(args.tasks))
        return 0

    question, task = " ".join(args.question).strip(), None
    if args.task:
        task = next((t for t in _all_tasks(args.tasks) if t["id"] == args.task), None)
        if task is None:
            parser.error(f"unknown task {args.task!r} (see --list-tasks)")
        if question:
            parser.error("--task asks the task's own question; don't pass one")
        question = task["question"]
    elif not question:
        parser.error('ask a question, e.g. litmus --mcp "Should I go long ETH right now?" (or run litmus alone)')

    if args.tasks and not (args.task or args.list_tasks):
        parser.error("--tasks goes with --task ID or --list-tasks")
    if args.all_tools and not args.mcp:
        parser.error("--all-tools only applies to --mcp (the benchmark always uses its pinned 9 tools)")

    if args.provider or args.model or args.base_url or args.reasoning_effort:
        try:
            agent.configure(model=args.model, base_url=args.base_url, provider=args.provider,
                            reasoning_effort=args.reasoning_effort)
        except llm.ConfigError as e:
            parser.error(str(e))
    problem = missing_key(task)
    if problem:
        parser.error(problem)
    if args.pace is not None:
        pacing.install(agent.client, args.pace)
    if task is not None and task.get("live"):
        try:
            task = prepare_task(task)
        except live_tasks.LiveTaskUnavailable as e:
            view.error(str(e), "Recorded tasks need no CMC key: litmus --task mcp_eur_threshold")
            return 2
        question = task["question"]

    live_mode = "all-tools" if args.all_tools else ("mcp" if args.mcp else "rest")
    code, _, _ = run_once(view, question, task=task, mode=live_mode, as_json=args.json)
    return code


def _start_shell():
    """Bare `litmus`: the interactive shell on a terminal; a short hint when
    piped or scripted (no terminal to be interactive with)."""
    cli_view.safe_console()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("litmus: run it in a terminal for the interactive shell, or pass a command, e.g.\n"
              "  litmus --task mcp_eur_threshold\n  litmus --help")
        return 0
    from pathlib import Path

    from litmus import shell  # prompt_toolkit is only needed for the shell
    return shell.Shell(history_path=Path.home() / ".litmus_history").run()


if __name__ == "__main__":
    sys.exit(main())
