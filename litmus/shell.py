"""
shell.py

The interactive `litmus` shell: run `litmus` with no arguments. Type a trading
question to ask the agent live (CoinMarketCap's MCP server by default), or a
/command; press Tab or just type "/" for suggestions. Everything the flags
can do is here, so nobody has to look up or paste a command.

    /task mcp_eur_threshold      the headline failure, graded, on recorded data
    /model o4-mini               switch model (keeps the provider)
    /report                      the published results, no keys needed

The shell only routes: runs go through cli.run_once, benchmarks and reports
through evaluate.main, exactly as the flags do. Input comes from a
prompt_toolkit session on a terminal, or from any `read_line` callable
(tests drive the shell that way).
"""

import difflib
import json
import os
import shlex

from litmus import agent, cli, cli_view, evaluate, llm, paths

MODE_HELP = {"mcp": "CMC's MCP server, the benchmark's 9 tools (default)",
             "all-tools": "CMC's MCP server, every tool it lists (incl. find_skill / execute_skill)",
             "rest": "CMC's REST API"}
SUITES = {"mcp": evaluate.MCP_TASKS_PATH, "rest": evaluate.TASKS_PATH}
REQUESTS_PER_TRIAL = 4  # measured average is ~3-4 model requests per trial
REASONING = ("none", "low", "medium", "high", "off")


def task_files():
    """Task files a user is likely to want: shipped, examples, experiments."""
    files = [paths.TASKS_DIR / "benchmark.jsonl", paths.TASKS_DIR / "mcp_benchmark.jsonl",
             paths.ROOT / "docs" / "examples" / "tasks.jsonl", *sorted((paths.TASKS_DIR / "experiments").glob("*.jsonl"))]
    return [f.relative_to(paths.ROOT).as_posix() for f in files if f.is_file()]


def saved_runs():
    """Saved benchmark runs, newest first: published ones in results/, then
    local ones in runs/benchmark/. Returned as paths relative to the repo."""
    found = sorted({p.parent for p in (paths.ROOT / "results").rglob(evaluate.RUN_FILE)}, reverse=True)
    if evaluate.OUT_ROOT.is_dir():
        found += sorted((p for p in evaluate.OUT_ROOT.iterdir() if (p / evaluate.RUN_FILE).is_file()), reverse=True)
    return [p.relative_to(paths.ROOT).as_posix() if p.is_relative_to(paths.ROOT) else str(p) for p in found]


LOCAL_RUNS_SHOWN = 8


def run_score(run):
    """'16/24' for a saved run, from its summary.json, or '' if unknown."""
    try:
        overall = json.loads((paths.ROOT / run / "summary.json").read_text(encoding="utf-8"))["overall"]
        return f"{overall['passed']}/{overall['n']}"
    except (OSError, ValueError, KeyError):
        return ""


def run_model(run):
    """'o4-mini · prompt v2.1' for a saved run, from its run.json."""
    try:
        config = json.loads((paths.ROOT / run / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    extras = [f"reasoning {config['reasoning_effort']}"] if config.get("reasoning_effort") else []
    extras += [f"prompt {config['agent_prompt']}"] if config.get("agent_prompt") else []
    return " · ".join([config.get("model", "?"), *extras])


class Shell:
    """State (live mode, your task file) plus a table of /commands."""

    def __init__(self, read_line=None, view=None, history_path=None):
        self.view = view or cli_view.LiveView()
        self.mode = "mcp"
        self.task_file = None
        self._read_line = read_line
        self._history_path = history_path
        self.commands = {
            "task": (self.cmd_task, "[id]", "run a benchmark task on its recording, graded (no CMC credits)"),
            "tasks": (self.cmd_tasks, "[file]", "list the tasks (the shipped 52, or your own file)"),
            "replay": (self.cmd_replay, "[run [task [trial]]]", "watch a saved real run: no keys, no cost"),
            "mode": (self.cmd_mode, "mcp|rest|all-tools", "what live questions run on"),
            "model": (self.cmd_model, "<name>", "switch model, keeping the provider"),
            "provider": (self.cmd_provider, "<preset> [model]", f"switch provider: {', '.join(sorted(llm.PROVIDERS))}"),
            "reasoning": (self.cmd_reasoning, "none|low|medium|high|off", "reasoning_effort sent to the model"),
            "report": (self.cmd_report, "[run]", "reprint a saved benchmark run (no model calls)"),
            "bench": (self.cmd_bench, "[mcp|rest|file] [trials]", "run a benchmark suite (asks before spending)"),
            "check": (self.cmd_check, "<file>", "validate a task file (no model calls)"),
            "status": (self.cmd_status, "", "model, keys and mode"),
            "help": (self.cmd_help, "", "this list"),
            "clear": (self.cmd_clear, "", "clear the screen"),
            "exit": (self.cmd_exit, "", "leave (or Ctrl+D)"),
        }

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run(self):
        self.view_banner()
        read = self._read_line or self._prompt_toolkit_reader()
        while True:
            try:
                line = read("litmus › ")
            except KeyboardInterrupt:  # Ctrl+C at the prompt clears the line, like other shells
                continue
            except EOFError:
                break
            if line is None:
                break
            try:
                if self.handle(line) is False:
                    break
            except KeyboardInterrupt:
                self.say("[yellow]cancelled[/]", "cancelled")
        self.say("[dim]bye[/]", "bye")
        return 0

    def handle(self, line):
        """One input line. Returns False to leave the shell."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            return self.ask(line)
        try:
            words = shlex.split(line[1:])
        except ValueError:
            words = line[1:].split()
        name, args = (words[0].lower(), words[1:]) if words else ("help", [])
        if name in ("quit", "q"):
            name = "exit"
        if name not in self.commands:
            close = difflib.get_close_matches(name, self.commands, n=1)
            hint = f" Did you mean /{close[0]}?" if close else ""
            self.say(f"[red]unknown command /{name}.[/]{hint} Type [bold]/help[/] for the list.",
                     f"unknown command /{name}.{hint} Type /help for the list.")
            return True
        return self.commands[name][0](args)

    def read(self, prompt):
        return (self._read_line or self._prompt_toolkit_reader())(prompt)

    def _prompt_toolkit_reader(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory

        history = FileHistory(str(self._history_path)) if self._history_path else InMemoryHistory()
        session = PromptSession(history=history, completer=Completer(self), complete_while_typing=True,
                                bottom_toolbar=self.toolbar)
        self._read_line = session.prompt
        return session.prompt

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def say(self, markup, plain=None):
        if self.view.rich:
            self.view.console.print(markup)
        else:
            print(plain if plain is not None else markup)

    def status(self):
        config = agent.LLM
        return {"model": config.model, "provider": config.provider, "model_key": config.has_key,
                "key_var": config.key_var, "cmc_key": bool(os.getenv("CMC_API_KEY")),
                "mode": cli.MODES[self.mode], "prompt": agent.PROMPT_VERSION,
                "reasoning": config.reasoning_effort, "task_file": self.task_file}

    def toolbar(self):
        s = self.status()
        return (f" {s['model']} @ {s['provider']}{' · reasoning ' + s['reasoning'] if s['reasoning'] else ''}"
                f" · {self.mode} · / for commands · Ctrl+D to exit")

    def view_banner(self):
        cli_view.render_banner(self.view, self.status())

    def hint(self, text):
        self.say(f"[dim]next: {text}[/]", f"next: {text}")

    # ------------------------------------------------------------------
    # Running the agent
    # ------------------------------------------------------------------

    def ask(self, question):
        problem = cli.missing_key()
        if problem:
            self.say(f"[red]{problem}[/]", problem)
            self.hint("/provider to switch model, or /task mcp_eur_threshold (recorded data, no CMC key needed)")
            return True
        cli.run_once(self.view, question, mode=self.mode)
        self.hint("/task mcp_eur_threshold grades a question like this against the right answer")
        return True

    def cmd_task(self, args):
        tasks = cli._all_tasks(self.task_file)
        if not args:
            self.render_tasks(tasks)
            self.hint("/task <id>  (Tab completes the id)")
            return True
        task = next((t for t in tasks if t["id"] == args[0]), None)
        if task is None:
            close = difflib.get_close_matches(args[0], [t["id"] for t in tasks], n=1)
            self.say(f"[red]no task {args[0]!r}.[/]" + (f" Did you mean {close[0]}?" if close else ""),
                     f"no task {args[0]!r}." + (f" Did you mean {close[0]}?" if close else ""))
            return True
        problem = cli.missing_key(task)
        if problem:
            self.say(f"[red]{problem}[/]", problem)
            return True
        try:
            task = cli.prepare_task(task)  # a live task gets today's threshold here
        except cli.live_tasks.LiveTaskUnavailable as e:
            self.say(f"[red]{e}[/]", str(e))
            return True
        cli.run_once(self.view, task["question"], task=task)
        self.hint("/model <name> and run it again to compare models · /report for the published runs")
        return True

    def cmd_replay(self, args):
        run, task_id, trial = (list(args) + [None, None, None])[:3]
        try:
            cli.replay(self.view, run, task_id, int(trial) if trial else None, delay=0.35 if self.view.rich else 0)
        except (LookupError, ValueError) as e:
            self.say(f"[red]{e}[/]", str(e))
            return True
        self.hint("/task mcp_eur_threshold runs it live on your model · /report for every saved run")
        return True

    def cmd_tasks(self, args):
        if args and args[0] == "default":
            self.task_file = None
        elif args:
            try:
                tasks = evaluate.load_tasks(args[0])
            except (OSError, ValueError) as e:
                self.say(f"[red]can't read {args[0]}: {e}[/]", f"can't read {args[0]}: {e}")
                return True
            self.task_file = args[0]
            self.render_tasks(tasks)
            self.say(f"[dim]/task now uses {args[0]}; /tasks default switches back[/]",
                     f"/task now uses {args[0]}; /tasks default switches back")
            return True
        self.render_tasks(cli._all_tasks(self.task_file))
        return True

    def render_tasks(self, tasks):
        cli_view.render_task_list(self.view, tasks)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def cmd_mode(self, args):
        if not args or args[0] not in cli.MODES:
            for name, text in MODE_HELP.items():
                mark = "●" if name == self.mode else "○"
                self.say(f"  {mark} [bold]{name:<10}[/] {text}", f"  {mark} {name:<10} {text}")
            self.hint("/mode <name>")
            return True
        self.mode = args[0]
        self.say(f"live questions now run on [bold]{cli.MODES[self.mode]}[/]", f"live questions now run on {cli.MODES[self.mode]}")
        return True

    def _configure(self, **kwargs):
        current = agent.LLM
        settings = {"model": current.model, "reasoning_effort": current.reasoning_effort or None}
        if current.provider == "custom":
            settings["base_url"] = current.base_url
        else:
            settings["provider"] = current.provider
        settings.update(kwargs)
        try:
            agent.configure(**settings)
        except llm.ConfigError as e:
            self.say(f"[red]{e}[/]", str(e))
            return
        self.cmd_status([])

    def cmd_model(self, args):
        if not args:
            self.cmd_status([])
            self.hint("/model <name>, e.g. /model o4-mini (the provider stays the same)")
            return True
        self._configure(model=args[0])
        return True

    def cmd_provider(self, args):
        if not args or args[0] not in llm.PROVIDERS:
            width = max(len(url) for url, _, _ in llm.PROVIDERS.values())
            for name, (url, key_var, _) in llm.PROVIDERS.items():
                key = "no key needed" if key_var is None else ("✓ " if os.getenv(key_var) else "✗ ") + key_var
                self.say(f"  [bold]{name:<11}[/] {url:<{width}}  [dim]{key}[/]", f"  {name:<11} {url:<{width}}  {key}")
            self.hint("/provider <preset> <model>, e.g. /provider openai o4-mini")
            return True
        preset = llm.PROVIDERS[args[0]]
        model = args[1] if len(args) > 1 else preset[2]
        if not model:
            self.say(f"[yellow]name a model too, e.g. /provider {args[0]} <model>[/]",
                     f"name a model too, e.g. /provider {args[0]} <model>")
            return True
        settings = {"provider": args[0], "model": model, "reasoning_effort": agent.LLM.reasoning_effort or None}
        try:
            agent.configure(**settings)
        except llm.ConfigError as e:
            self.say(f"[red]{e}[/]", str(e))
            return True
        self.cmd_status([])
        return True

    def cmd_reasoning(self, args):
        if not args or args[0] not in REASONING:
            self.say(f"reasoning_effort is {agent.LLM.reasoning_effort or '(not sent)'}; "
                     f"choose one of {', '.join(REASONING)}")
            return True
        self._configure(reasoning_effort="" if args[0] == "off" else args[0])
        return True

    def cmd_status(self, args):
        cli_view.print_status(self.view, self.status())
        return True

    # ------------------------------------------------------------------
    # Benchmark, reports, checks
    # ------------------------------------------------------------------

    def cmd_report(self, args):
        if not args:
            runs = saved_runs()
            published = [r for r in runs if r.startswith("results/")]
            local = [r for r in runs if not r.startswith("results/")]
            shown = published + local[:LOCAL_RUNS_SHOWN]
            if self.view.rich:
                from rich.table import Table
                table = Table(title="saved runs · /report <run>", title_justify="left")
                for col in ("run", "model", "score"):
                    table.add_column(col)
                for run in shown:
                    if run == (local[:1] or [None])[0]:
                        table.add_section()
                    table.add_row(run, run_model(run), run_score(run))
                self.view.console.print(table)
            else:
                for run in shown:
                    print(f"  {run}  {run_model(run)}  {run_score(run)}")
            if len(local) > LOCAL_RUNS_SHOWN:
                self.say(f"[dim]{len(local) - LOCAL_RUNS_SHOWN} older local runs not shown (Tab completes them)[/]",
                         f"{len(local) - LOCAL_RUNS_SHOWN} older local runs not shown (Tab completes them)")
            return True
        evaluate.main(["--report", args[0]])
        return True

    def cmd_check(self, args):
        if not args:
            self.say("check which file? e.g. /check docs/examples/tasks.jsonl")
            return True
        evaluate.main(["--check", args[0]])
        return True

    def cmd_bench(self, args):
        suite = args[0] if args else "mcp"
        path = SUITES.get(suite, suite)
        try:
            trials = int(args[1]) if len(args) > 1 else 3
            tasks = evaluate.load_tasks(path)
        except ValueError:
            self.say("usage: /bench [mcp|rest|file] [trials]")
            return True
        except OSError as e:
            self.say(f"[red]can't read {path}: {e}[/]", f"can't read {path}: {e}")
            return True
        problem = cli.missing_key(tasks[0] if tasks else None)
        if problem:
            self.say(f"[red]{problem}[/]", problem)
            return True
        requests = len(tasks) * trials * REQUESTS_PER_TRIAL
        answer = self.read(f"{len(tasks)} tasks × {trials} trials on {agent.MODEL} ≈ {requests} model requests. "
                           "Continue? [y/N] ")
        if (answer or "").strip().lower() not in ("y", "yes"):
            self.say("not started")
            return True
        evaluate.main(["--tasks", str(path), "--trials", str(trials), *model_flags()])
        self.hint("/report lists this run next to the published ones")
        return True

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def cmd_help(self, args):
        self.say("Type a trading question to ask the agent live, or a command:",
                 "Type a trading question to ask the agent live, or a command:")
        for name, (_, usage, text) in self.commands.items():
            self.say(f"  [bold cyan]/{name}[/] [dim]{usage}[/]{' ' * max(1, 28 - len(name) - len(usage))}{text}",
                     f"  /{name} {usage}{' ' * max(1, 28 - len(name) - len(usage))}{text}")
        return True

    def cmd_clear(self, args):
        if self.view.rich:
            self.view.console.clear()
        self.view_banner()
        return True

    def cmd_exit(self, args):
        return False


def model_flags():
    """The shell's current model as litmus-bench flags, so a run records the
    model actually in use."""
    config = agent.LLM
    flags = ["--model", config.model]
    flags += ["--base-url", config.base_url] if config.provider == "custom" else ["--provider", config.provider]
    if config.reasoning_effort:
        flags += ["--reasoning-effort", config.reasoning_effort]
    return flags


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------

def candidates(shell, text):
    """(completion, description) pairs for the text before the cursor. Pure,
    so tests can check it without a terminal; Completer adapts it."""
    if not text.startswith("/"):
        return []
    parts = text[1:].split(" ")
    if len(parts) == 1:
        return [(f"/{name}", help_text) for name, (_, _, help_text) in shell.commands.items()
                if name.startswith(parts[0].lower())]
    command, prefix = parts[0].lower(), parts[-1]
    if len(parts) > 2 and command not in ("provider", "bench"):
        return []
    if command == "task":
        options = [(t["id"], t["question"][:60]) for t in cli._all_tasks(shell.task_file)]
    elif command == "tasks":
        options = [("default", "the shipped 52 tasks")] + [(f, "") for f in task_files()]
    elif command == "check":
        options = [(f, "") for f in task_files()]
    elif command == "mode":
        options = list(MODE_HELP.items())
    elif command == "provider" and len(parts) == 2:
        options = [(name, url) for name, (url, _, _) in llm.PROVIDERS.items()]
    elif command == "reasoning":
        options = [(r, "") for r in REASONING]
    elif command in ("report", "replay") and len(parts) == 2:
        options = [(r, run_score(r)) for r in saved_runs()]
    elif command == "bench" and len(parts) == 2:
        options = [("mcp", "8 CMC MCP tasks"), ("rest", "44 CMC REST tasks")] + [(f, "") for f in task_files()]
    else:
        return []
    return [(value, meta) for value, meta in options if value.startswith(prefix)]


try:
    from prompt_toolkit.completion import Completer as _Base, Completion

    class Completer(_Base):
        def __init__(self, shell):
            self.shell = shell

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            prefix = text.split(" ")[-1] if " " in text else text
            for value, meta in candidates(self.shell, text):
                yield Completion(value, start_position=-len(prefix), display_meta=meta)
except ImportError:  # pragma: no cover - the shell needs prompt_toolkit; the flags don't
    Completer = None
