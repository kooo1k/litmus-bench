"""
external_agent.py

Bring your own agent: grade an agent Litmus didn't write, with the same
tasks, tools, risk governor and grader as Litmus's own.

Your agent runs as its own process, in any language, and talks to Litmus
in JSON Lines (one JSON object per line) on its stdin and stdout. Litmus
starts one process per trial, so no state leaks between trials:

  Litmus -> agent  {"type": "task", "protocol": 1, "question": "...",
                    "tools": [OpenAI function definitions, propose_trade included],
                    "time": "2026-09-27T12:00:00Z"}
  agent -> Litmus  {"type": "tool_call", "name": "get_crypto_quotes_latest", "arguments": {"id": "1027"}}
  Litmus -> agent  {"type": "tool_result", "name": "get_crypto_quotes_latest", "result": {...}}
                   (a failed call arrives as data: {"error": "..."}, as a model would see it)
  ... as many calls as the agent needs, up to MAX_TOOL_CALLS ...
  agent -> Litmus  {"type": "propose_trade", "arguments": {propose_trade's fields}}
             or    {"type": "final", "text": "..."}   (answered in prose, with no trade)

Your agent must answer from the tools Litmus hands it. Those are the task's
recorded data with its faults injected, and every call goes through Litmus,
which is how the grader knows what the agent saw. Whatever the agent writes
to stderr is kept in the trajectory as its log. litmus/agent_protocol.py
implements the agent side for Python in about 30 lines; see
docs/examples/agents/ for a runnable agent.

Whose failure is it? Failing to start (a bad command, or a crash before the
agent sends anything, e.g. an ImportError) is a harness error: it is left
out of every rate, like an API outage, and stops the benchmark run, since
every trial would fail the same way. Anything after the agent's first
message is its own behavior and is graded: hanging past the timeout
("timeout"), exiting without an answer ("agent_exited"), writing something
that isn't the protocol ("protocol_error"), or calling tools without end
("round_limit", the same stop reason as Litmus's own agent running out of
turns). None of those ends in a proposal, so each fails "completed".
"""

import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from litmus import agent

PROTOCOL_VERSION = 1

# Litmus's own agent gets 5 model turns, and can batch several calls into
# one. An external agent's calls arrive one at a time, so the cap is on calls:
# a 3-coin comparison with one retry each fits well under it.
MAX_TOOL_CALLS = 20

# A whole trial, from start to answer. An agent calling a hosted model can
# take a while; one that hasn't answered in two minutes is hung.
TIMEOUT_SECONDS = 120

STDERR_KEPT = 4000  # characters of the agent's log kept in the trajectory


class AgentStartError(RuntimeError):
    """The agent process could not be started: a harness error, not agent
    behavior."""


def split_command(command):
    """An --agent string -> argv. POSIX quoting everywhere, so a command
    written in the docs works the same on Windows; backslashes in a Windows
    path are kept as path separators."""
    return shlex.split(command.replace("\\", "/"))


def program_path(program):
    """The program to start: a file path made absolute (Windows won't start
    a relative path written with forward slashes), else what PATH finds."""
    if Path(program).is_file():
        return str(Path(program).resolve())
    return shutil.which(program) or program


def label(command):
    """A short name for the agent in reports: agent:<script or program name>."""
    argv = split_command(command)
    script = next((a for a in argv[1:] if not a.startswith("-")), None) if len(argv) > 1 else None
    return f"agent:{Path(script or argv[0]).name}"


def _chat_call(n, name, arguments):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{n}", "type": "function",
                            "function": {"name": name, "arguments": arguments}}]}


def run_external(command, question, tool_functions, tools=None, portfolio=None, market=None,
                 timeout=TIMEOUT_SECONDS, on_event=None, save=False):
    """Run one question through an external agent process and return its
    trajectory, in the same shape run_agent returns (so score_trajectory,
    the report and the replay all read it unchanged). The conversation is
    logged in OpenAI chat format, with no system message: the agent's own
    prompt, if it has one, is its business."""
    tools = list(agent.TOOLS if tools is None else tools)
    if agent.PROPOSE_TRADE not in [t["function"]["name"] for t in tools]:
        tools.append(agent.propose_trade_function)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    trajectory = agent.new_trajectory(question, label(command), tools, started_at)
    trajectory["tool_definitions"] = tools
    trajectory["agent"] = {"type": "external", "command": command, "protocol": PROTOCOL_VERSION}
    messages = [{"role": "user", "content": question}]

    argv = split_command(command)
    try:
        proc = subprocess.Popen([program_path(argv[0]), *argv[1:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except OSError as e:
        raise AgentStartError(f"can't start agent {command!r}: {e}") from e

    # Reader threads, so a silent agent can be timed out (a blocking read on
    # a pipe can't be) and a chatty stderr can't fill its pipe and deadlock.
    lines, stderr = queue.Queue(), []

    def read_stdout():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF: the agent exited or closed stdout

    readers = [threading.Thread(target=read_stdout, daemon=True),
               threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True)]
    for reader in readers:
        reader.start()

    def send(message):
        proc.stdin.write(json.dumps(message, default=str) + "\n")
        proc.stdin.flush()

    raw_proposal, calls, received = None, 0, 0

    def stop(reason, detail=None):
        trajectory["stop_reason"] = reason
        if detail:
            trajectory["agent"]["error"] = detail

    try:
        send({"type": "task", "protocol": PROTOCOL_VERSION, "question": question, "tools": tools,
              "time": started_at.strftime("%Y-%m-%dT%H:%M:%SZ")})
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                stop("timeout", f"no answer within {timeout}s")
                break
            if line is None:
                stop("agent_exited", f"the agent exited (code {proc.poll()}) without proposing or answering")
                break
            if not line.strip():
                continue
            received += 1
            try:
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError("not a JSON object")
            except ValueError:
                stop("protocol_error", f"not a protocol message: {line.strip()[:200]}")
                break
            kind = msg.get("type")
            arguments = msg.get("arguments", {})
            arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
            if kind == "tool_call" and msg.get("name") == agent.PROPOSE_TRADE:
                kind = "propose_trade"  # proposing through the tool is fine too
            if kind == "tool_call":
                if calls == MAX_TOOL_CALLS:
                    stop("round_limit", f"more than {MAX_TOOL_CALLS} tool calls")
                    break
                calls += 1
                name = str(msg.get("name"))
                record = agent.execute_tool_call(
                    SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments)), tool_functions)
                trajectory["tool_calls"].append(record)
                if on_event:
                    on_event("tool", record)
                content = json.dumps(record["result"], default=str)
                messages += [_chat_call(calls, name, arguments),
                             {"role": "tool", "tool_call_id": f"call_{calls}", "content": content}]
                send({"type": "tool_result", "name": name, "result": record["result"]})
            elif kind == "propose_trade":
                raw_proposal = arguments
                messages.append(_chat_call(calls + 1, agent.PROPOSE_TRADE, arguments))
                stop("proposal")
                break
            elif kind == "final":
                trajectory["final_text"] = str(msg.get("text") or "")
                messages.append({"role": "assistant", "content": trajectory["final_text"]})
                stop("text")
                break
            else:
                stop("protocol_error", f"unknown message type {kind!r}")
                break
    except (BrokenPipeError, OSError) as e:  # the agent closed stdin: it's gone
        if trajectory["stop_reason"] is None:
            stop("agent_exited", f"the agent stopped reading ({type(e).__name__})")
    finally:
        try:
            proc.stdin.close()  # EOF tells a well-behaved agent to exit
        except OSError:
            pass
        if trajectory["stop_reason"] == "timeout":
            proc.kill()  # it's hung: don't wait for it to notice EOF
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for reader in readers:
            reader.join(timeout=2)
    if received == 0 and proc.returncode and trajectory["stop_reason"] == "agent_exited":
        # It crashed before saying a word (an import error, a bad path): a
        # setup problem, like a model API that can't be reached, not
        # behavior worth grading. Once it has spoken, a crash is graded.
        raise AgentStartError(f"agent {command!r} exited with code {proc.returncode} before sending anything:\n"
                              + "".join(stderr)[-1500:].rstrip())

    trajectory["agent"]["exit_code"] = proc.returncode
    trajectory["agent"]["stderr"] = "".join(stderr)[-STDERR_KEPT:]
    trajectory["messages"] = messages
    return agent.finish_trajectory(trajectory, raw_proposal, portfolio, market, started, save)
