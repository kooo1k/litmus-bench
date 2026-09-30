"""Bring your own agent: external_agent.py (the Litmus side of the protocol),
agent_protocol.py (the Python agent side) and litmus-bench --agent.

Every agent here is a real child process speaking JSON Lines, started with
this interpreter. The tools are the MCP suite's recording, so nothing
reaches a network or a model."""

import io
import json
import sys
import textwrap

import pytest

from litmus import agent_protocol, evaluate, external_agent
from litmus.score import score_trajectory

ROOT = evaluate.ROOT
EXAMPLE = ROOT / "docs" / "examples" / "agents" / "lookup_and_hold.py"
TASKS = {t["id"]: t for t in evaluate.load_tasks(evaluate.MCP_TASKS_PATH)}

# A correct, fully grounded SHORT for mcp_eur_threshold, written as a script
# that looks up ETH and EURC first, the way the task requires.
EUR_SHORT = """
from litmus.agent_protocol import serve

def decide(task, call_tool):
    eth = call_tool("get_crypto_quotes_latest", {"id": "1027"})[0]["price"]
    eurc = call_tool("get_crypto_quotes_latest", {"id": "20641"})[0]["price"]
    print(f"ETH {eth} USD / EURC {eurc} USD = {eth / eurc:.1f} EUR, under 2,650")
    return {"action": "SHORT", "symbol": "ETH", "confidence": 0.7, "position_size_pct": 0.02, "leverage": 1,
            "entry_price": eth, "stop_loss_price": round(eth * 1.03, 2), "time_horizon": "swing",
            "evidence": [{"claim": "ETH price in USD", "source_tool": "get_crypto_quotes_latest", "value": eth},
                         {"claim": "EURC price in USD", "source_tool": "get_crypto_quotes_latest", "value": eurc}],
            "reasoning": "ETH converted to euros is below the threshold."}

serve(decide)
"""


def script(tmp_path, body, name="agent.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return f'"{sys.executable}" "{path}"'


def run(command, task_id="mcp_eur_threshold", **kwargs):
    task = TASKS[task_id]
    trajectory = external_agent.run_external(command, task["question"], evaluate.build_tools(task),
                                             tools=evaluate.tool_definitions(task), **kwargs)
    return trajectory, score_trajectory(task, trajectory)


# --------------------------------------------------------------------------
# Graded by the same code as Litmus's own agent
# --------------------------------------------------------------------------

def test_a_correct_external_agent_passes_and_its_run_is_recorded(tmp_path):
    trajectory, score = run(script(tmp_path, EUR_SHORT))
    assert score["passed"], score["failures"]
    assert trajectory["model"] == "agent:agent.py"
    assert trajectory["agent"]["exit_code"] == 0
    assert [c["args"] for c in trajectory["tool_calls"]] == [{"id": "1027"}, {"id": "20641"}]
    assert "under 2,650" in trajectory["agent"]["stderr"]  # print() went to the log, not the protocol
    assert trajectory["risk_result"] is not None  # it went through the governor
    # logged as OpenAI chat messages, so the SFT export can read it
    roles = [m["role"] for m in trajectory["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert trajectory["messages"][-1]["tool_calls"][0]["function"]["name"] == "propose_trade"


def test_an_invented_number_fails_grounding_as_it_would_for_litmus_s_own_agent(tmp_path):
    body = EUR_SHORT.replace('"value": eurc}', '"value": 0.92}')  # a euro rate it never fetched
    _, score = run(script(tmp_path, body))
    assert not score["passed"] and "grounded" in score["failures"]


def test_the_example_agent_holds_with_grounded_evidence(tmp_path):
    command = f'"{sys.executable}" "{EXAMPLE}"'
    trajectory, score = run(command, "mcp_quote_outage")  # the quote tool is down: HOLD is right
    assert score["passed"], score["failures"]
    assert trajectory["model"] == "agent:lookup_and_hold.py"
    _, score = run(command, "mcp_eur_threshold")  # the right answer is SHORT
    assert not score["passed"] and "action" in score["failures"] and score["checks"]["grounded"]


def test_proposing_through_a_propose_trade_tool_call_also_works(tmp_path):
    body = """
    import json, sys
    sys.stdin.readline()
    print(json.dumps({"type": "tool_call", "name": "propose_trade", "arguments": json.dumps(
        {"action": "HOLD", "symbol": "ETH", "confidence": 0.2, "position_size_pct": 0, "leverage": 1,
         "time_horizon": "swing", "evidence": [], "reasoning": "no data"})}), flush=True)
    """
    trajectory, _ = run(script(tmp_path, body))
    assert trajectory["stop_reason"] == "proposal" and trajectory["outcome"] == "proposal"


def test_a_prose_answer_is_graded_like_a_model_that_never_proposed(tmp_path):
    body = """
    import json, sys
    sys.stdin.readline()
    print(json.dumps({"type": "final", "text": "ETH is about 2,500 euros, so short it."}), flush=True)
    """
    trajectory, score = run(script(tmp_path, body))
    assert trajectory["stop_reason"] == "text" and trajectory["outcome"] == "no_proposal"
    assert "completed" in score["failures"]
    assert trajectory["unverified_text_numbers"] == 1  # its prose is checked too


# --------------------------------------------------------------------------
# Agent failures are graded; failing to start is a harness error
# --------------------------------------------------------------------------

@pytest.mark.parametrize("body, reason", [
    ("""
    import sys
    sys.stdin.readline()
    print("thinking...", flush=True)
    """, "protocol_error"),
    ("""
    import json, sys
    sys.stdin.readline()
    print(json.dumps({"type": "hello"}), flush=True)
    """, "protocol_error"),
    ("""
    import json, sys
    sys.stdin.readline()
    print(json.dumps({"type": "tool_call", "name": "get_global_metrics_latest", "arguments": {}}), flush=True)
    sys.stdin.readline()
    sys.exit(3)
    """, "agent_exited"),
])
def test_agent_misbehavior_is_graded_not_excused(tmp_path, body, reason):
    trajectory, score = run(script(tmp_path, body))
    assert trajectory["stop_reason"] == reason
    assert trajectory["agent"]["error"]
    assert not score["passed"] and "completed" in score["failures"]


def test_a_hung_agent_times_out(tmp_path):
    body = """
    import sys, time
    sys.stdin.readline()
    time.sleep(30)
    """
    trajectory, score = run(script(tmp_path, body), timeout=1)
    assert trajectory["stop_reason"] == "timeout" and "completed" in score["failures"]


def test_endless_tool_calls_stop_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(external_agent, "MAX_TOOL_CALLS", 3)
    body = """
    import json, sys
    sys.stdin.readline()
    while True:
        print(json.dumps({"type": "tool_call", "name": "get_global_metrics_latest", "arguments": {}}), flush=True)
        if not sys.stdin.readline():
            break
    """
    trajectory, score = run(script(tmp_path, body))
    assert trajectory["stop_reason"] == "round_limit" and len(trajectory["tool_calls"]) == 3
    assert score["stop_reason"] == "round_limit"  # counted as "ran out of turns" in the report


def test_a_crash_before_the_first_message_is_a_start_error(tmp_path):
    with pytest.raises(external_agent.AgentStartError, match="ModuleNotFoundError"):
        run(script(tmp_path, "import no_such_module_anywhere\n"))


def test_a_missing_program_is_a_start_error():
    with pytest.raises(external_agent.AgentStartError, match="can't start agent"):
        run("no-such-program-anywhere --flag")


def test_label_names_the_script_or_program():
    assert external_agent.label("python docs/examples/agents/lookup_and_hold.py") == "agent:lookup_and_hold.py"
    assert external_agent.label('"C:\\Program Files\\Python\\python.exe" -u my_agent.py') == "agent:my_agent.py"
    assert external_agent.label("./trader") == "agent:trader"
    assert external_agent.label("node agent.js --fast") == "agent:agent.js"


# --------------------------------------------------------------------------
# agent_protocol.serve, the Python agent side
# --------------------------------------------------------------------------

def test_serve_speaks_the_protocol_and_keeps_prints_off_stdout(capsys):
    task = {"type": "task", "question": "q", "tools": [], "time": "t"}
    stdin = io.StringIO(json.dumps(task) + "\n" + json.dumps({"type": "tool_result", "result": {"price": 1}}) + "\n")
    stdout = io.StringIO()

    def decide(t, call_tool):
        print("debug line")
        return {"action": "HOLD", "seen": call_tool("get_price", {"symbol": "BTC"})}

    agent_protocol.serve(decide, stdin=stdin, stdout=stdout)
    sent = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert sent == [{"type": "tool_call", "name": "get_price", "arguments": {"symbol": "BTC"}},
                    {"type": "propose_trade", "arguments": {"action": "HOLD", "seen": {"price": 1}}}]
    assert "debug line" in capsys.readouterr().err


def test_serve_sends_a_string_answer_as_final():
    stdin, stdout = io.StringIO(json.dumps({"type": "task"}) + "\n"), io.StringIO()
    agent_protocol.serve(lambda t, call: "no trade", stdin=stdin, stdout=stdout)
    assert json.loads(stdout.getvalue()) == {"type": "final", "text": "no trade"}


# --------------------------------------------------------------------------
# litmus-bench --agent
# --------------------------------------------------------------------------

def test_benchmark_run_with_an_external_agent_needs_no_model_key(tmp_path, capsys):
    command = f'"{sys.executable}" "{EXAMPLE}"'
    code = evaluate.main(["--agent", command, "--tasks", str(evaluate.MCP_TASKS_PATH),
                          "--ids", "mcp_quote_outage", "mcp_eur_threshold"])
    assert code == 0
    run_dir = next(evaluate.OUT_ROOT.iterdir())
    config = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert config["agent"] == command and config["model"] == "agent:lookup_and_hold.py"
    assert config["provider"] is None and config["agent_prompt"] is None
    out = capsys.readouterr().out
    assert "mcp_quote_outage     1/1" in out and "mcp_eur_threshold    0/1" in out


def test_an_agent_that_cannot_start_stops_the_run(tmp_path, capsys):
    bad = script(tmp_path, "import no_such_module_anywhere\n")
    code = evaluate.main(["--agent", bad, "--tasks", str(evaluate.MCP_TASKS_PATH), "--trials", "2"])
    assert code == 2
    out = capsys.readouterr().out
    assert "ModuleNotFoundError" in out and "STOPPED: the agent could not start" in out
    run_dir = next(evaluate.OUT_ROOT.iterdir())
    assert len((run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()) == 1  # stopped at once


def test_resume_keeps_the_original_agent(tmp_path, capsys):
    command = f'"{sys.executable}" "{EXAMPLE}"'
    assert evaluate.main(["--agent", command, "--tasks", str(evaluate.MCP_TASKS_PATH),
                          "--ids", "mcp_quote_outage"]) == 0
    run_id = next(evaluate.OUT_ROOT.iterdir()).name
    assert evaluate.main(["--resume", run_id]) == 0  # no --agent given: taken from run.json
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", run_id, "--agent", "python other.py"])


@pytest.mark.parametrize("argv, message", [
    (["--agent", "python x.py", "--model", "gpt-6-sol"], "brings its own model"),
    (["--agent", "no-such-program-anywhere x.py"], "can't find"),
    (["--report", "results/2026-09-27_mcp_k3_flash-lite", "--agent", "python x.py"], "don't combine"),
])
def test_bad_agent_arguments_are_usage_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        evaluate.main(argv)
    assert e.value.code == 2 and message in capsys.readouterr().err
