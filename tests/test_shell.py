"""The interactive shell (`litmus` with no arguments), offline.

Driven by scripted input lines through Shell(read_line=...), with the model
scripted and CoinMarketCap replayed, plus one test through a real
prompt_toolkit session on a pipe, to prove the terminal wiring."""

import json

import openai
import pytest

from litmus import agent, cli, cli_view, evaluate, llm, mcp_tools, shell
from conftest import ROOT, propose
from test_cli import EUR_FAIL, REPLAY, trade

PINNED = json.loads((ROOT / "tasks" / "mcp_snapshots" / "2026-09-27_live" / "tools_list.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("COLUMNS", "220")
    for name in ("LLM", "MODEL", "client"):  # /model and /provider rebind these
        monkeypatch.setattr(agent, name, getattr(agent, name))
    monkeypatch.setattr(evaluate, "MODEL", evaluate.MODEL)


@pytest.fixture
def live_mcp(monkeypatch):
    """CMC's MCP server answered from the recording, as the live server would."""
    def call_tool(name, arguments=None, spacing=0):
        result = REPLAY[name](**(arguments or {}))
        err = isinstance(result, dict) and "error" in result
        return {"is_error": err, "text": result["error"] if err else json.dumps(result), "json": None if err else result}
    monkeypatch.setattr(mcp_tools, "call_tool", call_tool)
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": PINNED})
    monkeypatch.setenv("CMC_API_KEY", "test-key-never-sent")


def drive(lines, capsys=None):
    """Run a plain-text shell over `lines`; returns (exit code, output)."""
    feed = iter(lines)
    sh = shell.Shell(read_line=lambda prompt: next(feed, None), view=cli_view.LiveView(plain=True))
    code = sh.run()
    return code, (capsys.readouterr().out if capsys else "")


def test_banner_help_and_exit(capsys):
    code, out = drive(["/help", "/exit", "/never reached"], capsys)
    assert code == 0
    assert "LITMUS" in out and agent.MODEL in out and "CoinMarketCap MCP" in out
    for name in ("/task", "/tasks", "/mode", "/model", "/provider", "/report", "/bench", "/check", "/exit"):
        assert name in out
    assert "never reached" not in out and out.rstrip().endswith("bye")


def test_rich_banner_draws_the_logo(monkeypatch):
    from rich.console import Console
    view = cli_view.LiveView()
    view.console = Console(record=True, width=120, color_system=None)
    monkeypatch.setattr(cli_view, "_can_draw_blocks", lambda: True)
    cli_view.render_banner(view, shell.Shell(read_line=lambda p: None, view=view).status())
    text = view.console.export_text()
    assert cli_view.logo_lines()[0] in text and "/task mcp_eur_threshold" in text and "prompt" in text


def test_a_plain_question_runs_live_on_mcp(scripted, live_mcp, capsys):
    scripted([[[("get_crypto_quotes_latest", {"id": "1027"})], [propose(trade("HOLD"))]]])
    code, out = drive(["Should I trade ETH?"], capsys)
    assert code == 0 and "CoinMarketCap MCP · live" in out and "ETH $2,714.95" in out
    assert "next:" in out  # the next-step hint


def test_task_is_graded_with_the_governor_line(scripted, capsys):
    scripted([EUR_FAIL])
    _, out = drive(["/task mcp_eur_threshold"], capsys)
    assert "BENCHMARK mcp_eur_threshold: FAIL" in out and "Only the benchmark caught it" in out


def test_task_without_an_id_lists_and_a_typo_gets_a_suggestion(capsys):
    _, out = drive(["/task", "/task mcp_eur_treshold"], capsys)
    assert "mcp_eur_threshold" in out and "Did you mean mcp_eur_threshold?" in out


def test_mode_switches_what_questions_run_on(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "run_once", lambda view, q, task=None, mode="rest", as_json=False: seen.append(mode))
    monkeypatch.setenv("CMC_API_KEY", "k")
    drive(["q1", "/mode rest", "q2", "/mode all-tools", "q3", "/mode"], capsys)
    assert seen == ["mcp", "rest", "all-tools"]


def test_switching_to_a_provider_without_a_key_says_what_to_set(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    _, out = drive(["/provider openai gpt-x", "/status"], capsys)
    assert agent.LLM.provider == "openai" and agent.MODEL == "gpt-x"
    assert "key NO (OPENAI_API_KEY)" in out


def test_model_keeps_the_provider_and_reasoning_is_sent(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    drive(["/provider openai gpt-x", "/model gpt-y", "/reasoning none"], capsys)
    assert (agent.LLM.provider, agent.MODEL, agent.LLM.reasoning_effort) == ("openai", "gpt-y", "none")
    assert shell.model_flags() == ["--model", "gpt-y", "--provider", "openai", "--reasoning-effort", "none"]
    drive(["/reasoning off"], capsys)
    assert agent.LLM.reasoning_effort == ""


def test_unknown_command_gets_a_suggestion(capsys):
    _, out = drive(["/tsk"], capsys)
    assert "Did you mean /task?" in out


def test_a_model_api_error_does_not_end_the_shell(scripted, live_mcp, capsys):
    scripted([openai.APIConnectionError(request=None)])
    code, out = drive(["Should I trade ETH?", "/status"], capsys)
    assert code == 0 and "model   " in out  # /status still ran after the failure


def test_report_reprints_a_published_run(capsys):
    _, out = drive(["/report", "/report results/2026-09-27_mcp_k3_flash-lite"], capsys)
    listing = next(line for line in out.splitlines() if line.strip().startswith("results/2026-09-27_mcp_k3_flash-lite"))
    assert "gemini-3.5-flash-lite" in listing and listing.rstrip().endswith("16/24")
    assert any(line.startswith("OVERALL") and "16/24" in line for line in out.splitlines())


def test_bench_asks_before_spending_and_no_means_no(capsys):
    feed = iter(["/bench mcp 3", "n"])
    sh = shell.Shell(read_line=lambda prompt: next(feed, None), view=cli_view.LiveView(plain=True))
    sh.run()
    assert "not started" in capsys.readouterr().out  # the model was never called (it would raise)


def test_completions():
    sh = shell.Shell(read_line=lambda p: None, view=cli_view.LiveView(plain=True))
    names = lambda text: [value for value, _ in shell.candidates(sh, text)]
    assert names("/ta") == ["/task", "/tasks"]
    assert "mcp_eur_threshold" in names("/task mcp_e")
    assert set(names("/provider o")) == {"openai", "openrouter", "ollama"}
    assert "results/2026-09-27_mcp_k3_flash-lite" in names("/report results/2026-09-27")
    assert names("/mode r") == ["rest"]
    assert names("hello") == [] and names("/task mcp_eur_threshold extra") == []


def test_real_prompt_toolkit_session_on_a_pipe(capsys):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as pipe:
        pipe.send_text("/status\r/exit\r")
        with create_app_session(input=pipe, output=DummyOutput()):
            code = shell.Shell(view=cli_view.LiveView(plain=True)).run()
    out = capsys.readouterr().out
    assert code == 0 and "model   " in out and out.rstrip().endswith("bye")


# --------------------------------------------------------------------------
# Replays of saved runs: no model call, no key
# --------------------------------------------------------------------------

def test_replay_plays_the_headline_run_with_no_model(monkeypatch, capsys):
    monkeypatch.setattr(agent, "client", None)  # any model call would crash
    assert cli.main(["--replay", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "gemini-3.5-flash-lite @ gemini" in out and "no model call, no API key" in out
    assert "BENCHMARK mcp_eur_threshold: FAIL" in out and "Only the benchmark caught it" in out


def test_replay_a_chosen_trial_and_a_bad_one(capsys):
    assert cli.main(["--replay", "results/2026-09-28_eur_patch_experiment_v2.1/o4-mini",
                     "mcp_eur_threshold_eurc_hint", "1", "--plain"]) == 0
    assert "BENCHMARK mcp_eur_threshold_eurc_hint: PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--replay", "results/no_such_run"])


def test_shell_replay_and_its_completion(monkeypatch, capsys):
    monkeypatch.setattr(agent, "client", None)
    _, out = drive(["/replay", "/replay results/no_such_run"], capsys)
    assert "BENCHMARK mcp_eur_threshold: FAIL" in out and "no saved run" in out
    sh = shell.Shell(read_line=lambda p: None, view=cli_view.LiveView(plain=True))
    assert "results/2026-09-27_mcp_k3_flash-lite" in [v for v, _ in shell.candidates(sh, "/replay results/2026-09-27")]
