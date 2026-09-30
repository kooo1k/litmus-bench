"""The terminal CLI (cli.main + cli_view) and evaluate.py --report, offline.

The model is scripted and CoinMarketCap is either the committed replay or a
monkeypatched MCP client, so nothing here spends quota. Output is captured
with a wide terminal so rich never wraps the lines being checked."""

import json

import openai
import pytest

from litmus import agent, cli, cli_view, evaluate, mcp_tools
from conftest import ROOT, propose

SCENARIO = ROOT / "tasks" / "mcp_snapshots" / "2026-09-27_live"
REPLAY = mcp_tools.replay_mcp_tools(SCENARIO)
ETH = REPLAY["get_crypto_quotes_latest"](id="1027")[0]["price"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Trajectories go to tmp, never the real runs/; a wide terminal for rich."""
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("COLUMNS", "220")


def trade(action, symbol="ETH", price=ETH, reasoning="scripted"):
    p = {"action": action, "symbol": symbol, "confidence": 0.8, "position_size_pct": 0.03 if action != "HOLD" else 0,
         "leverage": 1, "time_horizon": "intraday", "reasoning": reasoning,
         "evidence": [{"claim": "ETH price", "source_tool": "get_crypto_quotes_latest", "value": price}]}
    if action in ("LONG", "SHORT"):
        side = 1 if action == "LONG" else -1
        p.update(entry_price=price, stop_loss_price=round(price * (1 - 0.02 * side), 2),
                 take_profit_price=round(price * (1 + 0.04 * side), 2))
    return p


EUR_FAIL = [[("get_crypto_quotes_latest", {"id": "1027"})],
            [propose(trade("LONG", reasoning="USD/EUR is roughly at parity, so ETH is above €2,650."))]]
EUR_PASS = [[("get_crypto_quotes_latest", {"id": "1027,20641"})], [propose(trade("SHORT"))]]


# --------------------------------------------------------------------------
# One-line tool summaries
# --------------------------------------------------------------------------

def rec(name, args, result=None, error=None, kind=None):
    return {"name": name, "args": args, "result": result if error is None else {"error": error},
            "error": error, "error_kind": kind or ("tool" if error else None), "raw_arguments": json.dumps(args)}


@pytest.mark.parametrize("name, args, expect", [
    ("search_cryptos", {"query": "bit"}, "BIT (BitDAO) → id 11221"),
    ("get_crypto_quotes_latest", {"id": "1,1027"}, "ETH $2,714.95"),
    ("get_crypto_technical_analysis", {"id": "1027"}, "RSI14 63.93"),
    ("get_crypto_technical_analysis", {"id": "1", "includeFields": ["RSI"]}, "{} — empty"),
    ("get_crypto_marketcap_technical_analysis", {"id": "1027"}, "id ignored"),
    ("get_global_metrics_latest", {}, "Fear & Greed 72"),
    ("get_global_crypto_derivatives_metrics", {}, "no timestamp"),
    ("get_crypto_latest_news", {"id": "1027"}, "article(s); latest:"),
    ("trending_crypto_narratives", {}, "row(s)"),
    ("get_upcoming_macro_events", {}, "row(s)"),
])
def test_every_mcp_tool_has_a_readable_summary(name, args, expect):
    line = cli_view.summarize_tool(rec(name, args, REPLAY[name](**args)))
    assert expect in line and len(line) < 220


def test_error_skipped_and_unknown_tool_summaries():
    assert cli_view.summarize_tool(rec("x", {}, error=mcp_tools.OPAQUE_500)).startswith("ERROR ")
    assert "skipped" in cli_view.summarize_tool(rec("x", {}, kind="skipped"))
    assert cli_view.summarize_tool(rec("mystery_tool", {}, {"a": 1})) == '{"a": 1}'
    # a summarizer that chokes on an odd payload falls back instead of crashing the run
    assert cli_view.summarize_tool(rec("get_global_metrics_latest", {}, ["not", "a", "dict"]))


# --------------------------------------------------------------------------
# --task: run a benchmark task on its replay and grade it
# --------------------------------------------------------------------------

def test_task_failure_shows_the_governor_approved_it(scripted, capsys):
    scripted([EUR_FAIL])
    assert cli.main(["--task", "mcp_eur_threshold"]) == 1
    out = capsys.readouterr().out
    assert "benchmark mcp_eur_threshold: FAIL" in out
    assert "APPROVED" in out and "Only the benchmark caught it" in out
    assert "USD/EUR is roughly at parity" in out            # the false premise is on screen
    assert "[harness] entry_price" in out and "2,714.95" in out  # markup escaped, numbers formatted
    assert "✓ get_crypto_quotes_latest(id='1027')" in out


def test_task_pass(scripted, capsys):
    scripted([EUR_PASS])
    assert cli.main(["--task", "mcp_eur_threshold"]) == 0
    out = capsys.readouterr().out
    assert "benchmark mcp_eur_threshold: PASS" in out and "Only the benchmark caught it" not in out


def test_task_plain_output(scripted, capsys):
    scripted([EUR_FAIL])
    assert cli.main(["--task", "mcp_eur_threshold", "--plain"]) == 1
    out = capsys.readouterr().out
    assert "BENCHMARK mcp_eur_threshold: FAIL" in out and "Only the benchmark caught it" in out
    assert "\x1b[" not in out  # no terminal escape codes


def test_task_json_output(scripted, capsys):
    scripted([EUR_FAIL])
    assert cli.main(["--task", "mcp_eur_threshold", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["grade"]["passed"] is False and data["trajectory"]["outcome"] == "proposal"


def test_rest_task_runs_too(scripted, capsys):
    scripted([[[("get_price", {"symbol": "BTC"})], [propose({**trade("HOLD", "BTC"), "evidence": []})]]])
    assert cli.main(["--task", "clean_01", "--plain"]) == 0
    assert "BENCHMARK clean_01: PASS" in capsys.readouterr().out


def test_list_tasks_needs_no_model(capsys):
    assert cli.main(["--list-tasks", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "clean_01" in out and "mcp_eur_threshold" in out


# --------------------------------------------------------------------------
# Argument and configuration errors: clear, and before any model call
# --------------------------------------------------------------------------

def test_bare_litmus_without_a_terminal_prints_a_hint(capsys):
    """No arguments opens the interactive shell on a terminal; piped or in
    CI there is nobody to talk to, so it explains instead of failing."""
    assert cli.main([]) == 0
    assert "interactive shell" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--task", "no_such_task"],
    ["--task", "mcp_eur_threshold", "extra question"],
    ["--mcp", "--task", "clean_01"],             # modes are exclusive
    ["--provider", "nope", "q"],
])
def test_bad_arguments_exit_cleanly(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_missing_model_key_names_the_variable(monkeypatch, capsys):
    for name in ("LLM", "MODEL", "client"):
        monkeypatch.setattr(agent, name, getattr(agent, name))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--provider", "openai", "--model", "gpt-x", "q"])
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_live_mode_needs_a_cmc_key(monkeypatch, capsys):
    monkeypatch.setenv("CMC_API_KEY", "")
    with pytest.raises(SystemExit):
        cli.main(["--mcp", "q"])
    assert "CMC_API_KEY" in capsys.readouterr().err


def test_model_api_failure_is_a_message_not_a_traceback(scripted, capsys):
    scripted([openai.APIConnectionError(request=None)])
    assert cli.main(["--task", "mcp_eur_threshold"]) == 2
    out = capsys.readouterr().out
    assert "the model API failed" in out and "Traceback" not in out


# --------------------------------------------------------------------------
# --mcp: CoinMarketCap's MCP server, live (monkeypatched here)
# --------------------------------------------------------------------------

@pytest.fixture
def pinned_listing():
    return json.loads((SCENARIO / "tools_list.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_mcp(monkeypatch):
    """Tool calls answered from the replay, as the live server would."""
    def call_tool(name, arguments=None, spacing=0):
        result = REPLAY[name](**(arguments or {}))
        err = isinstance(result, dict) and "error" in result
        return {"is_error": err, "text": result["error"] if err else json.dumps(result), "json": None if err else result}
    monkeypatch.setattr(mcp_tools, "call_tool", call_tool)


HOLD_ETH = [[("get_crypto_quotes_latest", {"id": "1027"})], [propose(trade("HOLD"))]]


def test_mcp_live_uses_the_live_tool_list_and_says_it_matches(scripted, fake_mcp, monkeypatch, pinned_listing, capsys):
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": pinned_listing})
    scripted([HOLD_ETH])
    assert cli.main(["--mcp", "Should I trade ETH?"]) == 0
    out = capsys.readouterr().out
    assert "CoinMarketCap MCP · live" in out and "match the benchmark's pinned copy" in out
    assert "ETH $2,714.95" in out


def test_mcp_live_notices_changed_descriptions(scripted, fake_mcp, monkeypatch, pinned_listing, capsys):
    changed = json.loads(json.dumps(pinned_listing))
    changed["tools"][0]["description"] += " (edited)"
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": changed})
    scripted([HOLD_ETH])
    cli.main(["--mcp", "Should I trade ETH?"])
    assert "have changed since the benchmark recording" in capsys.readouterr().out


def test_mcp_live_falls_back_to_the_pinned_list(scripted, fake_mcp, monkeypatch, capsys):
    def down(method, params=None):
        raise ConnectionError("offline")
    monkeypatch.setattr(mcp_tools, "mcp_request", down)
    scripted([HOLD_ETH])
    assert cli.main(["--mcp", "Should I trade ETH?"]) == 0
    assert "showing the model the pinned copy" in capsys.readouterr().out


# --------------------------------------------------------------------------
# run_agent's event hook
# --------------------------------------------------------------------------

def test_on_event_reports_each_round_and_each_tool_call(scripted):
    events = []
    scripted([[[("search_cryptos", {"query": "ETH"}), ("get_crypto_quotes_latest", {"id": "1027"})],
               [propose(trade("HOLD"))]]])
    agent.run_agent("q", tool_functions=REPLAY, tools=mcp_tools.replay_definitions(SCENARIO), save=False,
                    verbose=False, on_event=lambda kind, data: events.append((kind, data.get("round") or data.get("name"))))
    assert events == [("model", 1), ("tool", "search_cryptos"), ("tool", "get_crypto_quotes_latest"), ("model", 2)]


# --------------------------------------------------------------------------
# evaluate.py --report
# --------------------------------------------------------------------------

def test_report_reprints_a_run_without_a_model_or_key(scripted, capsys, monkeypatch):
    scripted([[[("get_price", {"symbol": "BTC"})], [propose({**trade("HOLD", "BTC"), "evidence": []})]]])
    evaluate.main(["--ids", "clean_01"])
    end_of_run = capsys.readouterr().out
    run_id = next(evaluate.OUT_ROOT.iterdir()).name
    monkeypatch.setattr(agent, "client", None)  # any model call would now crash
    assert evaluate.main(["--report", run_id]) == 0
    reprint = capsys.readouterr().out
    # the report table's row (not the per-trial progress line, "clean_01  t1  PASS ...")
    row = next(line for line in end_of_run.splitlines() if line.startswith("clean_01 ") and "1/1" in line)
    assert row in reprint.splitlines()
    overall = next(line for line in end_of_run.splitlines() if line.startswith("OVERALL"))
    assert overall in reprint.splitlines()


def test_report_reads_a_run_directory_saved_on_windows(scripted, capsys, monkeypatch, tmp_path):
    """The published runs in results/ are reprinted by path on any OS: their
    run.json was written on Windows, with a backslash tasks_path."""
    scripted([[[("get_price", {"symbol": "BTC"})], [propose({**trade("HOLD", "BTC"), "evidence": []})]]])
    evaluate.main(["--ids", "clean_01"])
    capsys.readouterr()
    saved = next(evaluate.OUT_ROOT.iterdir())
    published = tmp_path / "published_run"
    published.mkdir()
    for name in (evaluate.RUN_FILE, evaluate.TRIALS_FILE):
        (published / name).write_bytes((saved / name).read_bytes())
    config = json.loads((published / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    assert config["tasks_path"] == "tasks/benchmark.jsonl"  # new runs: relative to the repo, forward slashes
    config["tasks_path"] = "tasks\\benchmark.jsonl"
    (published / evaluate.RUN_FILE).write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # not the repo root: the tasks path must resolve against the repo anyway
    monkeypatch.setattr(agent, "client", None)
    assert evaluate.main(["--report", str(published)]) == 0
    assert any(line.startswith("clean_01 ") and "1/1" in line for line in capsys.readouterr().out.splitlines())


def test_report_names_where_it_looked():
    with pytest.raises(SystemExit):
        evaluate.main(["--report", "no_such_run"])


def test_report_refuses_other_flags():
    with pytest.raises(SystemExit):
        evaluate.main(["--report", "x", "--trials", "3"])


def test_long_evidence_claims_are_shown_in_full(scripted, capsys):
    """The false premise often sits at the END of a claim (live run 2026-09-27:
    '... above €2,650 (assuming roughly parity or higher ...)'); cutting claims
    short hid it."""
    premise = "which is above 2,650 euros assuming roughly parity or higher, and definitely above 2,650 units"
    p = trade("LONG")
    p["evidence"] = [{"claim": f"ETH price is $2,714.95 {premise}", "source_tool": "get_crypto_quotes_latest", "value": ETH}]
    scripted([[[("get_crypto_quotes_latest", {"id": "1027"})], [propose(p)]]])
    cli.main(["--task", "mcp_eur_threshold"])
    out = capsys.readouterr().out
    # the claim wraps across table rows, so check its last word, which the old
    # 60-character cut dropped; nothing else on screen contains it
    assert "units" in out


# --------------------------------------------------------------------------
# --mcp --all-tools: every tool on CMC's server, live only
# --------------------------------------------------------------------------

def test_all_tools_offers_every_listed_tool(monkeypatch, pinned_listing):
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": pinned_listing})
    listed = [t["name"] for t in pinned_listing["tools"]]
    definitions, functions, _ = cli._live_mcp_setup(all_tools=True)
    assert [d["function"]["name"] for d in definitions] == listed and set(functions) == set(listed)
    assert {"find_skill", "execute_skill", "get_crypto_info"} <= set(functions)
    definitions, functions, _ = cli._live_mcp_setup()
    assert [d["function"]["name"] for d in definitions] == list(mcp_tools.BENCHMARK_TOOLS) == list(functions)


def test_all_tools_falls_back_to_every_pinned_tool(monkeypatch, pinned_listing):
    def down(method, params=None):
        raise ConnectionError("offline")
    monkeypatch.setattr(mcp_tools, "mcp_request", down)
    definitions, functions, notes = cli._live_mcp_setup(all_tools=True)
    assert len(definitions) == len(pinned_listing["tools"]) == len(functions)
    assert "pinned copy" in notes[0]


def test_execute_skill_gets_the_long_timeout(monkeypatch):
    seen = {}

    def request(method, params=None, timeout=None):
        seen[params["name"]] = timeout
        return {"result": {"content": [{"type": "text", "text": "{}"}]}}
    monkeypatch.setattr(mcp_tools, "mcp_request", request)
    mcp_tools.call_tool("execute_skill", {"unique_name": "x"}, spacing=0)
    mcp_tools.call_tool("get_crypto_quotes_latest", {"id": "1"}, spacing=0)
    assert seen == {"execute_skill": mcp_tools.SLOW_TOOL_TIMEOUTS["execute_skill"],
                    "get_crypto_quotes_latest": mcp_tools.REQUEST_TIMEOUT_SECONDS}


def test_all_tools_needs_mcp():
    with pytest.raises(SystemExit):
        cli.main(["--all-tools", "--task", "mcp_eur_threshold"])


def test_all_tools_run_says_what_it_offers(scripted, fake_mcp, monkeypatch, pinned_listing, capsys):
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": pinned_listing})
    scripted([HOLD_ETH])
    assert cli.main(["--mcp", "--all-tools", "--plain", "Should I trade ETH?"]) == 0
    out = capsys.readouterr().out
    assert f"live · {len(pinned_listing['tools'])} tools" in out and "execute_skill runs hosted skills" in out


def test_find_skill_summary_on_the_recorded_probe():
    raw = json.loads((ROOT / "runs" / "mcp_probe" / "20260927T084543Z" / "find_skill.json").read_text(encoding="utf-8"))
    line = cli_view.summarize_tool({"name": "find_skill", "result": mcp_tools.as_agent_result(raw["find_skill"]), "args": {}})
    assert line.startswith(("1 skill", "2 skill", "3 skill", "4 skill", "5 skill")) and "detect_funding_rate_regime_shift" in line


def test_plain_output_survives_a_cp1252_console(scripted, fake_mcp, monkeypatch, pinned_listing):
    """Found live 2026-09-28: piped on Windows, stdout is cp1252, which has no
    "→"; the tool summary crashed the run after the first model call."""
    import io
    import sys
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": pinned_listing})
    scripted([[[("search_cryptos", {"query": "eth"})], [propose(trade("HOLD"))]]])
    assert cli.main(["--mcp", "--plain", "Should I trade ETH?"]) == 0
    out.flush()
    assert b"?" in out.buffer.getvalue()  # the arrow became a placeholder instead of an exception
