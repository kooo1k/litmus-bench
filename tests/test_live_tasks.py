"""live_tasks.py: tasks graded on today's market. The live prices here come
from the committed MCP recording (ETH $2,714.95, EURC $1.1387) or from
hand-set numbers, served through a monkeypatched MCP client, so nothing
reaches CoinMarketCap and the model is scripted."""

import json

import pytest

from litmus import agent, cli, live_tasks, mcp_tools
from conftest import ROOT, propose

SCENARIO = ROOT / "tasks" / "mcp_snapshots" / "2026-09-27_live"
REPLAY = mcp_tools.replay_mcp_tools(SCENARIO)
ETH = REPLAY["get_crypto_quotes_latest"](id="1027")[0]["price"]     # $2,714.95
EURC = REPLAY["get_crypto_quotes_latest"](id="20641")[0]["price"]   # $1.1387


def quotes(eth, eurc):
    """A stand-in for CMC's live quote tool with these prices."""
    def tool(id):
        rows = {"1027": {"id": "1027", "symbol": "ETH", "price": eth},
                "20641": {"id": "20641", "symbol": "EURC", "price": eurc}}
        return [rows[i] for i in id.split(",") if i in rows]
    return tool


# --------------------------------------------------------------------------
# Building today's task
# --------------------------------------------------------------------------

def test_threshold_sits_between_the_euro_and_the_dollar_readings():
    task = live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(ETH, EURC), long_if="more")
    ref = task["reference"]
    assert round(ref["eth_eur"]) == 2384 and ref["threshold"] == 2550
    assert ref["eth_eur"] < ref["threshold"] < ETH  # the right reading and the misread on opposite sides
    assert task["question"] == ("If 1 ETH is worth more than €2,550 right now, go long ETH; "
                                "if it's worth less, short it.")
    assert task["expected"]["acceptable_actions"] == ["SHORT"]
    assert "get_crypto_quotes_latest(EURC)" in task["expected"]["required_tools"]
    assert "7.0% below €2,550: SHORT" in task["rationale"] and "6.5% above it: LONG" in task["rationale"]


def test_the_flipped_question_makes_long_right_and_keeps_the_trap():
    task = live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(ETH, EURC), long_if="less")
    assert task["question"] == ("If 1 ETH is worth less than €2,550 right now, go long ETH; "
                                "if it's worth more, short it.")
    assert task["expected"]["acceptable_actions"] == ["LONG"]
    assert "below €2,550: LONG" in task["rationale"] and "above it: SHORT" in task["rationale"]


def test_the_direction_is_drawn_at_random(monkeypatch):
    """So an agent that always makes the same trade can't pass."""
    drawn = {live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(ETH, EURC))["expected"]["acceptable_actions"][0]
             for _ in range(40)}
    assert drawn == {"LONG", "SHORT"}  # 40 draws: missing one side has probability 2^-39
    with pytest.raises(ValueError):
        live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(ETH, EURC), long_if="sideways")


def test_threshold_follows_the_market():
    task = live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(4000.0, 1.08))
    ref = task["reference"]
    assert ref["eth_eur"] < ref["threshold"] < 4000.0 and ref["threshold"] % 10 == 0


@pytest.mark.parametrize("tool, message", [
    (quotes(ETH, 1.005), "too close"),                                         # euro ~ dollar: no trap to test
    (quotes(ETH, None), "didn't return both prices"),
    (lambda id: {"error": "401 unauthorized"}, "401 unauthorized"),
    (lambda id: (_ for _ in ()).throw(ConnectionError("down")), "ConnectionError"),
])
def test_refuses_rather_than_grade_a_coin_flip(tool, message):
    with pytest.raises(live_tasks.LiveTaskUnavailable, match=message):
        live_tasks.build(live_tasks.EUR_THRESHOLD, tool)


def test_recheck_flags_a_market_that_crossed_the_line():
    task = live_tasks.build(live_tasks.EUR_THRESHOLD, quotes(ETH, EURC), long_if="more")
    assert live_tasks.recheck(task, quotes(ETH * 1.001, EURC)) is None
    assert "rose to €" in live_tasks.recheck(task, quotes(ETH * 1.08, EURC))
    assert "fell to $" in live_tasks.recheck(task, quotes(2500.0, EURC))
    assert live_tasks.recheck(task, lambda id: {"error": "timeout"}) is None  # can't check: the grade stands


# --------------------------------------------------------------------------
# litmus --task live_eur_threshold, end to end
# --------------------------------------------------------------------------

@pytest.fixture
def live_mcp(monkeypatch, tmp_path):
    """CMC's MCP server answered from the recording, as the live server would."""
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("COLUMNS", "220")
    listing = json.loads((SCENARIO / "tools_list.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(mcp_tools, "mcp_request", lambda method, params=None: {"result": listing})

    def call_tool(name, arguments=None, spacing=0):
        result = REPLAY[name](**(arguments or {}))
        return {"is_error": False, "text": json.dumps(result), "json": result}
    monkeypatch.setattr(mcp_tools, "call_tool", call_tool)

    def direction(long_if):
        monkeypatch.setattr(live_tasks.random, "choice", lambda options: long_if)
    direction("more")  # the headline's wording, unless a test flips it
    return direction


def trade(action):
    p = {"action": action, "symbol": "ETH", "confidence": 0.8, "position_size_pct": 0.03, "leverage": 1,
         "time_horizon": "intraday", "reasoning": "scripted", "entry_price": ETH,
         "stop_loss_price": round(ETH * (1.02 if action == "SHORT" else 0.98), 2),
         "take_profit_price": round(ETH * (0.96 if action == "SHORT" else 1.04), 2),
         "evidence": [{"claim": "ETH price", "source_tool": "get_crypto_quotes_latest", "value": ETH},
                      {"claim": "EURC price", "source_tool": "get_crypto_quotes_latest", "value": EURC}]}
    return p


def test_a_converting_agent_passes_on_today_s_market(scripted, live_mcp, capsys):
    scripted([[[("get_crypto_quotes_latest", {"id": "1027,20641"})], [propose(trade("SHORT"))]]])
    assert cli.main(["--task", "live_eur_threshold"]) == 0
    out = capsys.readouterr().out
    assert "live_eur_threshold · CoinMarketCap MCP · live" in out
    assert "€2,550" in out and "graded on today's market" in out
    assert "PASS" in out
    saved = json.loads(next((agent.RUNS_DIR).glob("*.json")).read_text(encoding="utf-8"))
    assert saved["live_task"]["reference"]["threshold"] == 2550  # the task existed only for this run
    assert saved["live_task"]["grade"]["passed"] and saved["live_task"]["grade"]["live_inconclusive"] is None


def test_the_euro_misread_fails_on_today_s_market(scripted, live_mcp, capsys):
    scripted([[[("get_crypto_quotes_latest", {"id": "1027"})], [propose(trade("LONG"))]]])
    assert cli.main(["--task", "live_eur_threshold"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_on_the_flipped_question_the_same_short_fails_and_long_passes(scripted, live_mcp, capsys):
    """An agent that always shorts ETH would pass only half the time."""
    live_mcp("less")
    convert = [("get_crypto_quotes_latest", {"id": "1027,20641"})]
    scripted([[convert, [propose(trade("SHORT"))]]])
    assert cli.main(["--task", "live_eur_threshold"]) == 1
    assert "worth less than €2,550" in capsys.readouterr().out
    scripted([[convert, [propose(trade("LONG"))]]])
    assert cli.main(["--task", "live_eur_threshold"]) == 0


def test_live_task_needs_a_cmc_key(monkeypatch, capsys):
    monkeypatch.setenv("CMC_API_KEY", "")
    with pytest.raises(SystemExit):
        cli.main(["--task", "live_eur_threshold"])
    assert "CMC_API_KEY" in capsys.readouterr().err


def test_live_task_is_listed_and_completes_in_the_shell():
    assert "live_eur_threshold" in [t["id"] for t in cli._all_tasks()]
    assert all(not t.get("live") for t in cli._all_tasks(str(ROOT / "tasks" / "mcp_benchmark.jsonl")))
