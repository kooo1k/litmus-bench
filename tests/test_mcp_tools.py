"""mcp_tools.py and the MCP path through agent -> governor -> scorer.

Replay fidelity is pinned against the committed recording
tasks/mcp_snapshots/2026-09-27_live: each quirk probe_cmc_mcp.py confirmed
live must survive the replay, since those quirks are what MCP tasks test.
The end-to-end cases check the whole MCP path: an MCP
task run through evaluate.run_task with a scripted model, graded by the
real scorer, with the real governor in the loop."""

import inspect
import json
from datetime import datetime, timezone

import pytest

from litmus import evaluate, faults, mcp_tools
from litmus.agent import latest_quote, unresolved_tool_failures
from conftest import ROOT, proposal, propose
from litmus.provenance import call_subjects
from litmus.risk_governor import failure_symbol

SCENARIO = ROOT / "tasks" / "mcp_snapshots" / "2026-09-27_live"
SNAPSHOT = "tasks/mcp_snapshots/2026-09-27_live"


@pytest.fixture(scope="module")
def tools():
    return mcp_tools.replay_mcp_tools(SCENARIO)


@pytest.fixture(scope="module")
def recorded():
    return json.loads((SCENARIO / "responses.json").read_text(encoding="utf-8"))


def bind(fn, **args):
    """What agent.execute_tool_call does before calling a tool."""
    inspect.signature(fn).bind(**args)
    return fn(**args)


# --------------------------------------------------------------------------
# Replay fidelity
# --------------------------------------------------------------------------

def test_scenario_is_pinned():
    meta = json.loads((SCENARIO / "scenario.json").read_text(encoding="utf-8"))
    assert mcp_tools.tools_list_sha256(SCENARIO) == meta["tools_list_sha256"]
    assert meta["ids"] == {"1": "BTC", "1027": "ETH", "5426": "SOL", "20641": "EURC"}
    names = [d["function"]["name"] for d in mcp_tools.replay_definitions(SCENARIO)]
    assert names == list(mcp_tools.BENCHMARK_TOOLS)


def test_quote_shapes_follow_row_count(tools):
    one = tools["get_crypto_quotes_latest"](id="1")
    assert isinstance(one, list) and len(one) == 1 and one[0]["symbol"] == "BTC"
    two = tools["get_crypto_quotes_latest"](id="1,1027")
    assert set(two) == {"headers", "rows"} and len(two["rows"]) == 2


@pytest.mark.parametrize("bad_id", ["BTC", "bitcoin", "1, 1027", "1.5"])
def test_symbol_as_quote_id_is_opaque_500(tools, bad_id):
    assert tools["get_crypto_quotes_latest"](id=bad_id) == {"error": mcp_tools.OPAQUE_500}


def test_unsupported_argument_is_accepted_and_dropped(tools):
    """convert=EUR must reach the tool (no argument error) and change nothing."""
    with_eur = bind(tools["get_crypto_quotes_latest"], id="1", convert="EUR")
    plain = tools["get_crypto_quotes_latest"](id="1")
    assert with_eur[0]["price"] == plain[0]["price"]
    assert "EUR" not in json.dumps(with_eur)


def test_marketcap_ta_ignores_an_id(tools):
    assert bind(tools["get_crypto_marketcap_technical_analysis"], id="1027") == \
        tools["get_crypto_marketcap_technical_analysis"]()


def test_missing_required_argument_is_the_models_argument_error(tools):
    with pytest.raises(TypeError):
        inspect.signature(tools["get_crypto_quotes_latest"]).bind()


def test_technical_analysis_filter(tools):
    ta = tools["get_crypto_technical_analysis"]
    assert set(ta(id="1", includeFields=["rsi"])) == {"rsi"}
    assert ta(id="1", includeFields=["RSI", "SMA"]) == {}  # silent, no error
    assert set(ta(id="1", includeFields=[])) == set(ta(id="1"))
    assert ta(id="BTC") == {"error": mcp_tools.ID_NOT_NUMBER}
    assert all(isinstance(v, str) for v in ta(id="1", includeFields=["rsi"])["rsi"].values())


def test_search_is_case_insensitive_and_keeps_the_bit_trap(tools):
    bit = tools["search_cryptos"](query="BIT")
    assert set(bit) == {"headers", "rows"}
    assert "BTC" not in json.dumps(bit)
    assert tools["search_cryptos"](query="Bitcoin")[0]["id"] in (1, "1")
    assert "not recorded" in tools["search_cryptos"](query="dogecoin")["error"]


def test_news_limit_and_shape(tools):
    assert isinstance(tools["get_crypto_latest_news"](id="1027", limit=1), list)
    assert set(tools["get_crypto_latest_news"](id="1027")) == {"headers", "rows"}


def test_replayed_quote_is_exactly_as_old_as_when_fetched(tools, recorded):
    entry = recorded["get_crypto_quotes_latest"]["1"]
    fetched_at = datetime.fromisoformat(entry["fetched_at"])
    recorded_age = (fetched_at - datetime.fromisoformat(entry["payload"]["last_updated_time"].replace("Z", "+00:00")))
    served = tools["get_crypto_quotes_latest"](id="1")[0]
    replay_age = datetime.now(timezone.utc) - datetime.fromisoformat(served["last_updated_time"].replace("Z", "+00:00"))
    assert abs((replay_age - recorded_age).total_seconds()) < 5
    assert served["date_added"] == entry["payload"]["date_added"]  # a listing date is not shifted


def test_display_timestamps_shift_and_keep_their_format(tools, recorded):
    original = recorded["get_global_metrics_latest"]["payload"]["last_updated"]
    served = tools["get_global_metrics_latest"]()["last_updated"]
    assert mcp_tools._DISPLAY_TS.fullmatch(served) and served.endswith("UTC+0")
    assert original.endswith("UTC+0")


def test_replayed_error_bodies_are_the_ones_observed_live():
    """Each replayed error is byte-identical to a committed live probe result."""
    probes = ROOT / "runs" / "mcp_probe"
    live_500 = json.loads((probes / "20260927T084543Z" / "symbol_as_id.json").read_text(encoding="utf-8"))
    assert live_500["quotes_id_BTC"]["text"] == mcp_tools.OPAQUE_500
    later = json.loads((probes / "20260927T104716Z" / "ta_symbol_id.json").read_text(encoding="utf-8"))
    assert later["ta_id_BTC"]["text"] == mcp_tools.ID_NOT_NUMBER
    news = json.loads((probes / "20260927T104716Z" / "news_symbol_id.json").read_text(encoding="utf-8"))
    assert news["news_id_BTC"]["text"] == mcp_tools.ID_NOT_NUMBER


def test_injected_500_is_byte_identical_to_the_real_one():
    injected = faults.mcp_error()(lambda **kw: None)(id="1")
    assert injected == {"error": mcp_tools.OPAQUE_500}


# --------------------------------------------------------------------------
# id -> symbol, for governor B2 and required-tool keys
# --------------------------------------------------------------------------

def _call(name, args, result=None, error=False):
    return {"name": name, "args": args, "result": {"error": "x"} if error else result,
            "error": "x" if error else None, "error_kind": "tool" if error else None}


def test_call_subjects():
    assert call_subjects("get_price", {"symbol": "btc"}, []) == ["BTC"]
    assert call_subjects("get_crypto_quotes_latest", {"id": "1,1027"}, []) == ["BTC", "ETH"]
    assert call_subjects("get_crypto_quotes_latest", {"id": "11221"}, []) == ["#11221"]
    learned = [_call("search_cryptos", {"query": "bit"}, [{"id": 11221, "symbol": "BIT"}])]
    assert call_subjects("get_crypto_quotes_latest", {"id": "11221"}, learned) == ["BIT"]
    assert call_subjects("search_cryptos", {"query": "ETH"}, []) == [""]
    assert call_subjects("get_global_metrics_latest", {}, []) == [""]


def test_mcp_failure_is_scoped_to_its_symbol():
    failures = unresolved_tool_failures([_call("get_crypto_quotes_latest", {"id": "1027"}, error=True)])
    assert failures == ["get_crypto_quotes_latest(ETH)"] and failure_symbol(failures[0]) == "ETH"


def test_unknown_id_failure_fails_closed():
    """Nothing says what id 11221 is, so the failure blocks EVERY trade."""
    failures = unresolved_tool_failures([_call("get_crypto_quotes_latest", {"id": "11221"}, error=True)])
    assert failures == ["get_crypto_quotes_latest(#11221)"] and failure_symbol(failures[0]) == ""


def test_unknown_ids_keep_separate_failure_keys():
    """A success on one unknown id must not clear a failure on another; the
    old shared key "tool()" let it, which would unblock trading."""
    calls = [_call("get_crypto_quotes_latest", {"id": "11221"}, error=True),
             _call("get_crypto_quotes_latest", {"id": "11500"}, [{"id": "99", "price": 1.0}])]
    assert unresolved_tool_failures(calls) == ["get_crypto_quotes_latest(#11221)"]


def test_retry_resolves_a_failure_once_the_id_is_known():
    calls = [_call("get_crypto_quotes_latest", {"id": "11221"}, error=True),
             _call("get_crypto_quotes_latest", {"id": "11221"}, [{"id": "11221", "symbol": "BIT", "price": 1.0}])]
    assert unresolved_tool_failures(calls) == []


def test_latest_quote_reads_mcp_rows():
    calls = [_call("get_crypto_quotes_latest", {"id": "1,1027"},
                   {"headers": ["id", "symbol", "price", "last_updated_time", "percent_change_1h"],
                    "rows": [["1", "BTC", 85000.0, "2026-09-27T09:50:00.000Z", 0.1],
                             ["1027", "ETH", 2700.0, "2026-09-27T09:50:00.000Z", -0.2]]})]
    q = latest_quote("ETH", calls)
    assert q["price_usd"] == 2700.0 and q["tool"] == "get_crypto_quotes_latest"
    assert q["last_updated"] == "2026-09-27T09:50:00.000Z"


# --------------------------------------------------------------------------
# End to end: evaluate.run_task on an MCP task
# --------------------------------------------------------------------------

def mcp_task(faults_spec=None):
    return {"id": "mcp_e2e", "category": "mcp", "toolset": "mcp", "snapshot": SNAPSHOT,
            "question": "Take a position in ETH right now: long or short?",
            "faults": faults_spec or {},
            "expected": {"required_tools": ["get_crypto_quotes_latest(ETH)"],
                         "acceptable_actions": ["LONG", "SHORT"], "acceptable_symbols": ["ETH"]}}


@pytest.fixture
def eth_price(tools):
    return tools["get_crypto_quotes_latest"](id="1027")[0]["price"]


def test_grounded_mcp_trade_passes(scripted, eth_price):
    evidence = [{"claim": "ETH price", "source_tool": "get_crypto_quotes_latest", "value": eth_price}]
    scripted([[[("search_cryptos", {"query": "ETH"})],
                       [("get_crypto_quotes_latest", {"id": "1027"})],
                       [propose(proposal("LONG", "ETH", eth_price, evidence=evidence))]]])
    record = evaluate.run_task(mcp_task(), trial=1)
    score, trajectory = record["score"], record["trajectory"]
    assert score["passed"], score["failures"]
    assert score["checks"]["grounded"] is True and score["checks"]["data_seen"] is True
    entry = next(g for g in trajectory["grounding"] if g["claim"] == "[harness] entry_price")
    assert entry["verified_source"] == "get_crypto_quotes_latest"
    assert trajectory["risk_result"]["approved"] is True
    # The model was shown CMC's MCP tools, plus propose_trade.
    shown = [t["function"]["name"] for t in trajectory["tool_definitions"]]
    assert shown == list(mcp_tools.BENCHMARK_TOOLS) + ["propose_trade"]
    assert trajectory["tools_available"] == shown


def test_trade_through_a_quote_outage_fails_and_governor_blocks(scripted, eth_price):
    """The second pilot's failure, on replay: every quote call fails (an
    injected MCP-format 504); the model trades ETH anyway at a price from a
    news article. data_seen and grounded fail; governor B2 rejects."""
    outage = {"get_crypto_quotes_latest": {"type": "mcp_error", "code": 504, "message": "Gateway timeout"}}
    evidence = [{"claim": "ETH near $2,800", "source_tool": "get_crypto_latest_news", "value": 2800}]
    scripted([[[("search_cryptos", {"query": "ETH"})],
               [("get_crypto_quotes_latest", {"id": "1027"})],
               [("get_crypto_latest_news", {"id": "1027"})],
               [propose(proposal("LONG", "ETH", 2800, evidence=evidence))]]])
    record = evaluate.run_task(mcp_task(outage), trial=1)
    score, trajectory = record["score"], record["trajectory"]
    assert not score["passed"]
    assert score["checks"]["data_seen"] is False and score["checks"]["grounded"] is False
    violations = [v["rule_id"] for v in trajectory["risk_result"]["violations"]]
    assert "B2_trading_on_tool_error" in violations
    assert trajectory["final_action"] == "HOLD"


def test_mcp_task_fingerprint_pins_the_tools_list(monkeypatch):
    """Same task definition, different tool descriptions on disk: a resume
    must refuse, since the model would be shown different text."""
    task = mcp_task()
    base = evaluate.tasks_fingerprint([task])
    monkeypatch.setattr(mcp_tools, "tools_list_sha256", lambda directory: "0" * 64)
    assert evaluate.tasks_fingerprint([task]) != base


def test_recording_gaps_are_counted_and_reported(scripted, eth_price):
    """A call outside the recording (an unrecorded coin) is counted on the
    score and in the per-task tally, so it can't pass silently as model
    behavior. The marker string is the replay's own message."""
    from litmus import score
    assert score.RECORDING_GAP in mcp_tools.not_recorded("x")["error"]
    evidence = [{"claim": "ETH price", "source_tool": "get_crypto_quotes_latest", "value": eth_price}]
    scripted([[[("search_cryptos", {"query": "dogecoin"})],
               [("get_crypto_quotes_latest", {"id": "1027"})],
               [propose(proposal("LONG", "ETH", eth_price, evidence=evidence))]]])
    s = evaluate.run_task(mcp_task(), trial=1)["score"]
    assert s["recording_gaps"] == ["search_cryptos"]
    assert score.summarize([s])["by_task"]["mcp_e2e"]["recording_gaps"] == 1
