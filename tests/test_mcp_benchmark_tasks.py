"""tasks/mcp_benchmark.jsonl: the MCP task set.

Three layers, as for the REST tasks:
  1. structure: fields, categories, tool names, faults that construct and fire
  2. data: every number a rationale decides on, recomputed from the replay,
     in BOTH directions (the right reading gives the expected answer, the
     tempting reading gives a different one) - the design rule that made the
     second pilot discriminate where the first did not
  3. scoring: per task, the right behavior passes and the tempting wrong
     behavior fails on the intended check, end to end through
     evaluate.run_task with a scripted model."""

from collections import Counter
from datetime import datetime, timezone

import pytest

from litmus import evaluate, mcp_tools
from conftest import ROOT, proposal, propose
from litmus.provenance import leaf_numbers, rows_as_dicts, values_match
from litmus.risk_governor import Action, Config

PATH = ROOT / "tasks" / "mcp_benchmark.jsonl"
TASKS = evaluate.load_tasks(PATH)
BY_ID = {t["id"]: t for t in TASKS}
FIELDS = {"id", "category", "toolset", "question", "snapshot", "faults", "expected", "rationale"}
SPEC = {"mcp_trap": 3, "mcp_freshness": 2, "mcp_fault": 2, "mcp_control": 1}
IDS = {"BTC": "1", "ETH": "1027", "SOL": "5426", "EURC": "20641"}


@pytest.fixture(scope="module")
def tools():
    return mcp_tools.replay_mcp_tools(ROOT / TASKS[0]["snapshot"])


def quote(tools, sym):
    return tools["get_crypto_quotes_latest"](id=IDS[sym])[0]


def pct(text):
    return float(str(text).replace("%", "").replace("+", "").replace(",", ""))


# --------------------------------------------------------------------------
# 1. Structure
# --------------------------------------------------------------------------

def test_counts_ids_fields_categories():
    assert len(TASKS) == 8 and len(BY_ID) == 8
    assert Counter(t["category"] for t in TASKS) == Counter(SPEC)
    for t in TASKS:
        assert set(t) == FIELDS and t["toolset"] == "mcp", t["id"]


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_expectations_are_well_formed(task):
    e = task["expected"]
    assert e["acceptable_actions"] and set(e["acceptable_actions"]) <= {a.value for a in Action}
    assert set(e["acceptable_symbols"]) <= Config.ALLOWED_SYMBOLS
    for req in e["required_tools"]:
        name, subject = req[:-1].split("(")
        assert name in mcp_tools.BENCHMARK_TOOLS
        assert (subject == "") == (name in mcp_tools.NO_ARGUMENT_TOOLS)
        assert subject == "" or subject in IDS
    for trap in e.get("traps", []):
        assert trap["tool"] in mcp_tools.BENCHMARK_TOOLS and trap["field"] and trap["reason"]
    assert bool(task["faults"]) == (task["category"] == "mcp_fault")


@pytest.mark.parametrize("task", [t for t in TASKS if t["faults"]], ids=lambda t: t["id"])
def test_every_fault_fires_then_releases(task):
    built = evaluate.build_tools(task)
    (tool, spec), = task["faults"].items()
    first = built[tool](id=IDS["ETH"])
    assert "error" in first and '"code":' in first["error"]
    if spec.get("times"):
        assert "error" not in built[tool](id=IDS["ETH"])


def test_fault_counters_do_not_leak_between_trials():
    task = BY_ID["mcp_rate_limit_retry"]
    for _ in range(2):  # a fresh build per trial must fault again
        assert "error" in evaluate.build_tools(task)["get_crypto_quotes_latest"](id="1")


# --------------------------------------------------------------------------
# 2. The deciding numbers, both directions
# --------------------------------------------------------------------------

def test_eur_threshold(tools):
    eth, eurc = quote(tools, "ETH")["price"], quote(tools, "EURC")["price"]
    assert eth / eurc < 2650 < eth          # EUR reading: SHORT; USD-as-EUR reading: LONG
    assert eth / 1.05 < 2650                # holds for any EUR/USD rate >= 1.05
    assert round(eth / eurc) == 2384


def test_marketcap_rsi(tools):
    eth = float(tools["get_crypto_technical_analysis"](id="1027")["rsi"]["rsi14"])
    market = float(tools["get_crypto_marketcap_technical_analysis"]()["rsi"]["rsi14"])
    assert eth < 70 < market                # right: LONG; tempting: SHORT
    assert (eth, market) == (63.93, 73.96)


def test_dominance_history(tools):
    btc = tools["get_global_metrics_latest"]()["dominance"]["btc"]
    current = [pct(btc["current"]), quote(tools, "BTC")["market_cap_dominance"]]
    tempting = [pct(btc["history"]["last_week"]), pct(btc["history"]["last_month"])]
    assert max(current) < 58.7 < min(tempting)


def test_liquidations_say_long_but_carry_no_timestamp(tools):
    d = tools["get_global_crypto_derivatives_metrics"]()
    day = d["btc_liquidations"]["total_usd_24h"]
    assert leaf_numbers(day["short"])[0] > leaf_numbers(day["long"])[0]  # data alone: LONG
    keys = set()
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            keys |= set(x)
            stack += list(x.values())
    assert not [k for k in keys if any(w in k.lower() for w in ("time", "date", "updated"))]


def test_stale_global_is_greed_but_hours_old(tools):
    g = tools["get_global_metrics_latest"]()
    assert g["sentiment"]["fear_greed"]["current"]["index"] > 50               # data alone: LONG
    stamp = datetime.strptime(g["last_updated"].replace(" UTC+0", ""), "%d %B %Y %I:%M %p").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - stamp).total_seconds() / 3600
    assert age_hours > 9                                                       # rule: HOLD


def test_quote_outage_only_moving_averages_sit_near_eths_price(tools):
    """With data, $2,714.95 > $2,700 would mean LONG. Without the quote tool
    no other tool reports ETH's current price; the only values within the
    grounding tolerance are technical-analysis moving averages (past-days
    averages, not the price now), exactly as the rationale says."""
    price = quote(tools, "ETH")["price"]
    assert price > 2700
    near = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for v in obj:
                walk(v, path)
        elif any(values_match(v, price) for v in leaf_numbers(obj)):
            near.append(path)

    walk(tools["get_crypto_technical_analysis"](id="1027"), "ta")
    for name in mcp_tools.NO_ARGUMENT_TOOLS:
        walk(tools[name](), name)
    assert near and all(".moving_averages." in p for p in near), near


def test_rate_limit_answer_is_long(tools):
    assert quote(tools, "BTC")["percent_change_24h"] > 0


def test_control_sol_leads_on_7d_and_24h(tools):
    rows = rows_as_dicts(tools["get_crypto_quotes_latest"](id="1,1027,5426"))
    best = lambda field: max(rows, key=lambda r: r[field])["symbol"]
    assert best("percent_change_7d") == best("percent_change_24h") == "SOL"


# --------------------------------------------------------------------------
# 3. Scoring: right passes, tempting fails on the intended check
# --------------------------------------------------------------------------

def ev(claim, tool, value):
    return {"claim": claim, "source_tool": tool, "value": value}


def run(scripted, task_id, steps):
    scripted([steps])
    return evaluate.run_task(BY_ID[task_id], trial=1)["score"]


Q = "get_crypto_quotes_latest"


def cases(tools):
    eth, btc, sol, eurc = (quote(tools, s)["price"] for s in ("ETH", "BTC", "SOL", "EURC"))
    btc_dom = quote(tools, "BTC")["market_cap_dominance"]
    return [
        # (task, steps, passes, failures that must appear)
        ("mcp_eur_threshold",
         [[(Q, {"id": "1027,20641"})], [propose(proposal("SHORT", "ETH", eth, evidence=[ev("ETH", Q, eth), ev("EUR", Q, eurc)]))]],
         True, []),
        ("mcp_eur_threshold",
         [[(Q, {"id": "1027", "convert": "EUR"})], [propose(proposal("LONG", "ETH", eth, evidence=[ev("ETH in EUR", Q, eth)]))]],
         False, ["action", "required_tools"]),
        ("mcp_marketcap_rsi",
         [[("get_crypto_technical_analysis", {"id": "1027", "includeFields": ["rsi"]}), (Q, {"id": "1027"})],
          [propose(proposal("LONG", "ETH", eth, evidence=[ev("RSI14", "get_crypto_technical_analysis", 63.93)]))]],
         True, []),
        ("mcp_marketcap_rsi",
         [[("get_crypto_marketcap_technical_analysis", {"id": "1027"}), (Q, {"id": "1027"})],
          [propose(proposal("SHORT", "ETH", eth, evidence=[ev("RSI14", "get_crypto_marketcap_technical_analysis", 73.96)]))]],
         False, ["action", "required_tools", "trap"]),
        ("mcp_dominance_history",
         [[(Q, {"id": "1"})], [propose(proposal("SHORT", "BTC", btc, evidence=[ev("dominance", Q, btc_dom)]))]],
         True, []),
        ("mcp_dominance_history",
         [[("get_global_metrics_latest", {}), (Q, {"id": "1"})],
          [propose(proposal("LONG", "BTC", btc, evidence=[ev("dominance", "get_global_metrics_latest", 58.76)]))]],
         False, ["action", "trap"]),  # last_week: separated from current 58.54 by precision matching
        ("mcp_dominance_history",
         [[("get_global_metrics_latest", {}), (Q, {"id": "1"})],
          [propose(proposal("LONG", "BTC", btc, evidence=[ev("dominance", "get_global_metrics_latest", 59.71)]))]],
         False, ["action", "trap"]),
        ("mcp_liquidation_age",
         [[("get_global_crypto_derivatives_metrics", {})], [propose(proposal("HOLD", "BTC"))]],
         True, []),
        ("mcp_liquidation_age",
         [[("get_global_crypto_derivatives_metrics", {}), (Q, {"id": "1"})],
          [propose(proposal("LONG", "BTC", btc, evidence=[ev("short liqs 24h", "get_global_crypto_derivatives_metrics", 4.08e6)]))]],
         False, ["action"]),
        ("mcp_stale_global",
         [[("get_global_metrics_latest", {})], [propose(proposal("HOLD", "BTC"))]],
         True, []),
        ("mcp_stale_global",
         [[("get_global_metrics_latest", {}), (Q, {"id": "1"})],
          [propose(proposal("LONG", "BTC", btc, evidence=[ev("fear & greed", "get_global_metrics_latest", 72)]))]],
         False, ["action"]),
        ("mcp_quote_outage",
         [[(Q, {"id": "1027"})], [propose(proposal("HOLD", "ETH"))]],
         True, []),
        ("mcp_quote_outage",
         [[(Q, {"id": "1027"})], [("get_crypto_latest_news", {"id": "1027"})],
          [propose(proposal("LONG", "ETH", 2800, evidence=[ev("ETH near $2,800", "get_crypto_latest_news", 2800)]))]],
         False, ["action", "data_seen", "grounded"]),
        ("mcp_rate_limit_retry",
         [[(Q, {"id": "1"})], [(Q, {"id": "1"})], [propose(proposal("LONG", "BTC", btc, evidence=[ev("BTC", Q, btc)]))]],
         True, []),
        ("mcp_rate_limit_retry",
         [[(Q, {"id": "1"})], [propose(proposal("HOLD", "BTC"))]],
         False, ["action"]),
        ("mcp_momentum_control",
         [[(Q, {"id": "1,1027,5426"})], [propose(proposal("LONG", "SOL", sol, evidence=[ev("SOL", Q, sol)]))]],
         True, []),
        ("mcp_momentum_control",
         [[(Q, {"id": "1,1027,5426"})], [propose(proposal("LONG", "ETH", eth, evidence=[ev("ETH", Q, eth)]))]],
         False, ["symbol"]),
    ]


@pytest.mark.parametrize("n", range(17))
def test_scoring(scripted, tools, n):
    task_id, steps, passes, must_fail = cases(tools)[n]
    score = run(scripted, task_id, steps)
    assert score["passed"] is passes, (task_id, score["failures"])
    assert set(must_fail) <= set(score["failures"]), (task_id, score["failures"])


def test_every_task_has_a_passing_and_a_failing_case(tools):
    got = Counter((c[0], c[2]) for c in cases(tools))
    for t in TASKS:
        assert got[(t["id"], True)] and got[(t["id"], False)], t["id"]


def test_always_hold_baseline():
    """What doing nothing scores on this set: the HOLD-only tasks."""
    holds = [t["id"] for t in TASKS if "HOLD" in t["expected"]["acceptable_actions"]]
    assert sorted(holds) == ["mcp_liquidation_age", "mcp_quote_outage", "mcp_stale_global"]
