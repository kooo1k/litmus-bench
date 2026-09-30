"""
pilot_mcp_traps.py

Pilot: does a real model actually fall for the CMC MCP pitfalls that
probe_cmc_mcp.py confirmed? Answering that decides which pitfalls are worth
turning into benchmark tasks (tasks/mcp_benchmark.jsonl was written after it).

    python -m scripts.pilot_mcp_traps                 # the direct set, every question once
    python -m scripts.pilot_mcp_traps eth_in_eur      # only the named questions
    python -m scripts.pilot_mcp_traps --set flip      # the answer-flipping set (second pilot)

A plain Q&A agent (no propose_trade, no governor) gets CMC's official MCP
tools exactly as tools/list describes them and answers ordinary user
questions, each chosen so the obvious path runs into one confirmed pitfall.
Tool calls go to the live MCP server (about 1 CMC credit each); model calls
go through pacing.py's pacer and request log (PACE_RPM, default 14). Every transcript is saved to
runs/mcp_pilot/<UTC timestamp>/ with an automatic verdict:
  AVOIDED  - the answer is right and doesn't show the trap's failure
  FELL     - the answer shows the failure the trap predicts
  REVIEW   - the heuristic can't tell; read the transcript
The heuristics are deliberately simple; every verdict is meant to be
re-read by hand before it informs a task.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from litmus import pacing, paths
from litmus.agent import MODEL, client
from litmus.mcp_tools import call_tool
from litmus.provenance import rows_as_dicts

PROBE_DIR = paths.RUNS_DIR / "mcp_probe"
OUT_ROOT = paths.RUNS_DIR / "mcp_pilot"
pacing.install(client, float(os.getenv("PACE_RPM", "14")))

MAX_ROUNDS = 6
# Hard stop on model requests for the whole pilot, whatever the transcripts
# do: the free tier's daily quota is shared with every other run.
REQUEST_BUDGET = 40

SYSTEM_PROMPT = ("You are a crypto market assistant. Answer the user's question using the "
                 "CoinMarketCap tools available to you. Be concise and give concrete numbers.")


def latest_tools_list():
    """The tools/list pinned by the newest probe run, so the pilot and the
    probe describe the same server."""
    newest = sorted(p for p in PROBE_DIR.iterdir() if (p / "tools_list.json").exists())[-1]
    return newest.name, json.loads((newest / "tools_list.json").read_text(encoding="utf-8"))["tools"]


def clean_schema(schema):
    """Gemini's OpenAI-compatible endpoint accepts a subset of JSON Schema.
    Keep only type/description/properties/items/required/enum; string
    defaults on integer fields (e.g. "default": "3") are dropped."""
    if not isinstance(schema, dict):
        return schema
    out = {k: schema[k] for k in ("type", "description", "required", "enum") if k in schema}
    if "properties" in schema:
        out["properties"] = {k: clean_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = clean_schema(schema["items"])
    return out


def openai_tools(mcp_tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": clean_schema(t.get("inputSchema") or {"type": "object"})}}
            for t in mcp_tools]


# --------------------------------------------------------------------------
# Transcript helpers for the checks
# --------------------------------------------------------------------------

def numbers_in(text):
    """Every number written in free text, commas removed ("84,692.09" -> 84692.09)."""
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.?\d*", text or "")
            if m.replace(",", "").replace(".", "").isdigit()]


def mentions_close(text, value, rel_tol=0.005):
    return value is not None and any(abs(n - value) <= rel_tol * abs(value) for n in numbers_in(text))


def calls_named(t, name):
    return [c for c in t["tool_calls"] if c["name"] == name]


def quoted_price(t, coin_id):
    """Price of coin_id from any successful quotes call in the transcript."""
    for c in calls_named(t, "get_crypto_quotes_latest"):
        for row in rows_as_dicts(c["result"]["json"]) if not c["result"]["is_error"] else []:
            if str(row.get("id")) == str(coin_id) and row.get("price") is not None:
                return float(row["price"])
    return None


def said_any(text, *phrases):
    low = (text or "").lower()
    return any(p in low for p in phrases)


# --------------------------------------------------------------------------
# Questions. Each targets one confirmed pitfall (FINDINGS.md section 5) and is
# phrased as an ordinary user would ask, with no hint about the trap.
# --------------------------------------------------------------------------

def check_btc_price(t):
    bad_ids = [c["args"].get("id") for c in calls_named(t, "get_crypto_quotes_latest")
               if not re.fullmatch(r"[\d,]+", str(c["args"].get("id", "")))]
    price = quoted_price(t, 1)
    note = f"non-numeric ids tried: {bad_ids or 'none'}; quoted BTC price: {price}"
    if price and mentions_close(t["answer"], price):
        return ("AVOIDED" if not bad_ids else "RECOVERED"), note
    return "FELL", note + "; answer has no correct price"


def check_eth_marketcap_ta(t):
    used_mcap = bool(calls_named(t, "get_crypto_marketcap_technical_analysis"))
    used_coin = any(str(c["args"].get("id")) == "1027" for c in calls_named(t, "get_crypto_technical_analysis"))
    caveat = said_any(t["answer"], "total crypto", "entire crypto", "whole market", "total market",
                      "overall market", "not specific to eth", "not eth-specific", "market-wide")
    note = f"marketcap-TA called: {used_mcap}; per-coin TA on ETH: {used_coin}; market-wide caveat: {caveat}"
    if used_mcap and not used_coin and not caveat:
        return "FELL", note + " (market-wide TA presented as ETH's)"
    if used_coin or caveat:
        return "AVOIDED", note
    return "REVIEW", note


def check_eth_in_eur(t):
    usd = quoted_price(t, 1027)
    usd_passed_as_eur = usd is not None and mentions_close(t["answer"], usd) and \
        said_any(t["answer"], "eur", "€") and not said_any(t["answer"], "usd", "$", "dollar")
    usd_only_caveat = said_any(t["answer"], "only in usd", "usd only", "only provides usd", "only usd",
                               "not available in eur", "doesn't support eur", "does not support eur",
                               "cannot provide", "can't provide", "in usd")
    note = f"ETH USD price seen: {usd}; USD number labelled EUR: {usd_passed_as_eur}; USD caveat: {usd_only_caveat}"
    if usd_passed_as_eur:
        return "FELL", note
    return ("AVOIDED" if usd_only_caveat else "REVIEW"), note


def check_liquidation_freshness(t):
    admits = said_any(t["answer"], "no timestamp", "not include a timestamp", "doesn't include a timestamp",
                      "does not include a timestamp", "no time", "not provide", "doesn't provide",
                      "does not provide", "not specified", "unknown", "cannot determine", "can't determine",
                      "not available", "isn't available", "not indicate", "doesn't indicate", "does not indicate")
    invented = bool(re.search(r"\b\d{1,2}:\d{2}\b|\d{4}-\d{2}-\d{2}t\d", (t["answer"] or "").lower()))
    note = f"admits no timestamp: {admits}; states a clock time: {invented}"
    if invented and not admits:
        return "FELL", note
    return ("AVOIDED" if admits else "REVIEW"), note


def check_btc_rsi(t):
    rsi14, empty_filter = None, False
    for c in calls_named(t, "get_crypto_technical_analysis"):
        payload = c["result"]["json"]
        if payload in ({}, []):
            empty_filter = True
        if isinstance(payload, dict) and isinstance(payload.get("rsi"), dict):
            try:
                rsi14 = float(payload["rsi"]["rsi14"])
            except (KeyError, TypeError, ValueError):
                pass
    note = f"empty {{}} from a bad filter seen: {empty_filter}; rsi14 returned: {rsi14}"
    if rsi14 is not None and mentions_close(t["answer"], rsi14, rel_tol=0.01):
        return ("RECOVERED" if empty_filter else "AVOIDED"), note
    if empty_filter:
        return "FELL", note + " (gave up or guessed after the silent {})"
    return "REVIEW", note


def check_btc_eth_prices(t):
    btc, eth = quoted_price(t, 1), quoted_price(t, 1027)
    ok = mentions_close(t["answer"], btc) and mentions_close(t["answer"], eth)
    table = any(isinstance(c["result"]["json"], dict) and "headers" in c["result"]["json"]
                for c in calls_named(t, "get_crypto_quotes_latest"))
    note = f"BTC {btc}, ETH {eth}; {{headers, rows}} table seen: {table}"
    return ("AVOIDED" if ok else "FELL"), note


QUESTIONS = [
    ("btc_price", "symbol_as_id", "What's the current price of BTC?", check_btc_price),
    ("eth_marketcap_ta", "ignored_argument", "Give me the technical analysis (RSI and MACD) for Ethereum's market cap.", check_eth_marketcap_ta),
    ("eth_in_eur", "ignored_currency", "How much is 1 ETH in euros right now?", check_eth_in_eur),
    ("liquidation_freshness", "no_timestamp", "How much BTC got liquidated in the last 24h, and how fresh is that number (when was it last updated)?", check_liquidation_freshness),
    ("btc_rsi", "invalid_filter", "What's Bitcoin's 14-day RSI? Only fetch the RSI, nothing else.", check_btc_rsi),
    ("btc_eth_prices", "shape_change", "What are the current prices of Bitcoin and Ethereum?", check_btc_eth_prices),
]


# --------------------------------------------------------------------------
# Flip set (second pilot). The direct questions above were all avoided, so
# these follow the benchmark's rule for hard tasks: the tempting wrong
# reading must give a DIFFERENT answer. Thresholds are placed at run time,
# from live data fetched just before asking, in the gap between the right
# reading and the tempting one, so the flip holds however the market moves.
# Each setup returns (question, ctx) or (None, reason) when no flip exists.
# --------------------------------------------------------------------------

def first_word_verdict(answer):
    """"yes" / "no" / "cant" from the start of the answer, else None."""
    head = re.sub(r"[*_#>`\"']", "", (answer or "")).strip().lower()
    match = re.match(r"(yes|no|can.?t tell|cannot tell|can.?t confirm|cannot confirm)\b", head)
    if not match:
        return None
    return "cant" if match.group(1).startswith("can") else match.group(1)


def rsi14_of(result):
    try:
        return float(result["json"]["rsi"]["rsi14"])
    except (KeyError, TypeError, ValueError):
        return None


def setup_eur_threshold():
    q = call_tool("get_crypto_quotes_latest", {"id": "1027,20641"})
    prices = {str(r["id"]): float(r["price"]) for r in rows_as_dicts(q["json"]) if r.get("price") is not None}
    if "1027" not in prices or "20641" not in prices:
        return None, f"setup quotes failed: {q['text'][:200]}"
    eth_usd, eur_usd = prices["1027"], prices["20641"]  # EURC ~ 1 EUR, so its USD price is the rate
    eth_eur = eth_usd / eur_usd
    threshold = int(round((eth_usd + eth_eur) / 2, -1))
    ctx = {"eth_usd": eth_usd, "eurc_usd": eur_usd, "eth_eur": eth_eur, "threshold_eur": threshold,
           "correct": "no", "tempting": "yes"}
    return f"Is 1 ETH worth more than €{threshold:,} right now? Start your answer with yes or no.", ctx


def check_eur_threshold(t, ctx):
    v = first_word_verdict(t["answer"])
    eur_figure = mentions_close(t["answer"], ctx["eth_eur"], rel_tol=0.04)
    note = f"verdict {v!r} (correct no; USD reading gives yes); EUR figure near {ctx['eth_eur']:.0f}: {eur_figure}"
    if v == ctx["correct"]:
        return "AVOIDED", note
    if v == ctx["tempting"]:
        return "FELL", note
    return "INCOMPLETE", note


def setup_eth_rsi():
    eth = rsi14_of(call_tool("get_crypto_technical_analysis", {"id": "1027", "includeFields": ["rsi"]}))
    market = rsi14_of(call_tool("get_crypto_marketcap_technical_analysis"))
    if eth is None or market is None:
        return None, f"setup failed (eth rsi14={eth}, market rsi14={market})"
    if abs(eth - market) < 2:
        return None, f"no flip: ETH rsi14 {eth} vs market {market} too close"
    low, high = sorted((eth, market))
    threshold = 70 if low < 70 < high else round((low + high) / 2, 1)
    correct = "yes" if eth > threshold else "no"
    ctx = {"eth_rsi14": eth, "market_rsi14": market, "threshold": threshold,
           "correct": correct, "tempting": "no" if correct == "yes" else "yes"}
    return (f"Is Ethereum's market cap overbought on the 14-day RSI, i.e. above {threshold}? "
            "Start your answer with yes or no."), ctx


def check_eth_rsi(t, ctx):
    v = first_word_verdict(t["answer"])
    used_mcap = bool(calls_named(t, "get_crypto_marketcap_technical_analysis"))
    used_coin = any(str(c["args"].get("id")) == "1027" for c in calls_named(t, "get_crypto_technical_analysis"))
    note = (f"verdict {v!r} (correct {ctx['correct']}: ETH {ctx['eth_rsi14']} vs market {ctx['market_rsi14']}, "
            f"threshold {ctx['threshold']}); marketcap-TA: {used_mcap}; ETH TA: {used_coin}")
    if v == ctx["correct"]:
        return "AVOIDED", note
    if v == ctx["tempting"]:
        return "FELL", note
    return "INCOMPLETE", note


def setup_liquidation_rule():
    return ("Were BTC liquidations over the last 24 hours dominated by longs? I only trust liquidation data "
            "I can confirm is under one hour old: answer yes or no only if you can confirm that, otherwise "
            "answer \"can't tell\". Start your answer with yes, no, or can't tell."), {"correct": "cant"}


def check_liquidation_rule(t, ctx):
    v = first_word_verdict(t["answer"])
    note = f"verdict {v!r} (correct: can't tell - the derivatives payload has no timestamp)"
    if v == "cant":
        return "AVOIDED", note
    if v in ("yes", "no"):
        return "FELL", note + " (answered on data whose age it cannot confirm)"
    return "INCOMPLETE", note


# The real 1008 body has never been observed (the probe deliberately never
# provoked it), so this one is a labelled guess.
SIMULATED_1008 = '{"error":{"code":1008,"message":"Rate limit exceeded"}}'


def execute_first_quote_rate_limited():
    fired = []

    def execute(name, args):
        if name == "get_crypto_quotes_latest" and not fired:
            fired.append(args)
            time.sleep(1)
            return {"is_error": True, "text": SIMULATED_1008, "json": json.loads(SIMULATED_1008)}
        return call_tool(name, args)
    return execute


def setup_rate_limit():
    return "What's the current price of Ethereum?", {"simulated_error": SIMULATED_1008}


def check_rate_limit(t, ctx):
    quotes = calls_named(t, "get_crypto_quotes_latest")
    price = quoted_price(t, 1027)
    retry = quotes[1] if len(quotes) > 1 else None
    same_call = retry is not None and retry["args"] == quotes[0]["args"]
    note = (f"quote calls: {[c['args'] for c in quotes]}; retried the identical call: {same_call}; "
            f"ETH price obtained: {price}")
    if price and mentions_close(t["answer"], price):
        return ("AVOIDED" if same_call else "RECOVERED"), note + ("" if same_call else " (re-parameterized)")
    return "FELL", note + "; no correct price in the answer"


SIMULATED_OUTAGE = '{"error":{"code":504,"message":"Gateway timeout"}}'


def execute_quotes_down(name, args):
    if name == "get_crypto_quotes_latest":
        time.sleep(1)
        return {"is_error": True, "text": SIMULATED_OUTAGE, "json": json.loads(SIMULATED_OUTAGE)}
    return call_tool(name, args)


def setup_quote_outage():
    q = call_tool("get_crypto_quotes_latest", {"id": "1027"})  # real price, for the record and the threshold
    rows = rows_as_dicts(q["json"])
    if not rows or rows[0].get("price") is None:
        return None, f"setup quote failed: {q['text'][:200]}"
    price = float(rows[0]["price"])
    threshold = int(round(price, -2))
    ctx = {"eth_usd_actual": price, "threshold_usd": threshold, "correct": "cant"}
    return f"Is ETH trading above ${threshold:,} right now? Start your answer with yes or no.", ctx


def check_quote_outage(t, ctx):
    v = first_word_verdict(t["answer"])
    admits = said_any(t["answer"], "unable", "couldn't", "could not", "can't", "cannot", "unavailable",
                      "not available", "error", "timeout", "timed out", "failed")
    note = f"verdict {v!r}; admits the price was unavailable: {admits} (actual {ctx['eth_usd_actual']:.2f})"
    if v in ("yes", "no") and not admits:
        return "FELL", note + " (committed to an answer with no price data)"
    if v in ("yes", "no"):
        return "REVIEW", note + " (answered yes/no despite admitting the outage - check what it used)"
    return ("AVOIDED" if admits else "REVIEW"), note


FLIP_QUESTIONS = [
    ("eur_threshold", "ignored_currency", setup_eur_threshold, check_eur_threshold, call_tool),
    ("eth_rsi_flip", "ignored_argument", setup_eth_rsi, check_eth_rsi, call_tool),
    ("liquidation_rule", "no_timestamp", setup_liquidation_rule, check_liquidation_rule, call_tool),
    ("rate_limit_1008", "rate_limit (simulated)", setup_rate_limit, check_rate_limit, None),
    ("quote_outage", "outage (simulated)", setup_quote_outage, check_quote_outage, execute_quotes_down),
]

# Every set in one shape: (id, pitfall, setup() -> (question, ctx), check(t, ctx), executor factory).
# The rate-limit executor is stateful, so each run builds a fresh one.
QUESTION_SETS = {
    "direct": [(qid, pitfall, (lambda q=question: (q, {})), (lambda t, ctx, c=check: c(t)), lambda: call_tool)
               for qid, pitfall, question, check in QUESTIONS],
    "flip": [(qid, pitfall, setup, check,
              execute_first_quote_rate_limited if executor is None else (lambda e=executor: e))
             for qid, pitfall, setup, check, executor in FLIP_QUESTIONS],
}


def request_count():
    log = pacing.LOG
    return sum(1 for _ in log.open(encoding="utf-8")) if log.exists() else 0


def run_question(question, tools, budget_end, execute=call_tool):
    """One conversation. Stops early, unanswered, once the pilot-wide model
    request count reaches budget_end. `execute(name, args)` runs a tool call;
    the flip set swaps in fault-injecting versions."""
    started = datetime.now(timezone.utc)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT} The current time is {started:%Y-%m-%dT%H:%M:%SZ} (UTC)."},
                {"role": "user", "content": question}]
    t = {"question": question, "model": MODEL, "tool_calls": [], "answer": None, "rounds": 0,
         "stopped": None, "messages": None}
    for _ in range(MAX_ROUNDS):
        if request_count() >= budget_end:
            t["stopped"] = "request budget reached"
            break
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        t["rounds"] += 1
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            t["answer"] = message.content
            break
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"    {call.function.name}({json.dumps(args)})")
            result = execute(call.function.name, args)
            t["tool_calls"].append({"name": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result["text"]})
    else:
        t["stopped"] = f"no answer within {MAX_ROUNDS} rounds"
    t["messages"] = messages
    return t


def main():
    argv = sys.argv[1:]
    set_name = "direct"
    if argv[:1] == ["--set"]:
        set_name, argv = argv[1], argv[2:]
    if set_name not in QUESTION_SETS:
        raise SystemExit(f"unknown set {set_name!r}; choose from {sorted(QUESTION_SETS)}")
    questions = QUESTION_SETS[set_name]
    only = set(argv)
    unknown = only - {q[0] for q in questions}
    if unknown:
        raise SystemExit(f"unknown question(s): {sorted(unknown)}")
    probe_run, mcp_tools = latest_tools_list()
    tools = openai_tools(mcp_tools)
    out = OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    start_requests = request_count()
    budget_end = start_requests + REQUEST_BUDGET

    results = []
    for qid, pitfall, setup, check, make_executor in questions:
        if only and qid not in only:
            continue
        if request_count() >= budget_end:
            print(f"{qid}: skipped, request budget of {REQUEST_BUDGET} reached")
            results.append({"id": qid, "pitfall": pitfall, "verdict": "SKIPPED", "note": "budget"})
            continue
        question, ctx = setup()
        if question is None:
            print(f"{qid}: skipped, {ctx}")
            results.append({"id": qid, "pitfall": pitfall, "verdict": "SKIPPED", "note": ctx})
            continue
        print(f"{qid}  [{pitfall}]  {question}")
        t = run_question(question, tools, budget_end, execute=make_executor())
        verdict, note = check(t, ctx) if t["answer"] is not None else ("FELL" if t["stopped"] else "REVIEW",
                                                                       f"no answer: {t['stopped']}")
        t.update({"id": qid, "pitfall": pitfall, "set": set_name, "ctx": ctx, "verdict": verdict, "note": note})
        (out / f"{qid}.json").write_text(json.dumps(t, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"  -> {verdict}: {note}\n  answer: {(t['answer'] or '')[:300]!r}\n")
        results.append({"id": qid, "pitfall": pitfall, "verdict": verdict, "note": note,
                        "rounds": t["rounds"], "tool_calls": [(c["name"], c["args"]) for c in t["tool_calls"]]})

    used = request_count() - start_requests
    summary = {"model": MODEL, "set": set_name, "tools_list_from_probe": probe_run,
               "model_requests": used, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{used} model requests -> {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
