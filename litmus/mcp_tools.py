"""
mcp_tools.py

CoinMarketCap's official MCP server (https://mcp.coinmarketcap.com/mcp) as
agent tools: a live client, a recorder, and a replay.

    python -m litmus.mcp_tools record 2026-09-27_live     # ~27 live calls, ~1 credit each

Why a replay at all: the MCP tools only serve the CURRENT market (no
history), so recording is the only way to show every model and every trial
the same data. A scenario directory under tasks/mcp_snapshots/ holds:
  tools_list.json   the server's tools/list, byte-pinned (its SHA-256 is in
                    scenario.json); the replay shows the model these exact
                    descriptions, since they are part of what is tested
  responses.json    every recorded payload with the moment it was fetched
  scenario.json     metadata: when, which server, id -> symbol

The replay is not a lookup table. Each tool re-implements the server's
behavior over the recorded data, including the quirks probe_cmc_mcp.py
confirmed live on 2026-09-27, because those quirks are what the benchmark
tests:
  - unsupported arguments are accepted and silently dropped (the replay
    functions take **ignored, so agent.py's argument binding lets them through)
  - a symbol passed as a quote id fails with an opaque 500 that names no cause
  - an all-invalid includeFields filter returns {} with no error
  - one row comes back as [{...}], two or more as {"headers", "rows"}
Timestamps are shifted at replay so every payload is exactly as old as it
was when fetched, the same rule snapshots.py applies to the REST scenarios.
A call the recording can't answer returns an error that says so plainly
(a recording gap), never an imitation of a real API failure.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from litmus import paths
from litmus.provenance import rows_as_dicts
from litmus.snapshots import format_ts, parse_ts

load_dotenv()
API_KEY = os.getenv("CMC_API_KEY")

MCP_URL = "https://mcp.coinmarketcap.com/mcp"
MCP_SNAPSHOT_ROOT = paths.MCP_SNAPSHOT_ROOT
REQUEST_TIMEOUT_SECONDS = 60
# All MCP tools share one per-minute budget (30/min on the free plan).
# 2.5s between recorder calls keeps well under it.
CALL_SPACING_SECONDS = 2.5
# execute_skill runs a hosted analysis for 30-300s (probed 2026-09-27), far
# past the normal per-request timeout. Only the live --all-tools mode offers it.
SLOW_TOOL_TIMEOUTS = {"execute_skill": 330}

# The tools a benchmark agent gets: everything a market decision could use.
# Left out: find_skill/execute_skill (execute is billed and takes 30-300s
# with non-deterministic output), search_crypto_info (free-text semantic
# search whose rows can exceed 85,000 characters), get_crypto_info and
# get_crypto_metrics (static profile and holder data, no decision value).
BENCHMARK_TOOLS = (
    "search_cryptos", "get_crypto_quotes_latest", "get_crypto_technical_analysis",
    "get_crypto_marketcap_technical_analysis", "get_global_metrics_latest",
    "get_global_crypto_derivatives_metrics", "get_crypto_latest_news",
    "trending_crypto_narratives", "get_upcoming_macro_events",
)
NO_ARGUMENT_TOOLS = (
    "get_crypto_marketcap_technical_analysis", "get_global_metrics_latest",
    "get_global_crypto_derivatives_metrics", "trending_crypto_narratives", "get_upcoming_macro_events",
)

# --------------------------------------------------------------------------
# Live client
# --------------------------------------------------------------------------

_rpc_id = [0]


def mcp_request(method, params=None, timeout=REQUEST_TIMEOUT_SECONDS):
    """One JSON-RPC request to the stateless streamable-HTTP endpoint.
    Returns the parsed JSON-RPC message (the server answers as one SSE event)."""
    _rpc_id[0] += 1
    response = requests.post(
        MCP_URL,
        headers={"X-CMC-MCP-API-KEY": API_KEY, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": _rpc_id[0], "method": method, "params": params or {}},
        timeout=timeout,
    )
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):])
    return json.loads(response.text)


def call_tool(name, arguments=None, spacing=CALL_SPACING_SECONDS):
    """Call one tool. Returns {"is_error", "text", "json"}: `json` is the
    parsed payload when the text is JSON, else None."""
    if spacing:
        time.sleep(spacing)
    message = mcp_request("tools/call", {"name": name, "arguments": arguments or {}},
                          timeout=SLOW_TOOL_TIMEOUTS.get(name, REQUEST_TIMEOUT_SECONDS))
    if "error" in message:  # protocol-level error, not a tool result
        return {"is_error": True, "text": json.dumps(message["error"]), "json": None}
    result = message["result"]
    text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    return {"is_error": bool(result.get("isError")), "text": text, "json": payload}


def as_agent_result(raw):
    """A call_tool result in the shape agent.py expects: the payload on
    success, {"error": <server text>} on a tool error (agent.py treats a dict
    with "error" as a tool failure)."""
    if raw["is_error"]:
        return {"error": raw["text"]}
    return raw["json"] if raw["json"] is not None else {"text": raw["text"]}


def live_mcp_tools(names=BENCHMARK_TOOLS):
    """Tools `names` against the live server, for run_agent (by default the
    benchmark's 9). Every argument is forwarded as given; the server decides
    what to drop."""
    def make(name):
        def tool(**arguments):
            return as_agent_result(call_tool(name, arguments, spacing=0))
        tool.__name__ = name
        return tool
    return {name: make(name) for name in names}


# --------------------------------------------------------------------------
# Tool definitions for the model
# --------------------------------------------------------------------------

def clean_schema(schema):
    """Gemini's OpenAI-compatible endpoint accepts a subset of JSON Schema.
    Keep type/description/properties/items/required/enum; string defaults on
    integer fields (e.g. "default": "3") are dropped."""
    if not isinstance(schema, dict):
        return schema
    out = {k: schema[k] for k in ("type", "description", "required", "enum") if k in schema}
    if "properties" in schema:
        out["properties"] = {k: clean_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = clean_schema(schema["items"])
    return out


def tool_definitions(mcp_tools, names=BENCHMARK_TOOLS):
    """OpenAI function definitions for `names` (None: every listed tool), with
    the server's own descriptions unedited. A name the server no longer lists
    is an error: silently dropping a tool would change what every task tests."""
    by_name = {t["name"]: t for t in mcp_tools}
    names = list(by_name) if names is None else names
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"tools/list no longer has: {missing}")
    return [{"type": "function", "function": {
                "name": n, "description": by_name[n].get("description", ""),
                "parameters": clean_schema(by_name[n].get("inputSchema") or {"type": "object"})}}
            for n in names]


def replay_definitions(directory):
    tools = json.loads((Path(directory) / "tools_list.json").read_text(encoding="utf-8"))["tools"]
    return tool_definitions(tools)


def patch_descriptions(definitions, patches):
    """Experiments only: `definitions` with a sentence appended to named
    tools' descriptions ({tool: sentence}), to measure whether a documentation
    fix changes agent behavior. CMC's own text is kept, unedited, before it.
    The input is not modified."""
    names = {d["function"]["name"] for d in definitions}
    unknown = sorted(set(patches) - names)
    if unknown:
        raise ValueError(f"description_patches name tools the model isn't shown: {unknown}")
    out = []
    for d in definitions:
        f = d["function"]
        extra = patches.get(f["name"])
        out.append({**d, "function": {**f, "description": f"{f['description']} {extra}"}} if extra else d)
    return out


def tools_list_sha256(directory):
    return hashlib.sha256((Path(directory) / "tools_list.json").read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Timestamps. An explicit key list, as in snapshots.py: shifting a field
# that only looked like a date would corrupt data. date_added (a coin's
# listing date) is deliberately not shifted.
# --------------------------------------------------------------------------

TIMESTAMP_KEYS = {"last_updated_time", "last_updated_time_dup", "last_updated", "publishedAt", "timestamp"}
# e.g. "27 September 2026 12:00 AM UTC+0" (global metrics).
_DISPLAY_TS = re.compile(r"(\d{1,2}) ([A-Za-z]+) (\d{4}) (\d{1,2}:\d{2} [AP]M) UTC\+0")


def _shift_value(value, offset):
    if not isinstance(value, str):
        return value
    match = _DISPLAY_TS.fullmatch(value.strip())
    if match:
        dt = datetime.strptime(" ".join(match.groups()), "%d %B %Y %I:%M %p").replace(tzinfo=timezone.utc)
        dt += offset
        return f"{dt.day} {dt.strftime('%B %Y %I:%M %p')} UTC+0"
    try:
        return format_ts(parse_ts(value) + offset)
    except ValueError:
        return value


def shift_timestamps(obj, offset):
    if isinstance(obj, dict):
        if "headers" in obj and "rows" in obj:
            cols = [i for i, h in enumerate(obj["headers"]) if h in TIMESTAMP_KEYS]
            rows = [[_shift_value(v, offset) if i in cols else v for i, v in enumerate(row)] for row in obj["rows"]]
            return {**obj, "rows": rows}
        return {k: _shift_value(v, offset) if k in TIMESTAMP_KEYS else shift_timestamps(v, offset)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [shift_timestamps(x, offset) for x in obj]
    return obj


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

DEFAULT_IDS = ("1", "1027", "5426", "20641")  # BTC, ETH, SOL, EURC (the euro stablecoin)
DEFAULT_QUERIES = ("btc", "bitcoin", "eth", "ethereum", "sol", "solana", "eur", "eurc", "bit", "bitcon")


def record_scenario(name, ids=DEFAULT_IDS, queries=DEFAULT_QUERIES, root=MCP_SNAPSHOT_ROOT):
    """Record one scenario. Refuses to overwrite an existing one (a re-record
    must be a new scenario, or the tasks written against the old data would
    silently change) and refuses to record an error (it would become a
    permanent fault in every task that uses the scenario)."""
    if not API_KEY:
        raise SystemExit("CMC_API_KEY is not set (.env)")
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    try:
        return _record_into(directory, name, ids, queries)
    except BaseException:
        shutil.rmtree(directory)  # only the directory this call just created
        raise


def _record_into(directory, name, ids, queries):
    init = mcp_request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "litmus-recorder", "version": "0"}})
    tools = mcp_request("tools/list")["result"]
    tools_bytes = json.dumps(tools, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (directory / "tools_list.json").write_bytes(tools_bytes)
    tool_definitions(tools["tools"])  # fail now if a benchmark tool disappeared

    calls = [0]

    def fetch(tool, args):
        raw = call_tool(tool, args)
        calls[0] += 1
        if raw["is_error"] or raw["json"] is None:
            raise RuntimeError(f"refusing to record a failed or non-JSON response: {tool}({args}) -> {raw['text'][:200]}")
        print(f"  recorded {tool}({json.dumps(args)})")
        return {"fetched_at": datetime.now(timezone.utc).isoformat(), "payload": raw["json"]}

    def fetch_rows(tool, args):
        entry = fetch(tool, args)
        return {**entry, "payload": rows_as_dicts(entry["payload"])}

    responses = {"search_cryptos": {}, "get_crypto_quotes_latest": {},
                 "get_crypto_technical_analysis": {}, "get_crypto_latest_news": {}}
    for q in queries:
        responses["search_cryptos"][q] = fetch_rows("search_cryptos", {"query": q})
    symbols = {}
    for coin_id in ids:
        entry = fetch_rows("get_crypto_quotes_latest", {"id": coin_id})
        if len(entry["payload"]) != 1:
            raise RuntimeError(f"expected one quote row for id {coin_id}, got {len(entry['payload'])}")
        responses["get_crypto_quotes_latest"][coin_id] = {**entry, "payload": entry["payload"][0]}
        symbols[coin_id] = str(entry["payload"][0]["symbol"]).upper()
        responses["get_crypto_technical_analysis"][coin_id] = fetch("get_crypto_technical_analysis", {"id": coin_id})
        responses["get_crypto_latest_news"][coin_id] = fetch_rows("get_crypto_latest_news", {"id": coin_id})
    for tool in NO_ARGUMENT_TOOLS:
        responses[tool] = fetch(tool, {})

    (directory / "responses.json").write_text(json.dumps(responses, indent=1, ensure_ascii=False), encoding="utf-8")
    scenario = {
        "name": name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "server": init.get("result", {}).get("serverInfo"),
        "protocol_version": init.get("result", {}).get("protocolVersion"),
        "tools_list_sha256": hashlib.sha256(tools_bytes).hexdigest(),
        "ids": symbols,
        "search_queries": list(queries),
        "tool_calls": calls[0],
    }
    (directory / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    return directory, scenario


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

# Observed live 2026-09-27 (probe_cmc_mcp.py, symbol_as_id): id="BTC".
OPAQUE_500 = '{"error":{"code":500,"message":"An internal server error occurred. "}}'
# Observed live 2026-09-27 (probe_cmc_mcp.py, ta_symbol_id / news_symbol_id):
# a non-numeric technical-analysis or news id. Plain text, not a JSON body
# (an earlier reconstruction from the docs guessed JSON and was wrong).
ID_NOT_NUMBER = "error: id is not number"
# Also observed live: a well-formed id CMC doesn't list returns [] with no
# error, for quotes and news alike. The replay can't tell such an id from a
# real coin that simply wasn't recorded, so it says "not recorded" instead of
# imitating either answer.


def not_recorded(what):
    return {"error": f"not recorded in this scenario: {what}"}


def shape_rows(rows):
    """The server's row shape: [] for none, [{...}] for one, a table for more."""
    if len(rows) <= 1:
        return rows
    headers = list(rows[0])
    return {"headers": headers, "rows": [[r.get(h) for h in headers] for r in rows]}


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def replay_mcp_tools(directory):
    """The benchmark tools served from a recorded scenario. Each function's
    signature names the server's parameters (so a missing required one is the
    model's argument error) plus **ignored (so an unsupported one is dropped,
    as the server drops it)."""
    directory = Path(directory)
    responses = json.loads((directory / "responses.json").read_text(encoding="utf-8"))

    def served(entry):
        offset = datetime.now(timezone.utc) - datetime.fromisoformat(entry["fetched_at"])
        return shift_timestamps(copy.deepcopy(entry["payload"]), offset)

    def search_cryptos(query: str, limit: int = 3, **ignored):
        entry = responses["search_cryptos"].get(str(query).strip().lower())
        if entry is None:
            return not_recorded(f"search_cryptos(query={query!r})")
        return shape_rows(served(entry)[:_clamp(limit, 1, 5, 3)])

    def get_crypto_quotes_latest(id: str, **ignored):
        text = str(id)
        if not re.fullmatch(r"\d+(,\d+)*", text):
            return {"error": OPAQUE_500}
        rows = []
        for coin_id in text.split(","):
            entry = responses["get_crypto_quotes_latest"].get(coin_id)
            if entry is None:
                return not_recorded(f"get_crypto_quotes_latest(id={coin_id!r})")
            rows.append(served(entry))
        return shape_rows(rows)

    def get_crypto_technical_analysis(id: str, includeFields: list = None, **ignored):
        if not re.fullmatch(r"\d+", str(id)):
            return {"error": ID_NOT_NUMBER}
        entry = responses["get_crypto_technical_analysis"].get(str(id))
        if entry is None:
            return not_recorded(f"get_crypto_technical_analysis(id={id!r})")
        payload = served(entry)
        if includeFields:  # omitted or [] means all blocks; invalid names drop silently
            payload = {k: v for k, v in payload.items() if k in includeFields}
        return payload

    def get_crypto_latest_news(id: str, limit: int = 10, **ignored):
        if not re.fullmatch(r"\d+", str(id)):
            return {"error": ID_NOT_NUMBER}
        entry = responses["get_crypto_latest_news"].get(str(id))
        if entry is None:
            return not_recorded(f"get_crypto_latest_news(id={id!r})")
        return shape_rows(served(entry)[:_clamp(limit, 1, 50, 10)])

    def no_argument(tool):
        def replay(**ignored):
            entry = responses.get(tool)
            return not_recorded(tool) if entry is None else served(entry)
        replay.__name__ = tool
        return replay

    tools = {"search_cryptos": search_cryptos, "get_crypto_quotes_latest": get_crypto_quotes_latest,
             "get_crypto_technical_analysis": get_crypto_technical_analysis,
             "get_crypto_latest_news": get_crypto_latest_news}
    tools.update({tool: no_argument(tool) for tool in NO_ARGUMENT_TOOLS})
    return {name: tools[name] for name in BENCHMARK_TOOLS}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "record":
        raise SystemExit("usage: python -m litmus.mcp_tools record <scenario-name>")
    out, meta = record_scenario(sys.argv[2])
    print(f"{meta['tool_calls']} tool calls -> {out}  (tools/list sha256 {meta['tools_list_sha256'][:12]}...)")
