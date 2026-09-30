"""
probe_cmc_mcp.py

Checks, against the live CoinMarketCap MCP server, the agent pitfalls that
CMC's own tool descriptions warn about. Each probe quotes CMC's claim, makes
the smallest call that could show it, and records CONFIRMED,
NOT_REPRODUCED or INCONCLUSIVE. Only confirmed pitfalls are candidates for
benchmark tasks.

    python -m scripts.probe_cmc_mcp                  # every probe
    python -m scripts.probe_cmc_mcp overlapping_sources   # only the named probes

Writes runs/mcp_probe/<UTC timestamp>/: the pinned tools/list (with its
SHA-256, since the server declares listChanged), every raw call result,
and summary.json. About 12 MCP tool calls, spaced under the shared
per-minute budget; credit usage is read from /v1/key/info before and after.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import requests

from litmus import paths
from litmus.mcp_tools import API_KEY, REQUEST_TIMEOUT_SECONDS, call_tool, mcp_request as mcp
from litmus.provenance import rows_as_dicts

KEY_INFO_URL = "https://pro-api.coinmarketcap.com/v1/key/info"
OUT_ROOT = paths.RUNS_DIR / "mcp_probe"


def credits_used_today():
    """Today's credit count from /v1/key/info (which itself costs no credits),
    or None if the endpoint doesn't say."""
    try:
        response = requests.get(KEY_INFO_URL, headers={"x-cmc_pro_api_key": API_KEY},
                                timeout=REQUEST_TIMEOUT_SECONDS)
        return response.json()["data"]["usage"]["current_day"]["credits_used"]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_keys(v)


def all_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from all_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_values(v)
    else:
        yield obj


def find_number(obj, *path_options):
    """First numeric-looking value under any of the given key names, searched
    recursively. Handles CMC's display strings ("63,612.69", "57.4%")."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in path_options and not isinstance(v, (dict, list)):
                try:
                    return float(str(v).replace(",", "").rstrip("%").lstrip("+"))
                except ValueError:
                    pass
            found = find_number(v, *path_options)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_number(v, *path_options)
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------
# Probes. Each returns (verdict, observation) and receives a `calls` dict it
# fills with raw results, so every verdict can be re-checked from disk.
# --------------------------------------------------------------------------

def probe_partial_symbol(calls):
    calls["search_bit"] = r = call_tool("search_cryptos", {"query": "bit"})
    calls["search_bitcoin"] = control = call_tool("search_cryptos", {"query": "bitcoin"})
    symbols = [(row.get("name"), row.get("symbol"), row.get("id")) for row in rows_as_dicts(r["json"])]
    control_ok = any(row.get("id") in (1, "1") for row in rows_as_dicts(control["json"]))
    if r["is_error"] or not control_ok:
        return "INCONCLUSIVE", f"search 'bit' -> {symbols}; control 'bitcoin' resolved to id 1: {control_ok}"
    return ("CONFIRMED" if "BTC" not in json.dumps(r["json"]) else "NOT_REPRODUCED",
            f"search 'bit' -> {symbols}; 'bitcoin' -> id 1")


def probe_symbol_as_id(calls):
    calls["quotes_id_BTC"] = r = call_tool("get_crypto_quotes_latest", {"id": "BTC"})
    names_cause = any(w in r["text"].lower() for w in ("symbol", "numeric", "must be", "not number", "invalid id"))
    if not r["is_error"]:
        return "NOT_REPRODUCED", f"id='BTC' did not error: {r['text'][:200]}"
    return ("CONFIRMED" if not names_cause else "NOT_REPRODUCED",
            f"id='BTC' error text: {r['text'][:200]!r}")


def probe_ignored_argument(calls):
    calls["mcap_ta_plain"] = a = call_tool("get_crypto_marketcap_technical_analysis")
    calls["mcap_ta_with_id"] = b = call_tool("get_crypto_marketcap_technical_analysis", {"id": "1027"})
    if a["is_error"] or b["is_error"]:
        return "INCONCLUSIVE", f"errors: plain={a['is_error']} with_id={b['is_error']} {b['text'][:200]}"
    same = a["text"] == b["text"]
    return ("CONFIRMED" if same else "INCONCLUSIVE",
            "id='1027' (ETH) accepted without error; response "
            + ("byte-identical to the no-argument call" if same else "differs (live values may have ticked)"))


def probe_ignored_currency(calls):
    calls["quotes_convert_eur"] = r = call_tool("get_crypto_quotes_latest", {"id": "1", "convert": "EUR"})
    if r["is_error"]:
        return "NOT_REPRODUCED", f"convert=EUR rejected: {r['text'][:200]}"
    mentions_eur = "EUR" in r["text"]
    return ("CONFIRMED" if not mentions_eur else "NOT_REPRODUCED",
            f"convert='EUR' accepted without error; response mentions EUR: {mentions_eur}")


def probe_overlapping_sources(calls):
    calls["quotes_btc_eth"] = q = call_tool("get_crypto_quotes_latest", {"id": "1,1027"})
    calls["global_metrics"] = g = call_tool("get_global_metrics_latest")
    if q["is_error"] or g["is_error"]:
        return "INCONCLUSIVE", "a call errored"
    btc = next((find_number(row, "market_cap_dominance") for row in rows_as_dicts(q["json"])
                if row.get("id") in (1, "1") or row.get("symbol") == "BTC"), None)
    dominance = (g["json"] or {}).get("dominance", {}) if isinstance(g["json"], dict) else {}
    glob = find_number(dominance.get("btc", {}), "current")
    if btc is None or glob is None:
        return "INCONCLUSIVE", f"could not locate both values (quote={btc}, global={glob}); see raw files"
    gap = abs(btc - glob)
    return ("CONFIRMED" if gap >= 0.1 else "NOT_REPRODUCED",
            f"BTC dominance: quotes {btc:.2f}% vs global metrics {glob:.2f}% (gap {gap:.2f} pp)")


def probe_no_timestamp(calls):
    calls["derivatives"] = r = call_tool("get_global_crypto_derivatives_metrics")
    if r["is_error"] or r["json"] is None:
        return "INCONCLUSIVE", f"error or non-JSON: {r['text'][:200]}"
    time_keys = sorted({k for k in all_keys(r["json"])
                        if any(w in k.lower() for w in ("time", "date", "updated", "as_of", "asof"))})
    return ("CONFIRMED" if not time_keys else "NOT_REPRODUCED",
            f"timestamp-like keys: {time_keys or 'none'}")


def probe_display_strings(calls):
    calls["ta_btc_rsi_ma"] = r = call_tool("get_crypto_technical_analysis",
                                           {"id": "1", "includeFields": ["rsi", "moving_averages"]})
    if r["is_error"] or r["json"] is None:
        return "INCONCLUSIVE", f"error or non-JSON: {r['text'][:200]}"
    leaves = [v for v in all_values(r["json"]) if v is not None]
    strings = [v for v in leaves if isinstance(v, str)]
    comma = [v for v in strings if "," in v]
    return ("CONFIRMED" if strings else "NOT_REPRODUCED",
            f"{len(strings)}/{len(leaves)} values are strings, e.g. {comma[:2] or strings[:2]}")


def probe_invalid_filter(calls):
    calls["ta_btc_bad_filter"] = r = call_tool("get_crypto_technical_analysis",
                                               {"id": "1", "includeFields": ["RSI", "SMA"]})
    empty = r["json"] in ({}, []) or r["text"].strip() in ("{}", "[]", "")
    return ("CONFIRMED" if (not r["is_error"] and empty) else "NOT_REPRODUCED",
            f"includeFields=['RSI','SMA']: isError={r['is_error']}, body={r['text'][:120]!r}")


def probe_find_skill(calls):
    calls["find_skill"] = r = call_tool("find_skill", {"query": "funding rate open interest regime shift", "top_k": 3})
    return ("INFO", f"isError={r['is_error']}; returned: {r['text'][:300]!r}")


# --------------------------------------------------------------------------
# Behaviors the replay (mcp_tools.py) had to reconstruct from docs or guess.
# These replace each guess with an observation.
# --------------------------------------------------------------------------

def _error_probe(calls, key, tool, args, expect_in_text):
    calls[key] = r = call_tool(tool, args)
    if not r["is_error"]:
        return "NOT_REPRODUCED", f"{tool}({args}) did not error: {r['text'][:200]!r}"
    return ("CONFIRMED" if expect_in_text in r["text"] else "NOT_REPRODUCED",
            f"{tool}({args}) error body: {r['text'][:200]!r}")


def probe_ta_symbol_id(calls):
    return _error_probe(calls, "ta_id_BTC", "get_crypto_technical_analysis", {"id": "BTC"}, "id is not number")


def probe_news_symbol_id(calls):
    return _error_probe(calls, "news_id_BTC", "get_crypto_latest_news", {"id": "BTC"}, "id is not number")


def probe_news_unknown_id(calls):
    calls["news_unknown"] = r = call_tool("get_crypto_latest_news", {"id": "999999999"})
    empty = not r["is_error"] and r["json"] in ([], {})
    return ("CONFIRMED" if empty else "NOT_REPRODUCED",
            f"unknown id: isError={r['is_error']}, body={r['text'][:200]!r}")


def probe_quote_unknown_id(calls):
    calls["quote_unknown"] = r = call_tool("get_crypto_quotes_latest", {"id": "999999999"})
    return "INFO", f"unknown id: isError={r['is_error']}, body={r['text'][:200]!r}"


def probe_multi_id_headers(calls):
    calls["quote_one"] = one = call_tool("get_crypto_quotes_latest", {"id": "1"})
    calls["quote_two"] = two = call_tool("get_crypto_quotes_latest", {"id": "1,1027"})
    single = list(rows_as_dicts(one["json"])[0]) if rows_as_dicts(one["json"]) else []
    headers = (two["json"] or {}).get("headers", []) if isinstance(two["json"], dict) else []
    return ("CONFIRMED" if headers == single else "NOT_REPRODUCED",
            f"table headers == single-row key order: {headers == single}; "
            f"only in table: {sorted(set(headers) - set(single))}; only in row: {sorted(set(single) - set(headers))}")


def probe_rate_limit(calls):
    """Burst find_skill (not billed) with no spacing until the shared
    per-minute budget trips, and record exactly what the server returns."""
    for n in range(1, 61):
        try:
            r = call_tool("find_skill", {"query": "funding rate", "top_k": 1}, spacing=0)
        except requests.HTTPError as e:
            calls["rate_limit"] = {"call": n, "http_status": e.response.status_code, "body": e.response.text[:1000]}
            return "INFO", f"HTTP {e.response.status_code} after {n} calls: {e.response.text[:200]!r}"
        if r["is_error"]:
            calls["rate_limit"] = {"call": n, **r}
            return ("CONFIRMED" if "1008" in r["text"] else "INFO",
                    f"tool error after {n} calls: {r['text'][:200]!r}")
    return "NOT_REPRODUCED", "60 unspaced calls, no rate-limit error"


PROBES = [
    ("partial_symbol", "search_cryptos: \"bit\" returns BitDAO and Biconomy, not Bitcoin", probe_partial_symbol),
    ("symbol_as_id", "get_crypto_quotes_latest: symbols ... fail with an opaque server error that names no cause", probe_symbol_as_id),
    ("ignored_argument", "get_crypto_marketcap_technical_analysis: an id ... is silently ignored and the response is identical", probe_ignored_argument),
    ("ignored_currency", "Every tool accepts and silently discards unsupported arguments ... USD only", probe_ignored_currency),
    ("overlapping_sources", "overlapping datasets agree on some fields and contradict on others", probe_overlapping_sources),
    ("no_timestamp", "get_global_crypto_derivatives_metrics: the payload carries no timestamp of any kind", probe_no_timestamp),
    ("display_strings", "get_crypto_technical_analysis: Every value is a display string, not a number", probe_display_strings),
    ("invalid_filter", "includeFields: an all-invalid filter returns {} with isError false", probe_invalid_filter),
    ("find_skill", "find_skill is not billed - no credits", probe_find_skill),
    ("ta_symbol_id", "get_crypto_technical_analysis: a symbol ... fails with \"id is not number\"", probe_ta_symbol_id),
    ("news_symbol_id", "get_crypto_latest_news: A symbol fails with \"id is not number\"", probe_news_symbol_id),
    ("news_unknown_id", "get_crypto_latest_news: a well-formed but unknown id returns an empty list", probe_news_unknown_id),
    ("quote_unknown_id", "(undocumented) what an unknown quote id returns", probe_quote_unknown_id),
    ("multi_id_headers", "(undocumented) a multi-id table's columns vs a single row's keys", probe_multi_id_headers),
    ("rate_limit", "All tools share one per-minute budget; exceeding it returns error 1008", probe_rate_limit),
]


def main():
    if not API_KEY:
        raise SystemExit("CMC_API_KEY is not set (.env)")
    out = OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)

    credits_before = credits_used_today()
    init = mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "litmus-probe", "version": "0"}})
    tools = mcp("tools/list")["result"]
    tools_bytes = json.dumps(tools, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (out / "tools_list.json").write_bytes(tools_bytes)
    (out / "server_info.json").write_text(json.dumps(init.get("result", {}), indent=2), encoding="utf-8")

    only = set(sys.argv[1:])
    unknown = only - {p[0] for p in PROBES}
    if unknown:
        raise SystemExit(f"unknown probe(s): {sorted(unknown)}")
    results, tool_calls = [], 0
    for probe_id, claim, fn in PROBES:
        if only and probe_id not in only:
            continue
        calls = {}
        try:
            verdict, observation = fn(calls)
        except Exception as e:  # a probe failing must not lose the others' evidence
            verdict, observation = "INCONCLUSIVE", f"probe crashed: {type(e).__name__}: {e}"
        tool_calls += len(calls)
        (out / f"{probe_id}.json").write_text(json.dumps(calls, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"probe": probe_id, "cmc_claim": claim, "verdict": verdict, "observation": observation})
        print(f"{verdict:15} {probe_id:20} {observation}")

    time.sleep(5)  # give usage accounting a moment before reading it back
    credits_after = credits_used_today()
    summary = {
        "probed_at": out.name,
        "server": init.get("result", {}).get("serverInfo"),
        "protocol_version": init.get("result", {}).get("protocolVersion"),
        "tools": [t["name"] for t in tools["tools"]],
        "tools_list_sha256": hashlib.sha256(tools_bytes).hexdigest(),
        "tool_calls": tool_calls,
        "credits_used_today_before": credits_before,
        "credits_used_today_after": credits_after,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    delta = (credits_after - credits_before) if None not in (credits_before, credits_after) else "unknown"
    print(f"\n{tool_calls} tool calls, credits used by this probe: {delta} "
          f"(key/info may lag) -> {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
