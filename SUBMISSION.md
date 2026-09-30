# #BuildwithCMC submission: Litmus

Most checks verify an AI trader's numbers. Litmus verifies its decision against the known right answer. It is built on CoinMarketCap's official MCP server (and REST API): it gives any AI model CMC market data, live or from recorded runs, and makes it commit to a trade.

On CMC's own tools, it found a trade that was wrong even though every number was real and the risk layer approved it: agents misread USD-only quotes as euros. It then measured the documentation fix: **0/9 → 9/9 agents finding the euro rate**. The same method makes Litmus **a regression test for CMC's MCP server**: change a tool description, rerun the suite, and see whether agents get better or worse.

This page maps each required item to its evidence. The product itself is in [README.md](README.md); the full report for CMC is in [FINDINGS.md](FINDINGS.md).

| Required item | Where |
|---|---|
| Public repository | [Github](https://github.com/kooo1k/litmus-bench.git) |
| Working demo / screen recording | [Demo video](https://youtu.be/cA3dPGCQljQ) · [how to reproduce it](#1-demo) |
| What the agent does that a plain API call could not | [README](README.md#what-the-agent-does-that-a-plain-api-call-cant): four graded runs, each replayable with no keys |
| CMC endpoints used, named explicitly | [§2](#3-cmc-endpoints-used) |
| Visible evidence of a real API call: code and response | [§3](#4-evidence-of-real-api-calls-code-and-response) |
| What the API made possible, and where it got in the way | [§4](#5-what-the-api-made-possible-and-where-it-got-in-the-way) |
| One track | AI Agents and Automation |


## 1. Demo

To see it work in 30 seconds with **no keys**, play back a saved real run:

```bash
git clone https://github.com/kooo1k/litmus-bench.git && cd litmus-bench
pip install -e .
litmus --replay           # the headline failure: tool calls, decision, governor APPROVED, benchmark FAIL
```

To reproduce the rest of the video, in the same order:

```bash
# add CMC_API_KEY and a model key to .env (see .env.example)
litmus                    # the interactive shell and its banner
litmus › /replay          # the headline failure, replayed
litmus › /replay results/2026-09-28_eur_patch_experiment_v2.1/o4-mini mcp_eur_threshold_eurc_hint 1   # the fix: EURC fetched, converted, SHORT, PASS
litmus › /task live_eur_threshold   # live on CMC's MCP server: today's prices set the threshold, graded against today's answer
litmus › /report          # every saved run, no keys needed
```

## 2. CMC endpoints used

**MCP server:** `https://mcp.coinmarketcap.com/mcp` (JSON-RPC over HTTP, header `X-CMC-MCP-API-KEY`), in `litmus/mcp_tools.py`.

| MCP method / tool | Used for |
|---|---|
| `tools/list` | the tool descriptions shown to the model, pinned by SHA-256 so every trial sees the same text |
| `search_cryptos` | resolving a symbol (ETH) to CMC's numeric id (1027) |
| `get_crypto_quotes_latest` | current price, market cap, dominance; including EURC (id 20641), the euro rate no model found |
| `get_crypto_technical_analysis` | one coin's RSI, moving averages, MACD |
| `get_crypto_marketcap_technical_analysis` | the whole market's RSI (a trap: it silently ignores `id`) |
| `get_global_metrics_latest` | BTC dominance, Fear & Greed, with their `yesterday`/`last_week` fields |
| `get_global_crypto_derivatives_metrics` | open interest, volume, liquidations (a trap: no timestamp) |
| `get_crypto_latest_news` | news for one coin (a trap: prices quoted in articles are old) |
| `trending_crypto_narratives`, `get_upcoming_macro_events` | market context |
| `find_skill` (probed), `execute_skill` and the rest (offered with `--mcp --all-tools`) | CMC's hosted skills, live mode only |

**REST API:** `https://pro-api.coinmarketcap.com` (header `x-cmc_pro_api_key`), in `litmus/market.py` (agent tools) and `scripts/record_snapshots.py` (recording past markets).

| Endpoint | Used for |
|---|---|
| `/v1/cryptocurrency/quotes/latest` | agent tool `get_price` |
| `/v3/cryptocurrency/quotes/historical` | agent tool `get_price_history`; recording past scenarios |
| `/v2/cryptocurrency/ohlcv/historical` | agent tool `get_ohlcv`; recording past scenarios |
| `/v1/global-metrics/quotes/latest` | agent tool `get_market_overview` |
| `/v1/global-metrics/quotes/historical` | recording past scenarios |
| `/v3/fear-and-greed/latest` | agent tool `get_fear_greed` |
| `/v3/fear-and-greed/historical` | recording past scenarios |
| `/v1/key/info` | measuring the credits the probes used |

## 3. Evidence of real API calls: code and response

**The code that calls CMC's MCP server** (`litmus/mcp_tools.py`, abridged):

```python
MCP_URL = "https://mcp.coinmarketcap.com/mcp"

def mcp_request(method, params=None, timeout=REQUEST_TIMEOUT_SECONDS):
    response = requests.post(
        MCP_URL,
        headers={"X-CMC-MCP-API-KEY": API_KEY, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": _rpc_id[0], "method": method, "params": params or {}},
        timeout=timeout)
    ...

def call_tool(name, arguments=None, ...):
    message = mcp_request("tools/call", {"name": name, "arguments": arguments or {}}, ...)
```

**A real request and its response.** On 2026-09-27 at 08:45 UTC, the probe asked for Bitcoin in euros:

```json
{"method": "tools/call", "params": {"name": "get_crypto_quotes_latest", "arguments": {"id": "1", "convert": "EUR"}}}
```

CMC answered in **dollars**, with no error; the word "EUR" appears nowhere in the response:

```json
[{"id": "1", "name": "Bitcoin", "symbol": "BTC", "price": 84692.09476885547,
  "last_updated_time": "2026-09-27T08:44:00.000Z", "percent_change_24h": 0.50635225, ...}]
```

The full response is in `runs/mcp_probe/20260927T084543Z/ignored_currency.json`. That silently dropped `convert` is the root of the headline finding.

**Proof the calls were real and billed:** the same probe read CMC's own credit counter (`/v1/key/info`) before and after. Credits used went from **0 to 11** across its 12 tool calls (`find_skill` is free). See `runs/mcp_probe/20260927T084543Z/summary.json`.

**The agent's own live call.** On 2026-09-28 at 08:54 UTC, gpt-6-sol called `get_crypto_quotes_latest {"id": "1027"}` and got ETH at $2,648.44, `last_updated_time` 08:52 UTC (`results/2026-09-28_openai_cli/mcp_live_eur_question_gpt-6-sol.json`).

**The recorded benchmark scenario:** 27 live calls across all 9 tools, recorded at 2026-09-27 09:51 UTC (`tasks/mcp_snapshots/2026-09-27_live/`). Every benchmark trial replays those responses byte for byte.

## 4. What the API made possible, and where it got in the way

**Made possible:**
- **Testing agents on the exact tools CMC ships for agents:** the official MCP server, with its own tool descriptions.
- **Tool descriptions that measurably steer agents.** They avoided 6/6 documented pitfalls in a pilot, and one added sentence (naming EURC) took agents from 0/9 to 9/9 finding the euro rate.
- **Reproducible tests from real history:** historical quotes, OHLCV and Fear & Greed turned 8 past market regimes into tests.
- **Generous rate limits:** 600 requests/minute on the hackathon plan.

**Got in the way:**
- **Unsupported arguments are dropped silently** (`convert=EUR` returns USD). No model, in 12 trials across 4 models, found the euro price on its own.
- **No usable timestamps:** derivatives data has none, and global metrics carry a midnight date on intraday values, so freshness rules can't be applied.
- **Inconsistent errors and shapes:** opaque 500s, plain-text errors next to JSON ones, and a result shape that changes with the number of rows.
- **MCP tools serve only the current market,** so reproducible tests needed a recorder and a faithful replay.

Details and 8 prioritized fixes: [FINDINGS.md §5–7](FINDINGS.md#5-server-behaviors-that-set-agents-up-to-fail-checked-live).
