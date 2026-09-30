# Agents on CoinMarketCap's MCP server: where they fail, and what would help

*Most checks verify an AI trader's numbers. Litmus verifies its decision against the known right answer. This report is what Litmus found on CoinMarketCap's official MCP server (`mcp.coinmarketcap.com/mcp`). Measured 2026-09-27 and 2026-09-28, on four models. Every number below links to evidence in this repository.*

## Summary

- **Your tool descriptions work.** When a question pointed straight at a pitfall your MCP docs warn about, the model avoided it 6 times out of 6.
- **Agents still fail when the pitfall is one step removed.** Asked *"If 1 ETH is worth more than €2,650, go long; otherwise short"*, **no model found a euro price, in 12 of 12 trials across 4 models.** 1 ETH was about €2,384, so the right call was SHORT.
  - The small model (gemini-3.5-flash-lite) read the USD price as euros every time: 3/3 in the benchmark, again on a live replay, and again fully live on your MCP server.
  - The stronger ones (gpt-6-sol, o4-mini, o3) saw that the quote was USD-only and held: 9/9, and again live. A free open-weight model (Qwen, via OpenRouter) did the same.
  - None thought of EURC, the euro stablecoin, one quote call away.
- **One sentence in your tool description fixes the search.** With "For a EUR price, also fetch EURC (id 20641)" added to `get_crypto_quotes_latest`, agents fetched the euro rate in **9 of 9** trials, against 0 of 9 without it. Stating "USD only" alone changed nothing (section 2a).
- **But a sentence can't make an agent apply a rule it ignores.** A note that global metrics' `last_updated` is a midnight date didn't stop the small model trading on hours-old Fear & Greed data under a "last hour only" rule: 0/3 before and after. That one needs a real timestamp from the server (section 2a).
- **Every number the agent cited was real, and a deterministic risk governor approved 4 of the 5 wrong trades.** It rejected the fifth only for an unrelated risk/reward ratio, never for the premise. Grounding checks and risk rules can't see a false premise; only an evaluation that knows the right answer can.
- **Several server behaviors make these failures more likely:**
  - unsupported arguments (e.g. `convert=EUR`) are dropped silently;
  - derivatives data has no timestamp;
  - global metrics carry a midnight date stamp on intraday values;
  - errors are opaque, and the response shape changes with the row count.

  Each has a small, concrete fix (section 5).
- **Doing nothing does badly on this suite:** an always-HOLD agent scores **0/24** (at most 9/24 even when it fetches the data first). The four models scored **14/24 to 18/24** at k=3 (section 3).

---

## 1. What was tested

- **Agent:** a tool-calling LLM agent that proposes one structured trade (`propose_trade`). A deterministic risk governor then approves, resizes or rejects it.
- **Tools:** 9 of your MCP tools, with your `tools/list` descriptions unedited (SHA-256 pinned). Left out: `find_skill` / `execute_skill` (execute is billed, takes 30–300s and isn't deterministic), `search_crypto_info`, `get_crypto_info` and `get_crypto_metrics`.
- **Reproducibility:** your MCP tools serve only the current market, so we recorded one live scenario and replay it with your server's observed behavior intact (silent argument drops, error bodies, row-shape flips; see `litmus/mcp_tools.py`). Every trial sees the same data.
- **Tasks:** 8 tasks where **misreading the data changes the trade decision**. Each threshold sits between the correct reading and the tempting one (`tasks/mcp_benchmark.jsonl`). We also ran 44 tasks on your REST API across 9 historical market scenarios.
- **Models:** `gemini-3.5-flash-lite` (2026-09-27), then `gpt-6-sol` (run with `reasoning_effort: none`, which OpenAI requires for tool calls with this model on Chat Completions) `o4-mini` and `o3` (reasoning on), on 2026-09-28. k=3 trials per task each, with the agent's prompt v1 (see section 2a on the prompt). The harness accepts any OpenAI-compatible model.

## 2. The headline failure: a false currency premise, approved by the risk layer

| Run | What the agent did | Governor | Benchmark |
|---|---|---|---|
| k=3 sweep, 3 trials | "USD/EUR at parity or EUR weaker than USD" → LONG | approved ×2; rejected ×1 for risk/reward 1.37 < 1.5, not the premise | **FAIL ×3** |
| Live model, recorded data | Dropped the €: "ETH price is ~2,714.95, which is above 2,650" → LONG | approved | **FAIL** |
| Fully live on your MCP server | "$2,687.59 which is above €2,650 (assuming roughly parity or higher…)" → LONG | approved | (live mode, not graded; ≈ €2,360 at the recorded EURC rate, so SHORT was right) |
| gpt-6-sol, k=3 | "the available quote is USD-only and no EUR exchange rate was supplied" → HOLD | approved (a HOLD) | FAIL ×3 |
| o4-mini, k=3 | "only the USD price is available and no exchange rate tool exists" → HOLD | approved (a HOLD) | FAIL ×3 |
| o3, k=3 | "available data only gives USD price and no EUR/USD conversion tool exists" → HOLD | approved ×2 (a HOLD); ×1 see ¹ | FAIL ×3 |
| gpt-6-sol, fully live (2026-09-28) | "the available quote is in USD and no EUR conversion rate was provided" → HOLD | approved (a HOLD) | (live mode, not graded) |

No agent ever fetched an exchange rate, although `EURC` (id 20641) is one `get_crypto_quotes_latest` call away. The stronger models' HOLD is the safer failure: they refused to guess. But it still can't answer a question the data answers. Every cited number matched a tool result exactly, so provenance checks pass. The mistake lives in an unstated assumption.

**Why your API makes it easy:** `get_crypto_quotes_latest` is USD-only and **silently drops `convert=EUR`**. An agent that asks for euros gets dollars back, with no error.

¹ In one trial o3 proposed its HOLD with leverage 0, which a governor schema rule rejects. The outcome is a HOLD either way.

Evidence:
- `results/2026-09-27_mcp_k3_flash-lite/`, `results/2026-09-28_mcp_k3_gpt-6-sol/`, `results/2026-09-28_mcp_k3_o4-mini/` and `results/2026-09-28_mcp_k3_o3/` (`trials.jsonl`, task `mcp_eur_threshold`);
- the CLI runs in `results/2026-09-27_live_cli/`, `results/2026-09-28_openai_cli/` and `results/2026-09-28_openrouter_cli/`.

### 2a. The fix, measured

We re-ran the EUR task with one sentence appended to your own `get_crypto_quotes_latest` description, and nothing else changed: same recorded data, same server behavior (`tasks/experiments/eur_description_patch.jsonl`). gpt-6-sol, o4-mini and o3, 3 trials each:

| Sentence added | Fetched EURC | Correct trade (SHORT) |
|---|---|---|
| none (today) | 0/9 | 0/9 |
| "Prices are in USD only; a convert argument is not supported." | 0/9 | 0/9 |
| … + "For a EUR price, also fetch EURC (id 20641), the euro stablecoin." | **9/9** | **4/9** |

- **Saying "USD only" doesn't help:** the models already knew. **Saying where the euro rate lives does.**
- **Most of the 5 remaining misses were our agent's fault, not your API's.**
  - Its prompt then said *"Never state or use a number you did not receive from a tool"*. Three trials fetched EURC and then refused to calculate:
    - o3 ×2 would not do the conversion: *"Directive forbids stating or using numbers not directly provided by tools. Converting … requires division."*
    - o4-mini ×1 would not set the stop-loss the trade needs: *"cannot derive a stop_loss_price solely from tool outputs"*.
  - One o3 trial ran out of turns.
  - One gpt-6-sol trial judged EURC too indirect a proxy for the euro.
- **We've since fixed our prompt** (v2.1: calculations allowed, shown in the reasoning). With it, gpt-6-sol and o4-mini made the correct trade in **5 of 6** hinted trials.
  - Every trial fetched EURC and converted (€2,384).
  - The one miss judged EURC too indirect a stand-in for the euro.
  - With only the "USD only" note, both still held 6/6, as before.

Evidence: `results/2026-09-28_eur_patch_experiment_v1/` (prompt v1) and `results/2026-09-28_eur_patch_experiment_v2.1/` (prompt v2.1).

**A second fix, measured, and it didn't work.** We tried the same method on recommendation 2, the global-metrics timestamp. One sentence was appended to `get_global_metrics_latest`: *"Its last_updated is the day's date at 12:00 AM UTC, not the time these values were computed, so it cannot show that the data is from the last hour."* We ran the freshness task ("only use data updated within the last hour") on gemini-3.5-flash-lite, the model that fails it (`tasks/experiments/freshness_description_patch.jsonl`):

| Sentence added | Correct HOLD |
|---|---|
| none (today) | 0/3 |
| the `last_updated` note above | 0/3 |

The sentence was in the tool definition in all 3 trials, and the model still went LONG on the hours-old Fear & Greed reading without mentioning freshness. The stronger models already hold on this task without it. The lesson for your descriptions: **a sentence fixes a gap in knowledge** (where the euro rate lives) **but not a failure to apply a rule** the agent already has. For timestamps the fix has to be the server's: a real ISO update time (recommendation 2). Evidence: `results/2026-09-30_freshness_patch_experiment/`, or `litmus-bench --compare results/2026-09-28_prompt_v2.1/gemini-3.5-flash-lite results/2026-09-30_freshness_patch_experiment/gemini-3.5-flash-lite --pair mcp_stale_global=mcp_stale_global_date_note`.

## 3. Other agent failures observed

| Task (k=3) | Result | What happened |
|---|---|---|
| Fear & Greed > 50 → long, **only on data updated within the last hour** | 0/3 | Traded anyway. The payload's only timestamp is `"27 September 2026 12:00 AM UTC+0"`. One trial cited the *BTC quote's* timestamp as proof that the Fear & Greed data was fresh. |
| Price feed down (every quote call fails with your opaque 500) | 1/3 | One trial went LONG using "$2,708 as of 09:00 UTC" **from a news article**. One ran out of turns retrying. One correctly held. In an earlier pilot, the model called a news line about "the $2,800 region **last week**" the current price. |
| Five other tasks (ETH vs whole-market RSI, current vs historical dominance, untimestamped liquidations, a rate-limit retry, a table-reading control) | 3/3 each | Handled correctly. |

The table above is gemini-3.5-flash-lite. Across the four models:

| Task (k=3) | flash-lite | gpt-6-sol | o4-mini | o3 |
|---|---|---|---|---|
| Overall | 16/24 | 18/24 | 17/24 | 14/24 |
| Freshness rule (Fear & Greed) | 0/3 | 3/3 | 3/3 | 3/3 |
| Price feed down | 1/3 | 3/3 | 3/3 | 1/3 (answered in prose ×2) |
| ETH vs whole-market RSI | 3/3 | 1/3 | 2/3 | 3/3 |
| Current vs historical dominance | 3/3 | 3/3 | 2/3 | 2/3 |
| Rate-limit retry | 3/3 | 3/3 | 3/3 | 2/3 |
| Momentum control | 3/3 | 2/3 | 1/3 (ran out of turns ×2) | 0/3 (ran out of turns ×3) |

The stronger models fix the small model's freshness and outage failures, but hold more often and over-explore. o3 makes one tool call per turn, so multi-coin tasks exhaust the agent's 5 turns; twice it gave the right answer in prose instead of the required structured trade. On the RSI task, gpt-6-sol twice cited the whole-market RSI (73.96). One of those trials picked the right action, citing that value only to dismiss it, but our trap check counts any citation (section 9).

On the REST suite (44 tasks, gemini-3.5-flash-lite only), the finding is **over-exploration**. **15 of the model's 45 failures were running out of turns** while still calling tools, not wrong answers.

The suite is fault-heavy by design: when a feed is down or data is stale, the right move is to hold, so HOLD is an acceptable answer in 29 of 44 tasks. The model's 66% only matched an agent that always holds, which is why the answer-flipping MCP suite exists, where always holding scores 0/24.

## 4. What works well

- **The pitfall warnings in your tool descriptions are effective.** In a pilot of 6 direct questions, each aimed at a documented trap (a symbol passed as an id, the ignored `id` on market-cap technical analysis, the ignored `convert`, the missing derivatives timestamp, an invalid `includeFields`, and the row-shape flip), the model avoided all 6. Two answers were incomplete, but none fell for the trap. **So adding a sentence to a description is a cheap, proven fix** for what an agent doesn't know (see recommendation 8). It is not a fix for a rule the agent doesn't apply: the freshness note measured 0/3 → 0/3 (section 2a).
- **Rate limits are generous** on the hackathon plan: 600 requests/minute, and 60 unspaced `find_skill` calls never tripped a limit.
- **Historical REST endpoints made the REST suite possible:** 8 past market regimes (crash, relief rally, trends, chop) rebuilt as reproducible tests.

## 5. Server behaviors that set agents up to fail, checked live

Probed 2026-09-27 (`scripts/probe_cmc_mcp.py`; raw responses in `runs/mcp_probe/`):

| Behavior | Observed | Suggested fix |
|---|---|---|
| Unsupported arguments dropped silently (`convert="EUR"` → USD, no error) | Confirmed | Support `convert` (the REST quote endpoint already does), or return `"ignored_arguments": ["convert"]` / an error |
| `id` passed to `get_crypto_marketcap_technical_analysis` ignored; byte-identical market-wide answer | Confirmed | Reject arguments on no-argument tools |
| Derivatives payload has no timestamp of any kind | Confirmed | Add an ISO `as_of` timestamp |
| Global metrics `last_updated` is a date at midnight (`"27 September 2026 12:00 AM UTC+0"`) on intraday values | Confirmed | Use the real ISO update time |
| Symbol passed as a quote id → `{"code":500,"message":"An internal server error occurred. "}` | Confirmed | A 4xx that names the fix: "id must be numeric; resolve symbols with search_cryptos" |
| Technical analysis / news with a symbol id → plain text `error: id is not number` (not JSON) | Confirmed | One structured error format across tools |
| Unknown numeric id → `[]` with no error (quotes and news) | Confirmed | Distinguish "unknown id" from "no data" |
| Result shape: 1 row → `[{...}]`, 2+ rows → `{"headers", "rows"}` | Confirmed (our own parser broke on it) | One shape regardless of count |
| Signed display strings (`"+58.5%"`, `"84,445.89"`, `"347.75 B"`) next to `yesterday` / `last_week` fields | Confirmed | Numeric fields (keep display strings optional) |
| Derivatives 24h volume `"142.05 T"` | Looks like a unit or label error (crypto derivatives volume is usually in the hundreds of billions); **not verified** | Worth a check |
| "Overlapping datasets contradict" (warned in your docs) | Not reproduced for BTC dominance (58.48–58.62% across tools) | — |
| Rate-limit error 1008 | Not observed (600/min plan) | Document the error body |

## 6. Recommendations, in priority order

1. **Stop dropping unsupported arguments silently**, starting with `convert`. It is the root of the headline failure. Even models that reason correctly about it can only give up, since nothing tells them a euro rate exists.
2. **Add real ISO timestamps** to derivatives and global metrics. Agents can't apply a freshness rule to data whose age they can't read, and a description note is no substitute: it was measured at 0/3 → 0/3 (section 2a).
3. **Return structured, actionable errors** (one format, 4xx for caller mistakes, naming the fix).
4. **Return one row shape** regardless of row count.
5. **Distinguish unknown ids from empty results.**
6. **Return numbers as numbers** (keep display strings as an extra field).
7. **Verify the derivatives volume units.**
8. **Extend the descriptions where the agent is missing knowledge,** since there they demonstrably work (they don't make an agent apply a rule it ignores; section 2a):
   - on `get_crypto_quotes_latest`: "prices are USD only; for EUR, also fetch EURC (id 20641)". **Measured: EURC fetches went from 0/9 to 9/9, and with our fixed agent, correct trades reached 5/6** (section 2a);
   - on `get_crypto_latest_news`: "prices quoted in articles are historical, not a current price source".

## 7. What the API made possible, and where it got in the way

**Made possible:**
- testing agents on the exact tools CMC's own agent ecosystem ships;
- historical quotes, OHLCV and Fear & Greed turned past market regimes into reproducible tests;
- rich tool descriptions whose warnings measurably improved agent behavior;
- a free skill search, and generous rate limits.

**Got in the way:**
- MCP tools are current-only, so reproducible tests need a recorder and replay;
- silent argument drops, inconsistent shapes, missing timestamps and opaque errors (above);
- the OpenAI-compatible Chat Completions endpoint with built-in CMC tools is enterprise-only, so it couldn't be tested.

## 8. CMC endpoints and tools used

- **REST:**
  - `/v1/cryptocurrency/quotes/latest`
  - `/v3/cryptocurrency/quotes/historical`
  - `/v2/cryptocurrency/ohlcv/historical`
  - `/v1/global-metrics/quotes/latest`
  - `/v1/global-metrics/quotes/historical`
  - `/v3/fear-and-greed/latest`
  - `/v3/fear-and-greed/historical`
  - `/v1/key/info`
- **MCP** (`mcp.coinmarketcap.com/mcp`):
  - `search_cryptos`
  - `get_crypto_quotes_latest`
  - `get_crypto_technical_analysis`
  - `get_crypto_marketcap_technical_analysis`
  - `get_global_metrics_latest`
  - `get_global_crypto_derivatives_metrics`
  - `get_crypto_latest_news`
  - `trending_crypto_narratives`
  - `get_upcoming_macro_events`
  - `find_skill` (probed only)

## 9. Method, evidence and limits

- **Reproduce:**
  - `pip install -e .` (or use `python -m litmus` / `python -m litmus.evaluate` without installing);
  - `litmus-bench --tasks tasks/mcp_benchmark.jsonl --trials 3` (any OpenAI-compatible model: `--provider`, `--model`, `--base-url`);
  - `litmus --task mcp_eur_threshold` (one graded replay);
  - `litmus --mcp "…"` (live on your server).
- **Evidence:**
  - `results/` holds the k=3 runs and the live CLI trajectories;
  - `runs/mcp_probe/` holds the probes, with a pinned `tools/list`;
  - `runs/mcp_pilot/` holds the pilots.
- **Limits:**
  - four models on the MCP suite and one on the REST suite, k=3 each; the agent's prompt v1 throughout (section 2a);
  - 8 MCP tasks on one recorded market moment (the MCP tools can't be recorded retroactively);
  - the EUR task accepts only SHORT: a reasoned HOLD is graded FAIL, because the data does allow the answer;
  - the trap check is deterministic, so it fails any citation of a trap value, even one the model explicitly dismisses;
  - two freshness tasks grade following the user's stated rule on the evidence the tool provides (the underlying data may truly be intraday).

  These results show that the failures exist and repeat. They are not model rankings.
