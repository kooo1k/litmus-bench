# Writing your own tasks

A task is one line of JSON: a question, the recorded market data it runs on, optional injected faults, and what a correct agent must do. Litmus replays the data, runs the agent, and grades the trade the model proposes.

The quickest way to learn the format is to copy the two worked examples in [`examples/tasks.jsonl`](examples/tasks.jsonl): one on the REST tools with an injected fault, one on CMC's MCP server with a trap.

## Anatomy of a task

Here is `mcp_eur_threshold`, the benchmark's headline task, with long strings shortened:

```json
{
  "id": "mcp_eur_threshold",
  "category": "mcp_trap",
  "toolset": "mcp",
  "question": "If 1 ETH is worth more than €2,650 right now, go long ETH; if it's worth less, short it.",
  "snapshot": "tasks/mcp_snapshots/2026-09-27_live",
  "faults": {},
  "expected": {
    "required_tools": ["get_crypto_quotes_latest(ETH)", "get_crypto_quotes_latest(EURC)"],
    "acceptable_actions": ["SHORT"],
    "acceptable_symbols": ["ETH"]
  },
  "rationale": "The quote tool is USD-only ... 1 ETH is about €2,384 < €2,650: SHORT. Reading the USD price as euros gives 2,715 > 2,650: LONG. ..."
}
```

| Field | Meaning |
|---|---|
| `id` | Unique in the file. |
| `category` | Free text, used to group results (`--category` runs one group). |
| `toolset` | `"rest"` (default: CMC REST tools such as `get_price`) or `"mcp"` (CMC's official MCP tools). |
| `question` | Exactly what the user asks. Put the decision rule in it ("if X, go long; otherwise short") so the right answer is checkable. |
| `snapshot` | The recorded scenario the tools replay: a folder under `tasks/snapshots/` (REST) or `tasks/mcp_snapshots/` (MCP). |
| `faults` | Failures to inject, per tool (see below). `{}` for none. |
| `expected.required_tools` | Calls a correct agent must make, as `tool(SYMBOL)`, or `tool()` for tools that take no symbol. |
| `expected.acceptable_actions` | Any of `LONG`, `SHORT`, `HOLD`, `CLOSE`. List several when several are defensible. |
| `expected.acceptable_symbols` | The asset(s) a LONG/SHORT may be on (`BTC`, `ETH`, `SOL`, `BNB`). |
| `expected.traps` | Optional. `[{"tool", "field", "reason"}]`: tempting numbers the agent must not rely on. |
| `expected.position_size_pct` | Optional. The exact size a LONG/SHORT must propose, when the question states one. |
| `rationale` | Why the expected answer is right, with the numbers. Reviewers and future you will need it. |

## How a task is graded

The grader (`litmus/score.py`) judges the model's **proposal**, not the risk governor's verdict. A task passes only if every check that applies passes:

| Check | Passes when |
|---|---|
| `completed` | The run ended in a valid `propose_trade`. |
| `required_tools` | Every `required_tools` call was made (an attempt that hit an injected fault still counts). |
| `data_seen` | LONG/SHORT only: every required call the model attempted for the traded asset eventually succeeded. No trading blind. |
| `action` | The action is in `acceptable_actions`. |
| `symbol` | LONG/SHORT only: the symbol is in `acceptable_symbols`. |
| `size` | Only if `position_size_pct` is set: the proposed size matches it. |
| `grounded` | Every number the model cited traces to a tool result. |
| `sign` | No cited number has its sign flipped. |
| `trap` | No number from a declared trap field is cited, in evidence or in the reasoning. The check is deterministic: citing a trap value only to dismiss it still fails. Every report counts the trials this costs (the right action, failed only on `trap`) and what the score would be without it, so the strictness is visible rather than hidden. |

## Designing a task that measures something

- **Make the tempting reading flip the answer.** Put the threshold between the correct reading of the data and the plausible wrong one. In the EUR task, €2,650 sits between the true €2,384 and the $2,715 a model gets by reading dollars as euros. If both readings led to the same trade, the task couldn't tell them apart.
- **Declare the wrong source as a trap.** If the wrong reading comes from a specific field (last week's dominance next to the current one), add it to `traps`. Traps are matched at the precision the model cites: 58.76 is caught, and the current 58.54 is not.
- **Write numbers into the rationale** and check them against the snapshot. `--check` validates structure, but only you (or a test) can confirm the answer follows from the data. The shipped tasks have tests that recompute every rationale number from the replay (`tests/test_benchmark_tasks.py`, `tests/test_mcp_benchmark_tasks.py`).
- **Add a control:** the same question on data where the other answer is right. It shows the agent reads the data rather than always giving one answer.
- **Know the baseline.** An agent that always says HOLD passes every task where HOLD is acceptable. Keep those tasks a minority, or the suite can't separate a careful agent from a lazy one.

## Faults

A fault is `{"type": ..., options}` on a tool name. Every fault also accepts:
- `times: N`: fail only the first N calls, then work (tests retrying);
- `symbols: [...]`: fail only for those assets.

| Type | Effect | Options |
|---|---|---|
| `timeout` | The request times out | `mode`: `"return"` (an error the model sees) or `"raise"` |
| `rate_limit` | HTTP 429 | pair with `times: 1` to test retrying |
| `unavailable_asset` | The asset isn't listed | |
| `stale` | Real data, but `last_updated` is old | `age_seconds` (default 86400) |
| `missing_field` | One field removed from real data | `field` |
| `contradict` | One field overridden so two tools disagree | `field`, `value` |
| `mcp_error` | CMC MCP's own error body | `code`, `message` |

Example: `"faults": {"get_price": {"type": "rate_limit", "times": 1}}` makes the first quote fail with a 429, and a retry succeeds.

## Recording your own market data

Tasks run on recorded data so that every model and every trial sees the same thing. To write tasks about a new market moment, record one (this needs `CMC_API_KEY`):

```bash
python -m litmus.mcp_tools record 2026-10-01_live      # MCP: about 27 calls (~1 credit each), BTC/ETH/SOL/EURC
python -m scripts.record_snapshots live               # REST: now -> tasks/snapshots/<today>_live/
python -m scripts.record_snapshots historical         # REST: the past scenarios listed in the script
```

A recording is never overwritten, so tasks written against it can't silently change.

## Check, replay, run

```bash
litmus-bench --check my_tasks.jsonl                    # structure, tools and faults build; no model, no keys
litmus --tasks my_tasks.jsonl --task my_first_task     # run one task on its replay and see it graded
litmus-bench --tasks my_tasks.jsonl --trials 3         # the benchmark: k=3, resumable, saved to runs/benchmark/
```

Try the checker on the examples first: `litmus-bench --check docs/examples/tasks.jsonl`.

## Live tasks: graded on today's market

A recorded task's right answer is worked out once, for the moment it was recorded. A live task's answer is set at run time: Litmus fetches the data itself, builds the question so the answer is clear-cut, runs the agent on CMC's MCP server live, and grades it.

```bash
litmus --task live_eur_threshold      # needs CMC_API_KEY and a model key
```

`live_eur_threshold` is the headline failure rebuilt for any day. Litmus fetches ETH and EURC, then sets the threshold halfway between ETH's price in euros and in dollars. The right reading (convert via EURC) and the trap (read the dollar quote as euros) land on opposite sides of it, each by several percent. Whether "worth more" or "worth less" means go long is drawn at random each run, so the right answer is LONG or SHORT and an agent can't pass by always making the same trade. Two safeguards keep the grade honest:
- **It refuses to run when the market can't separate the two readings,** e.g. with the euro near parity. A grade there would be a coin flip.
- **It fetches again after the run.** If ETH crossed the threshold while the agent worked, the grade is marked inconclusive.

Live tasks are defined in code ([`litmus/live_tasks.py`](../litmus/live_tasks.py)), not in a task file, because their question and answer are computed.
