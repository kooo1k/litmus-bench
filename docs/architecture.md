# Architecture

## Layout

```
litmus/         the harness: agent, tools, risk governor, grader, benchmark runner, CLI
scripts/       research and data tooling: MCP probes, model pilots, snapshot recording, SFT export
tests/         offline test suite (no network, no model calls)
tasks/         benchmark task files and the recorded market scenarios they replay
results/       the saved runs behind FINDINGS.md
runs/          working output: trajectories, benchmark runs, probe and pilot evidence
docs/          this file; tasks.md (writing your own tasks) and worked examples
```

## One question, end to end

```
question
   │
   ▼
agent.run_agent ──── model (llm.py: any OpenAI-compatible endpoint)
   │   ▲                 │ tool calls
   │   └── results ──────┤
   │                     ▼
   │           tools: live REST (market.py) · live MCP (mcp_tools.py)
   │                  · recorded replay (snapshots.py / mcp_tools.py) + faults.py
   │
   ▼  the model must end by calling propose_trade (a structured trade)
provenance: every cited number traced to the tool result it came from
   │
   ▼
risk_governor.check_decision ── deterministic rules: approve / resize / reject
   │
   ▼
trajectory (JSON): every message, tool call, proposal, check and verdict
   │
   ▼  benchmark only
score.score_trajectory ── grades the proposal against the task's known answer
```

The model never acts. Its proposal is untrusted input. Everything the governor treats as
fact (how old the data is, whether a cited number really came from a tool) is computed
from the actual tool results, never taken from the model's claims.

## Modules (`litmus/`)

| Module | Role |
|---|---|
| `llm.py` | Picks the model: provider presets, `LLM_*` env, flags. Builds the client with a request timeout. |
| `agent.py` | The agent loop, tool dispatch and error classification, the trajectory record. |
| `decision.py` | From an untrusted `propose_trade` call to a checked decision: schema validation, provenance of every cited number, data age. |
| `risk_governor.py` | Deterministic trade rules: stale data, trading on an unresolved tool failure, evidence, confidence, stop loss and risk/reward, position size, leverage, exposure, loss limits, volatility. |
| `market.py` | Live CoinMarketCap REST tools, shaped for the model. |
| `mcp_tools.py` | CoinMarketCap's official MCP server: live client, scenario recorder, and a replay that keeps the server's observed quirks. |
| `snapshots.py` | Recorded REST scenarios and their replay, with data ages preserved. |
| `faults.py` | Injected failures: timeouts, unavailable assets, rate limits, stale data, missing fields, contradicting sources, MCP errors. |
| `provenance.py` | Where a number came from; matches at the precision the model cited it. |
| `score.py` | The grader: checks (completed, required tools, data seen, action, symbol, size, grounded, sign, trap) for one trajectory against one task. |
| `evaluate.py` | The benchmark runner (`litmus-bench`): k trials per task, round-robin, resumable, fingerprinted. `--fail-under` gives CI an exit code. |
| `external_agent.py` | Bring your own agent (`litmus-bench --agent CMD`): runs another process as the agent, answers its tool calls with the task's tools, and builds the same trajectory `agent.py` does, so the governor and grader can't tell the two apart. Separates a setup failure (a harness error) from agent misbehavior (graded). See [agents.md](agents.md). |
| `agent_protocol.py` | The agent side of that protocol for Python agents: `serve(my_agent)`. Standard library only, so it can be copied into another project. |
| `compare.py` | Diffs two saved runs task by task (`litmus-bench --compare A B`): which tasks moved, which checks changed, 95% Wilson intervals and a Fisher exact p-value, so a k=3 difference isn't mistaken for a real one. `--fail-on-regression` gates CI, and `--tolerance N` lets B drop N passing trials first (a significance gate would never fire at k=3). |
| `live_tasks.py` | Tasks graded on today's market (`litmus --task live_eur_threshold`). Litmus fetches the data itself, sets the task so the right reading and the trap disagree by several percent, runs the agent live, and fetches again afterwards: if the market crossed the line mid-run, the grade is marked inconclusive. It refuses to build a task when the market can't separate the two readings. |
| `taskcheck.py` | Validates a task file before any model sees it (`litmus-bench --check`); see [tasks.md](tasks.md). |
| `pacing.py` | Spaces model requests for free-tier quotas and logs each one. |
| `paths.py` | Repository locations. |
| `cli.py`, `cli_view.py` | The `litmus` command and its terminal display. |
| `shell.py` | The interactive shell: `litmus` with no arguments. Questions and `/commands` with autocomplete, routed to the same code paths as the flags. |

## Benchmark runs

`litmus-bench` writes `runs/benchmark/<run_id>/`:

- `run.json`: what was asked for. It records the model, endpoint and trials, plus fingerprints of
  the task definitions (including the pinned MCP `tools/list`) and of the grading code.
- `trials.jsonl`: one line per finished trial, appended and fsynced.
- `summary.json`: rebuilt from `trials.jsonl`.

`--resume` continues a cut-off run and refuses if the tasks, grading code, model or endpoint
changed. `--report` reprints a saved run without calling a model.

## Reproducibility

CMC's MCP tools serve only the current market, and REST data moves. So every benchmark task
runs on a recorded scenario (`tasks/snapshots/`, `tasks/mcp_snapshots/`), and every model and
every trial sees identical data. Live modes (`litmus "…"`, `litmus --mcp "…"`) exist for
real use and demos; they are not graded.
