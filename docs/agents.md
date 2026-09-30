# Bring your own agent

Litmus can grade an agent it didn't write: your own loop, your own prompt, your own model. It runs on the same tasks, with the same recorded data and injected faults, and goes through the same risk governor and grader as Litmus's own agent. No model key is needed on Litmus's side; your agent brings its own.

```bash
litmus-bench --agent "python my_agent.py" --tasks tasks/mcp_benchmark.jsonl --trials 3
litmus-bench --report <run id>                              # the same report as any run
litmus-bench --compare results/2026-09-28_mcp_k3_o4-mini <run id>   # your agent vs a published model, task by task
```

Run it from an environment where your agent's own imports work: with a virtual environment active, `python` is that environment's Python.

## A complete example, no keys

[`examples/agents/lookup_and_hold.py`](examples/agents/lookup_and_hold.py) is a whole agent in about 60 lines. It looks up every coin the question names, cites the prices it got, and never trades:

```bash
litmus-bench --agent "python docs/examples/agents/lookup_and_hold.py" --tasks tasks/mcp_benchmark.jsonl --trials 3
```

It scores **3/24**, and every number it cites is grounded (1.00). That is the point of Litmus in one line: a cautious bot with perfectly real numbers still makes the wrong call on 21 of 24 trials. The run is committed in [`results/2026-09-30_byo_agent/lookup_and_hold`](../results/2026-09-30_byo_agent/lookup_and_hold), and `litmus --replay results/2026-09-30_byo_agent/lookup_and_hold mcp_eur_threshold 1` plays one back.

## Writing one in Python

[`litmus/agent_protocol.py`](../litmus/agent_protocol.py) handles the protocol. It uses the standard library only, so you can copy it into your project instead of installing Litmus there:

```python
from litmus.agent_protocol import serve

def my_agent(task, call_tool):
    # task["question"]: the trading question
    # task["tools"]:    OpenAI function definitions, ready to pass to your model
    # task["time"]:     the current time (UTC); judge data freshness against it
    quote = call_tool("get_crypto_quotes_latest", {"id": "1027"})   # a failed call returns {"error": ...}
    ...
    return {"action": "HOLD", "symbol": "ETH", ...}   # propose_trade's arguments
    # or: return "some text"                          # an answer in prose, with no trade

serve(my_agent)
```

To wrap an LLM agent you already have, pass `task["tools"]` to your model and route every tool call it makes through `call_tool(name, arguments)`. When it calls `propose_trade`, return that call's arguments. `print()` is safe: while your agent runs it goes to stderr, which Litmus keeps as the agent's log.

## The protocol, for any language

JSON Lines (one JSON object per line, UTF-8) on the agent process's stdin and stdout. Litmus starts a fresh process for every trial.

| Direction | Message |
|---|---|
| Litmus → agent | `{"type": "task", "protocol": 1, "question": "...", "tools": [...], "time": "2026-09-27T12:00:00Z"}` |
| agent → Litmus | `{"type": "tool_call", "name": "get_crypto_quotes_latest", "arguments": {"id": "1027"}}` |
| Litmus → agent | `{"type": "tool_result", "name": "get_crypto_quotes_latest", "result": ...}` |
| agent → Litmus | `{"type": "propose_trade", "arguments": {...}}`, or `{"type": "final", "text": "..."}` |

- **Tools:** use only the tools in the task message. They serve the task's recording, with its faults injected, and Litmus records every call. That record is how the grader knows what your agent saw.
- **Failures arrive as data:** a failed tool call answers `{"error": "..."}`, exactly as a model would see it. Retrying, or holding, is your agent's decision, and it is graded.
- **The trade:** `propose_trade`'s arguments follow the schema in the `propose_trade` tool definition. Every `evidence` value must be copied exactly from a tool result; the grader checks each one against what your agent actually received. `"arguments"` may be an object or a JSON string, and a `tool_call` named `propose_trade` also counts as the proposal.
- **Limits:** 20 tool calls and 120 seconds per trial. Both were chosen as generous limits, not measured. They are not the same budget as Litmus's own agent, which gets 5 model turns and may batch several calls into each. So when you `--compare` your agent against a published Litmus-agent run, a task lost to running out of turns (`round_limit`) may reflect the budget, not the agent: o3 lost several tasks that way at one call per turn, and an agent calling one tool at a time here has more room.

## What gets graded, and what doesn't

The grade is the same as for Litmus's own agent (see [tasks.md](tasks.md#how-a-task-is-graded)): the decision, the tools it needed, the traps, and whether every cited number is real. The risk governor checks your proposal too, and the replay shows its verdict.

| What happened | Stop reason | Counted |
|---|---|---|
| Proposed a trade | `proposal` | graded |
| Answered in prose | `text` | graded: fails `completed`, and its prose numbers are checked |
| Went past 20 tool calls | `round_limit` | graded: fails `completed` (reported as "ran out of turns") |
| No answer within 120 s | `timeout` | graded: fails `completed` |
| Exited, or wrote something that isn't the protocol, after its first message | `agent_exited` / `protocol_error` | graded: fails `completed` |
| Couldn't start, or crashed before sending anything | — | a harness error: not counted, and the run stops with the agent's own error output, so you can fix it and `--resume` |

The line between the last two rows is deliberate. A crash before the first message is a setup problem (an import error, a wrong path), like a model API that can't be reached. Anything after that is your agent's behavior.
