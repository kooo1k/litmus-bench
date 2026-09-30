"""
Litmus: a benchmark and terminal harness for LLM trading agents on
CoinMarketCap data.

An agent answers a trading question with CMC tools (live REST, CMC's official
MCP server, or a recorded replay of either) and must end by proposing one
structured trade. A deterministic risk governor approves, resizes or rejects
it; the benchmark then grades the proposal against the task's known answer.

    llm.py            which model: any OpenAI-compatible endpoint
    agent.py          the agent loop, tool execution, the trajectory record
    decision.py       an untrusted proposal -> a checked decision (schema, provenance)
    risk_governor.py  deterministic trade rules (the model never acts directly)
    market.py         live CMC REST tools
    mcp_tools.py      CMC MCP server: live client, recorder, faithful replay
    snapshots.py      recorded REST scenarios and their replay
    faults.py         injected failures (timeouts, stale data, errors, ...)
    provenance.py     where each number came from; precision-aware matching
    score.py          the grader: one proposal against one task
    evaluate.py       the benchmark runner (`litmus-bench`): k trials, resume
    compare.py        what changed between two saved runs (`--compare`), with CIs
    external_agent.py bring your own agent: grade another process over JSON Lines (`--agent`)
    agent_protocol.py that protocol's agent side, for Python agents (stdlib only)
    taskcheck.py      validates a task file before any model sees it (`--check`)
    live_tasks.py     tasks graded on today's market, the answer set from Litmus's own fetch
    pacing.py         request pacing for free-tier model quotas
    paths.py          repository locations
    cli.py            the `litmus` command; cli_view.py is its display
    shell.py          the interactive shell (`litmus` with no arguments)

See docs/architecture.md for how the pieces fit together.
"""
