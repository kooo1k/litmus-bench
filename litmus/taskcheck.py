"""
taskcheck.py

Checks a task file before any model sees it: `litmus-bench --check FILE`.
No model, no keys, no network. A task that fails here would otherwise fail
silently later, e.g. a misspelled tool name in required_tools can never be
satisfied, so every trial would fail for a reason that has nothing to do with
the model. See docs/tasks.md for what each field means.

What it can't check is whether the task is RIGHT: that the expected action
really follows from the recorded data. That needs the rationale, and a test
that recomputes its numbers from the replay (see tests/test_*_tasks.py).
"""

import re
from collections import Counter

from litmus import agent, evaluate, mcp_tools
from litmus.risk_governor import Action, Config

FIELDS = {"id", "category", "question", "snapshot", "faults", "expected", "rationale"}
OPTIONAL_FIELDS = {"toolset", "description_patches"}
EXPECTED_FIELDS = {"required_tools", "acceptable_actions", "acceptable_symbols"}
OPTIONAL_EXPECTED = {"traps", "position_size_pct"}
ACTIONS = {a.value for a in Action}
REQUIRED_CALL = re.compile(r"^(\w+)\(([A-Z0-9]*)\)$")  # "get_price(BTC)", "get_fear_greed()"

# Tools that take no symbol, per toolset.
REST_NO_ARGUMENT = {"get_market_overview", "get_fear_greed"}


def toolset_tools(task):
    """(tool names, tools that take no symbol) for the task's toolset."""
    if evaluate.is_mcp(task):
        return set(mcp_tools.BENCHMARK_TOOLS), set(mcp_tools.NO_ARGUMENT_TOOLS)
    return set(agent.TOOL_FUNCTIONS), REST_NO_ARGUMENT


def problems(task):
    """Everything structurally wrong with one task, as readable strings.
    An empty list means the task is well-formed and its tools build."""
    out = []
    missing = FIELDS - set(task)
    unknown = set(task) - FIELDS - OPTIONAL_FIELDS
    if missing:
        out.append(f"missing field(s): {sorted(missing)}")
    if unknown:
        out.append(f"unknown field(s): {sorted(unknown)}")
    if task.get("toolset", "rest") not in ("rest", "mcp"):
        out.append(f"toolset must be 'rest' or 'mcp', not {task['toolset']!r}")
    for field in ("id", "category", "question", "rationale", "snapshot"):
        if field in task and not (isinstance(task[field], str) and task[field].strip()):
            out.append(f"{field} must be a non-empty string")
    if out:
        return out  # the checks below assume the basic shape

    tools, no_argument = toolset_tools(task)
    expected = task["expected"]
    if not isinstance(expected, dict):
        return out + ["expected must be an object"]
    if EXPECTED_FIELDS - set(expected):
        out.append(f"expected is missing: {sorted(EXPECTED_FIELDS - set(expected))}")
    if set(expected) - EXPECTED_FIELDS - OPTIONAL_EXPECTED:
        out.append(f"expected has unknown key(s): {sorted(set(expected) - EXPECTED_FIELDS - OPTIONAL_EXPECTED)}")

    actions = expected.get("acceptable_actions") or []
    if not actions:
        out.append("acceptable_actions is empty: no answer could pass")
    if set(actions) - ACTIONS:
        out.append(f"unknown action(s) {sorted(set(actions) - ACTIONS)}; use {sorted(ACTIONS)}")
    symbols = expected.get("acceptable_symbols") or []
    if set(symbols) - Config.ALLOWED_SYMBOLS:
        out.append(f"symbol(s) {sorted(set(symbols) - Config.ALLOWED_SYMBOLS)} can't be traded "
                   f"(the risk governor allows {sorted(Config.ALLOWED_SYMBOLS)})")

    for req in expected.get("required_tools") or []:
        m = REQUIRED_CALL.match(str(req))
        if not m:
            out.append(f"required tool {req!r} must look like 'tool(SYMBOL)' or 'tool()'")
        elif m[1] not in tools:
            out.append(f"required tool {m[1]!r} is not in the {task.get('toolset', 'rest')} toolset")
        elif (m[2] == "") != (m[1] in no_argument):
            out.append(f"required tool {req!r}: {m[1]} " + ("takes no symbol" if m[1] in no_argument else "needs a symbol"))

    for trap in expected.get("traps") or []:
        if not (isinstance(trap, dict) and trap.get("tool") and trap.get("field") and trap.get("reason")):
            out.append(f"trap {trap!r} needs tool, field and reason")
        elif trap["tool"] not in tools:
            out.append(f"trap tool {trap['tool']!r} is not in the {task.get('toolset', 'rest')} toolset")

    size = expected.get("position_size_pct")
    if size is not None and not (isinstance(size, (int, float)) and 0 < size <= Config.MAX_POSITION_PCT_PER_TRADE):
        out.append(f"position_size_pct must be in (0, {Config.MAX_POSITION_PCT_PER_TRADE}]")

    patches = task.get("description_patches")
    if patches is not None:
        if not evaluate.is_mcp(task):
            out.append("description_patches only apply to MCP tasks (toolset 'mcp')")
        elif not (isinstance(patches, dict) and all(isinstance(v, str) and v.strip() for v in patches.values())):
            out.append("description_patches must be {tool: sentence}")
        elif set(patches) - tools:
            out.append(f"description_patches name tool(s) not in the mcp toolset: {sorted(set(patches) - tools)}")

    if not isinstance(task["faults"], dict):
        out.append("faults must be an object: {tool: {\"type\": ..., options}}")
    else:
        for tool, spec in task["faults"].items():
            if tool not in tools:
                out.append(f"fault on {tool!r}, which is not in the {task.get('toolset', 'rest')} toolset")
            if not isinstance(spec, dict) or spec.get("type") not in evaluate.FAULT_TYPES:
                out.append(f"fault on {tool!r} needs a type from {sorted(evaluate.FAULT_TYPES)}")
    if out:
        return out

    try:  # the snapshot exists and every fault constructs with its options
        evaluate.build_tools(task)
    except Exception as e:
        out.append(f"tools don't build: {e}")
    return out


def check_file(tasks):
    """{task id: [problems]} for a whole task file, including duplicate ids."""
    report = {}
    counts = Counter(t.get("id") for t in tasks)
    for i, task in enumerate(tasks):
        key = task.get("id") or f"(line {i + 1})"
        found = problems(task)
        if counts[task.get("id")] > 1:
            found.append("duplicate id")
        if found:
            report[key] = found
    return report
