"""
agent_protocol.py

The agent side of Litmus's bring-your-own-agent protocol (see
external_agent.py), for agents written in Python. It uses the standard
library only, so you can copy this one file into your own project instead
of installing Litmus there.

    from litmus.agent_protocol import serve

    def my_agent(task, call_tool):
        quote = call_tool("get_crypto_quotes_latest", {"id": "1027"})
        ...                                   # your loop, your model, your prompt
        return {"action": "HOLD", ...}        # propose_trade's arguments
        # or return "some text"               # an answer in prose, with no trade

    serve(my_agent)

task is Litmus's task message: task["question"], task["tools"] (OpenAI
function definitions, ready to hand to your model) and task["time"] (UTC).
call_tool(name, arguments) returns the tool's result. A failed call returns
{"error": "..."}; it doesn't raise, because what your agent does with a
failure is part of what Litmus grades.

While your agent runs, print() goes to stderr, which Litmus keeps as your
agent's log. stdout carries the protocol, and a stray print there would
break it.
"""

import json
import sys


def serve(agent, stdin=None, stdout=None):
    """Answer one Litmus task with agent(task, call_tool), then return."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    def read():
        line = stdin.readline()
        if not line:
            raise EOFError("Litmus closed the connection")
        return json.loads(line)

    def send(message):
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()

    task = read()
    if task.get("type") != "task":
        raise ValueError(f"expected a task message, got {task.get('type')!r}")

    def call_tool(name, arguments=None):
        send({"type": "tool_call", "name": name, "arguments": arguments or {}})
        return read().get("result")

    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        answer = agent(task, call_tool)
    finally:
        sys.stdout = real_stdout
    if isinstance(answer, str):
        send({"type": "final", "text": answer})
    else:
        send({"type": "propose_trade", "arguments": answer})
