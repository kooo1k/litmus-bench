"""
Child process for the kill/resume test: runs the real evaluate.main against
a scripted model, and dies abruptly (os._exit: no cleanup, no finally
blocks, no flushing) when conversation number KILL_AT starts. That is as
close to a SIGKILL or a closed laptop as a portable test can get.

    python tests/_child_run.py OUT_ROOT LOG_PATH KILL_AT -- <evaluate.py args>

Every conversation start is appended to LOG_PATH, so the parent can count
exactly how many trials each process began.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

os.environ["GEMINI_API_KEY"] = "test-key-never-sent"
os.environ["CMC_API_KEY"] = "test-key-never-sent"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from litmus import agent, evaluate  # noqa: E402


def main():
    out_root, log_path, kill_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluate_args = sys.argv[sys.argv.index("--") + 1:]

    def blocked(*a, **k):
        raise RuntimeError("child test process tried to make a real HTTP request")
    requests.get = blocked
    requests.Session.request = blocked
    evaluate.OUT_ROOT = Path(out_root)

    started = {"n": 0}

    def create(model, messages, tools):
        if len(messages) == 2:  # a new conversation
            started["n"] += 1
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "n": started["n"], "question": messages[1]["content"]}) + "\n")
            if started["n"] == kill_at:
                os._exit(9)
        # Every conversation: one immediate HOLD. Behavior doesn't matter
        # here, only which (task, trial) pairs ran.
        call = NS(id="c0", type="function", function=NS(name="propose_trade", arguments=json.dumps(
            {"action": "HOLD", "symbol": "BTC", "confidence": 0.2, "position_size_pct": 0, "leverage": 1,
             "time_horizon": "intraday", "evidence": [], "reasoning": "scripted"})))
        return NS(usage=None, choices=[NS(message=NS(content=None, tool_calls=[call]))])

    agent.client = NS(chat=NS(completions=NS(create=create)))
    sys.exit(evaluate.main(evaluate_args))


if __name__ == "__main__":
    main()
