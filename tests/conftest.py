"""
Shared test setup. Two guarantees for every test in this directory:

1. No test can spend API quota. Dummy keys are set BEFORE any project
   module is imported (load_dotenv never overrides an existing variable, so
   the real keys in .env are never loaded), every outbound HTTP request
   raises, and the model client fails loudly unless a test installs a
   scripted model.
2. Nothing is written into the real runs/ or tasks/snapshots/ directories.
   Output roots are redirected to pytest's tmp_path.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

os.environ["GEMINI_API_KEY"] = "test-key-never-sent"
os.environ["CMC_API_KEY"] = "test-key-never-sent"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import requests  # noqa: E402

from litmus import agent, evaluate  # noqa: E402
from litmus.snapshots import replay_tools  # noqa: E402


class NetworkBlocked(RuntimeError):
    pass


def _unscripted(**kwargs):
    raise NetworkBlocked("a test called the model without installing a scripted model")


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    def blocked(*a, **k):
        raise NetworkBlocked("a test tried to make a real HTTP request")
    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(agent, "client", NS(chat=NS(completions=NS(create=_unscripted))))
    monkeypatch.setattr(evaluate, "OUT_ROOT", tmp_path / "benchmark")
    monkeypatch.setattr(evaluate, "AGENT_COMMAND", None)  # --agent sets it; don't leak it into the next test
    monkeypatch.setattr(evaluate, "MODEL", evaluate.MODEL)


# --------------------------------------------------------------------------
# Scripted model
# --------------------------------------------------------------------------

class QuotaError(Exception):
    """Shaped like the openai SDK's RateLimitError: status_code 429 and the
    provider's message in str()."""
    status_code = 429

    def __init__(self, message="Error code: 429 ... Please retry in 5.0s."):
        super().__init__(message)


PER_MINUTE_429 = ("Error code: 429 - Quota exceeded for metric: generate_content_free_tier_requests, "
                  "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 5.0s.")
PER_DAY_429 = ("Error code: 429 - Quota exceeded for metric: generate_content_free_tier_requests, "
               "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier.")


def _message(step):
    """A step is a prose string, or a list of (tool_name, args) calls."""
    if isinstance(step, str):
        return NS(content=step, tool_calls=None)
    calls = [NS(id=f"call_{i}", type="function",
                function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
             for i, (name, args) in enumerate(step)]
    return NS(content=None, tool_calls=calls)


class ScriptedModel:
    """Replays scripted conversations. `conversations` is a list, one entry
    per run_agent call in order: a list of steps, or an Exception instance
    to raise on that conversation's first request. `sent` records every
    messages list the model received."""

    def __init__(self, conversations):
        self.queue = list(conversations)
        self.sent = []
        self.started = 0
        self._steps = None

    def create(self, model, messages, tools):
        self.sent.append(list(messages))
        if len(messages) == 2:  # system + user: a new conversation
            self.started += 1
            conv = self.queue.pop(0)
            if isinstance(conv, Exception):
                raise conv
            self._steps = iter(conv)
        return NS(usage=None, choices=[NS(message=_message(next(self._steps)))])


@pytest.fixture
def scripted(monkeypatch):
    """scripted([conv, conv, ...]) installs a ScriptedModel and returns it."""
    def install(conversations):
        model = ScriptedModel(conversations)
        monkeypatch.setattr(agent, "client", NS(chat=NS(completions=NS(create=model.create))))
        return model
    return install


# --------------------------------------------------------------------------
# Task and proposal helpers
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tasks():
    return {t["id"]: t for t in evaluate.load_tasks()}


@pytest.fixture(scope="session")
def price(tasks):
    """price(task_id, symbol): the replayed price in that task's scenario."""
    def get(task_id, symbol):
        return replay_tools(ROOT / tasks[task_id]["snapshot"])["get_price"](symbol)["price_usd"]
    return get


def proposal(action, symbol, entry=None, size=0.03, confidence=0.75, evidence=(), stop=None,
             target=None, horizon="intraday", omit_stop=False):
    """A propose_trade argument dict. Well built by default: stop 2% on the
    losing side, target 4% on the winning side (risk/reward 2.0)."""
    p = {"action": action, "symbol": symbol, "confidence": confidence,
         "position_size_pct": size if action in ("LONG", "SHORT") else 0, "leverage": 1,
         "time_horizon": horizon, "evidence": list(evidence), "reasoning": "scripted"}
    if entry is not None:
        side = 1 if action == "LONG" else -1
        p["entry_price"] = entry
        p["stop_loss_price"] = round(entry * (1 - 0.02 * side), 2) if stop is None else stop
        p["take_profit_price"] = round(entry * (1 + 0.04 * side), 2) if target is None else target
        if omit_stop:
            del p["stop_loss_price"]
    return p


def get_price(symbol):
    return ("get_price", {"symbol": symbol})


def get_history(symbol):
    return ("get_price_history", {"symbol": symbol})


def propose(p):
    return ("propose_trade", p)
