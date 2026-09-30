"""
pacing.py

Client-side request pacing and an exact request log for the model API.

evaluate.py on its own only reacts to a 429 after it happens, and every
429'd request costs daily quota. install() hooks the agent client's HTTP
transport (so the OpenAI SDK's own silent retries are paced and counted
too), spaces request starts at least 60/rpm seconds apart, and appends one
line per HTTP request (time, status, path) to runs/benchmark/requests.jsonl.

    litmus-bench --pace 14 --trials 3     # flash-lite: 15 RPM free tier

Free-tier limits differ by model (flash-lite 15 RPM confirmed; gemini-3.8-flash
429'd at ~5/min), so the rate is a parameter, not a constant.
"""

import json
import threading
import time
from datetime import datetime, timezone

from litmus import paths

LOG = paths.BENCHMARK_RUNS_DIR / "requests.jsonl"

_lock = threading.Lock()
_state = {"interval": None, "last": 0.0, "installed_on": None}


def _pace(request):
    with _lock:
        wait = _state["last"] + _state["interval"] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _state["last"] = time.monotonic()


def _log(response):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": response.status_code, "path": response.request.url.path}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def install(client, rpm):
    """Pace `client` (an openai.OpenAI) at `rpm` requests per minute and log
    every request. Idempotent: installing again only changes the rate, so the
    hooks can never stack (stacked hooks would double the wait and log each
    request twice)."""
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    _state["interval"] = 60.0 / rpm
    http = client._client
    if _state["installed_on"] is not http:
        http.event_hooks = {"request": [*http.event_hooks["request"], _pace],
                            "response": [*http.event_hooks["response"], _log]}
        _state["installed_on"] = http


def request_count():
    """Lines in the request log: every model API request ever paced."""
    return sum(1 for _ in LOG.open(encoding="utf-8")) if LOG.exists() else 0
