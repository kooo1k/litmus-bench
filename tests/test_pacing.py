"""pacing.py: request spacing and the request log, without a network."""

from types import SimpleNamespace as NS

import pytest

from litmus import pacing


class FakeHTTP:
    def __init__(self):
        self.event_hooks = {"request": [], "response": []}


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(pacing, "_state", {"interval": None, "last": 0.0, "installed_on": None})
    monkeypatch.setattr(pacing, "LOG", tmp_path / "requests.jsonl")
    return NS(_client=FakeHTTP())


def test_install_is_idempotent(fresh):
    pacing.install(fresh, 14)
    pacing.install(fresh, 4)  # only the rate changes; hooks must not stack
    assert len(fresh._client.event_hooks["request"]) == 1
    assert len(fresh._client.event_hooks["response"]) == 1
    assert pacing._state["interval"] == 15.0


def test_requests_are_spaced(fresh, monkeypatch):
    pacing.install(fresh, 60)  # one per second
    now = [100.0]
    slept = []
    monkeypatch.setattr(pacing.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pacing.time, "sleep", lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    hook = fresh._client.event_hooks["request"][0]
    hook(None)
    hook(None)  # immediately after: must wait the full second
    assert slept == [pytest.approx(1.0)]


def test_every_response_is_logged(fresh):
    pacing.install(fresh, 14)
    log_hook = fresh._client.event_hooks["response"][0]
    response = NS(status_code=429, request=NS(url=NS(path="/v1beta/openai/chat/completions")))
    log_hook(response)
    log_hook(NS(status_code=200, request=response.request))
    assert pacing.request_count() == 2


def test_rate_must_be_positive(fresh):
    with pytest.raises(ValueError):
        pacing.install(fresh, 0)


def test_hooks_fire_on_the_real_sdk_client(fresh):
    """The pacer hooks the OpenAI SDK's own HTTP client, whose library has
    changed between SDK versions (httpx, then httpx2). Drive a real client
    through an in-memory transport of whichever library it uses."""
    import importlib

    from litmus import llm
    client = llm.make_client(llm.resolve(env={"GEMINI_API_KEY": "k"}))
    base = next(c for c in type(client._client).__mro__ if c.__module__.split(".")[0].startswith("httpx"))
    http_lib = importlib.import_module(base.__module__.split(".")[0])
    body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}
    client._client._transport = http_lib.MockTransport(lambda request: http_lib.Response(200, json=body))
    pacing.install(client, 600)
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert reply.choices[0].message.content == "ok"
    assert pacing.request_count() == 1
