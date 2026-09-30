"""llm.py: model/provider resolution, and evaluate.py pinning a run to its endpoint.

The harness is model-agnostic (any OpenAI-compatible endpoint with tool
calling); only Gemini has been exercised live, so these offline tests are
what stand behind the other presets."""

import json

import pytest

from litmus import agent, evaluate, llm
from conftest import PER_DAY_429, QuotaError

OWNER_ENV = {  # a typical Gemini .env from before llm.py existed (values are dummies)
    "GEMINI_API_KEY": "g-key", "GEMINI_MODEL": "gemini-3.5-flash-lite",
    "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/",
}


def test_owner_env_resolves_to_the_previous_setup():
    c = llm.resolve(env=OWNER_ENV)
    assert (c.provider, c.model, c.base_url, c.api_key) == (
        "gemini", "gemini-3.5-flash-lite", llm.GEMINI_URL, "g-key")
    assert "native API" in c.notice  # the .../v1beta/ root was corrected, and it says so


def test_nothing_set_but_a_gemini_key_is_gemini():
    c = llm.resolve(env={"GEMINI_API_KEY": "k"})
    assert (c.provider, c.base_url, c.model) == ("gemini", llm.GEMINI_URL, "gemini-2.5-flash")


def test_flags_beat_env_which_beats_legacy():
    env = {**OWNER_ENV, "LLM_MODEL": "env-model", "OPENAI_API_KEY": "o-key"}
    assert llm.resolve(env=env).model == "env-model"
    c = llm.resolve(provider="openai", model="gpt-x", env=env)
    assert (c.provider, c.model, c.base_url, c.api_key) == ("openai", "gpt-x", "https://api.openai.com/v1", "o-key")
    c = llm.resolve(provider="openai", model="gpt-x", base_url="http://my-proxy/v1", env=env)
    assert c.base_url == "http://my-proxy/v1" and c.provider == "openai"


@pytest.mark.parametrize("provider, key_var", [
    ("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY")])
def test_each_preset_reads_its_own_key(provider, key_var):
    c = llm.resolve(provider=provider, model="m", env={key_var: "secret"})
    assert c.base_url == llm.PROVIDERS[provider][0] and c.api_key == "secret" and c.key_var == key_var


def test_legacy_gemini_model_never_leaks_into_another_provider():
    with pytest.raises(llm.ConfigError, match="--model"):
        llm.resolve(provider="openai", env={**OWNER_ENV, "OPENAI_API_KEY": "o"})


def test_local_server_needs_no_key():
    c = llm.resolve(provider="ollama", model="qwen3", env={})
    assert c.has_key and c.base_url == "http://localhost:11434/v1"


def test_custom_endpoint_uses_llm_api_key():
    c = llm.resolve(base_url="http://vllm.local:8000/v1", model="m", env={"LLM_API_KEY": "x"})
    assert (c.provider, c.api_key, c.key_var) == ("custom", "x", "LLM_API_KEY")


def test_missing_key_is_reported_by_name():
    c = llm.resolve(provider="openai", model="m", env={})
    assert not c.has_key and c.key_var == "OPENAI_API_KEY"


def test_unknown_provider_is_an_error():
    with pytest.raises(llm.ConfigError, match="unknown provider"):
        llm.resolve(provider="nope", env={})


def test_only_the_gemini_native_root_is_rewritten():
    assert llm.resolve(base_url="https://example.com/v1beta/", model="m", env={}).base_url == "https://example.com/v1beta/"


# --------------------------------------------------------------------------
# evaluate.py: flags, and a run pinned to its endpoint
# --------------------------------------------------------------------------

@pytest.fixture
def restore_agent(monkeypatch):
    """configure() rebinds agent globals; put them back after the test."""
    for name in ("LLM", "MODEL", "client"):
        monkeypatch.setattr(agent, name, getattr(agent, name))
    monkeypatch.setattr(evaluate, "MODEL", evaluate.MODEL)


def test_run_records_its_endpoint_and_resume_refuses_another(scripted, restore_agent, monkeypatch):
    scripted([QuotaError(PER_DAY_429)])
    evaluate.main(["--ids", "clean_01"])
    run_dir = next((evaluate.OUT_ROOT).iterdir())
    config = json.loads((run_dir / evaluate.RUN_FILE).read_text(encoding="utf-8"))
    assert config["base_url"] == agent.LLM.base_url and config["provider"] == agent.LLM.provider
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    other = llm.resolve(provider="openrouter", model=evaluate.MODEL)  # same model name, other provider
    monkeypatch.setattr(agent, "LLM", other)
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", run_dir.name])


def test_unknown_provider_flag_is_a_clean_cli_error(restore_agent):
    with pytest.raises(SystemExit):
        evaluate.main(["--provider", "nope", "--ids", "clean_01"])


def test_missing_key_stops_before_any_model_call(restore_agent, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        evaluate.main(["--provider", "openai", "--model", "gpt-x", "--ids", "clean_01"])


def test_client_has_a_request_timeout():
    """A request the provider never answers must fail, not hang: the SDK's
    default of 600s per attempt froze the first live CLI run."""
    client = llm.make_client(llm.resolve(env={"GEMINI_API_KEY": "k"}))
    assert client.timeout == llm.REQUEST_TIMEOUT_SECONDS == 60
    assert client.max_retries == llm.MAX_RETRIES


# --------------------------------------------------------------------------
# reasoning_effort: sent only when set
# --------------------------------------------------------------------------

def test_reasoning_effort_from_flag_or_env_and_sent_only_when_set():
    assert llm.resolve(env={"GEMINI_API_KEY": "k"}).reasoning_effort == ""
    assert llm.request_options(llm.resolve(env={"GEMINI_API_KEY": "k"})) == {}
    c = llm.resolve(provider="openai", model="m", reasoning_effort="none", env={"OPENAI_API_KEY": "k"})
    assert llm.request_options(c) == {"reasoning_effort": "none"}
    env = {"OPENAI_API_KEY": "k", "LLM_REASONING_EFFORT": "low"}
    assert llm.resolve(provider="openai", model="m", env=env).reasoning_effort == "low"
    assert llm.resolve(provider="openai", model="m", reasoning_effort="high", env=env).reasoning_effort == "high"


def test_agent_passes_reasoning_effort_to_the_model(restore_agent, monkeypatch):
    from types import SimpleNamespace as NS
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        call = NS(id="c0", type="function", function=NS(name="propose_trade", arguments=json.dumps(
            {"action": "HOLD", "symbol": "BTC", "confidence": 0.2, "position_size_pct": 0, "leverage": 1,
             "time_horizon": "intraday", "evidence": [], "reasoning": "scripted"})))
        return NS(usage=None, choices=[NS(message=NS(content=None, tool_calls=[call]))])
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    agent.configure(provider="openai", model="gpt-x", reasoning_effort="none")
    monkeypatch.setattr(agent, "client", NS(chat=NS(completions=NS(create=create))))
    agent.run_agent("q", save=False, verbose=False)
    assert seen[0]["reasoning_effort"] == "none" and seen[0]["model"] == "gpt-x"


def test_run_records_reasoning_effort_and_resume_refuses_another(scripted, restore_agent, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    agent.configure(reasoning_effort="none")
    monkeypatch.setattr(evaluate, "MODEL", agent.MODEL)
    scripted([QuotaError(PER_DAY_429)])
    evaluate.main(["--ids", "clean_01", "--reasoning-effort", "none"])
    run_dir = next(evaluate.OUT_ROOT.iterdir())
    assert json.loads((run_dir / evaluate.RUN_FILE).read_text(encoding="utf-8"))["reasoning_effort"] == "none"
    agent.configure()  # back to the default: nothing sent
    capsys.readouterr()
    with pytest.raises(SystemExit):
        evaluate.main(["--resume", run_dir.name])
    assert "reasoning_effort none" in capsys.readouterr().err
