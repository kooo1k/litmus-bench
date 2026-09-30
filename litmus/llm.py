"""
llm.py

Which model the agent talks to. The harness works with any OpenAI-compatible
chat-completions endpoint that supports tool calling (OpenAI, Gemini,
Anthropic, OpenRouter, a local Ollama or vLLM server, ...): developers bring
the model, the harness brings the agent, tools, risk governor and grading.

Resolution, first match wins:
    model     --model        > LLM_MODEL        > preset default / GEMINI_MODEL (gemini only)
    base URL  --base-url     > --provider preset > LLM_BASE_URL > Gemini's endpoint
    API key   preset's var (e.g. OPENAI_API_KEY)  > LLM_API_KEY > GEMINI_API_KEY (gemini only)
    provider  --provider     > LLM_PROVIDER     > inferred from the base URL
    reasoning --reasoning-effort > LLM_REASONING_EFFORT > not sent (the model's default)

With nothing set but GEMINI_API_KEY and GEMINI_MODEL, this resolves to
exactly what agent.py used before (Gemini's OpenAI-compatible endpoint).
Verified live: Gemini and OpenAI (2026-09-28); the other presets are verified offline.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import OpenAI

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# name -> (base URL, API-key environment variable or None, default model or None)
PROVIDERS = {
    "gemini": (GEMINI_URL, "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "ollama": ("http://localhost:11434/v1", None, None),
}

# The OpenAI SDK refuses an empty key; a keyless local server ignores it.
KEYLESS = "not-needed"

# Per-request timeout. The SDK's default is 600s per attempt, with 2 retries:
# a request the provider never answers froze the first live CLI run
# (2026-09-27) with no output for 5+ minutes. A normal request here takes
# 2-10s even with ~23 KB of tool descriptions, so 60s only cuts off hangs.
# Worst case is now ~3 minutes (3 attempts), then a clear error.
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2


class ConfigError(ValueError):
    """The model can't be resolved: unknown provider, or no model name."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str          # a PROVIDERS name, or "custom"
    model: str
    base_url: str
    api_key: str
    key_var: str           # where the key was (or should be) read from, for error messages
    notice: str = ""       # a one-line note for the user, e.g. a corrected URL
    reasoning_effort: str = ""  # sent as reasoning_effort when set, e.g. "none" or "low"

    @property
    def has_key(self):
        return self.api_key != "" and self.api_key is not None


def _infer_provider(base_url):
    host = urlparse(base_url).netloc
    for name, (url, _, _) in PROVIDERS.items():
        if urlparse(url).netloc == host:
            return name
    return "custom"


def _normalize(base_url):
    """Gemini's native root (".../v1beta/") is a common mistake for its
    OpenAI-compatible endpoint (".../v1beta/openai/"): every call would 404.
    Correct exactly that case and say so; leave every other URL alone."""
    parsed = urlparse(base_url)
    if parsed.netloc == "generativelanguage.googleapis.com" and "/openai" not in parsed.path:
        return GEMINI_URL, f"base URL {base_url} is Gemini's native API; using its OpenAI-compatible endpoint {GEMINI_URL}"
    return base_url, ""


def resolve(model=None, base_url=None, provider=None, env=None, reasoning_effort=None):
    env = os.environ if env is None else env
    provider = provider or env.get("LLM_PROVIDER") or None
    if provider is not None and provider not in PROVIDERS:
        raise ConfigError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)} or pass --base-url")

    if base_url:
        url = base_url
    elif provider:
        url = PROVIDERS[provider][0]
    else:
        url = env.get("LLM_BASE_URL") or GEMINI_URL
    url, notice = _normalize(url)
    name = provider or _infer_provider(url)

    preset_key_var = PROVIDERS[name][1] if name in PROVIDERS else None
    if name == "ollama":
        key, key_var = env.get("LLM_API_KEY") or KEYLESS, "LLM_API_KEY"
    elif preset_key_var and env.get(preset_key_var):
        key, key_var = env[preset_key_var], preset_key_var
    else:
        key, key_var = env.get("LLM_API_KEY", ""), preset_key_var or "LLM_API_KEY"

    chosen = model or env.get("LLM_MODEL")
    if not chosen and name == "gemini":
        chosen = env.get("GEMINI_MODEL") or PROVIDERS["gemini"][2]
    if not chosen:
        raise ConfigError(f"no model set for provider {name!r}: pass --model or set LLM_MODEL")
    effort = reasoning_effort or env.get("LLM_REASONING_EFFORT") or ""
    return LLMConfig(provider=name, model=chosen, base_url=url, api_key=key, key_var=key_var, notice=notice,
                     reasoning_effort=effort)


def request_options(config):
    """Extra chat-completions arguments for `config`: only what was set, so a
    provider that doesn't know reasoning_effort never receives it. Some models
    need it: OpenAI's gpt-6-sol and gpt-5.6 accept function tools on Chat
    Completions only with reasoning_effort "none" (probed 2026-09-28)."""
    return {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}


def make_client(config):
    """An OpenAI SDK client for `config`. A missing key doesn't fail here (the
    module must import without credentials); the CLI checks has_key first and
    names the variable to set."""
    return OpenAI(api_key=config.api_key or "missing-key", base_url=config.base_url,
                  timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
