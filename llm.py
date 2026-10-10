"""LLM client — a thin wrapper around any OpenAI-compatible API.

Config resolution order (first match wins):
    constructor argument  >  environment / config.env  >  built-in default

Local setup: copy .env.example to config.env and edit it (gitignored).
OpenAI setup: just export OPENAI_API_KEY — no config file needed.
"""

import os

from openai import OpenAI

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.env")


def _load_config():
    """Load KEY=VALUE lines from config.env into the environment.

    Real environment variables always win (setdefault).
    """
    if not os.path.exists(_CONFIG_PATH):
        return
    with open(_CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


class LLMClient:
    """Talks to any OpenAI-compatible chat endpoint.

    OpenAI, Ollama, vLLM, llama.cpp, Groq, Together — anything that
    speaks the OpenAI chat format.
    """

    def __init__(self, model=None, api_key=None, base_url=None,
                 temperature=None, max_tokens=None, tool_choice_default=None,
                 **kwargs):
        _load_config()
        self.model = model or os.environ.get("MINIAGENT_MODEL", "gpt-4o")
        self.temperature = float(
            temperature if temperature is not None
            else os.environ.get("MINIAGENT_TEMPERATURE", 0.7)
        )
        self.max_tokens = (
            int(max_tokens) if max_tokens
            else int(os.environ["MINIAGENT_MAX_TOKENS"]) if os.environ.get("MINIAGENT_MAX_TOKENS") else None
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        # Provider behavior: some local models (e.g. Qwen) need "required"
        # to reliably call tools; OpenAI works fine with "auto".
        self.tool_choice = (
            tool_choice_default
            if tool_choice_default is not None
            else os.environ.get("MINIAGENT_TOOL_CHOICE", "auto")
        )
        client_kwargs = {"api_key": api_key, **kwargs}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
        """One LLM call. Returns the raw API response."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or self.tool_choice
        params.update(kwargs)
        return self._client.chat.completions.create(**params)
