"""LLM client — a thin wrapper around any OpenAI-compatible API.

Config resolution order (first match wins):
    constructor argument  >  environment / config.env  >  built-in default

Local setup: copy .env.example to config.env and edit it (gitignored).
OpenAI setup: just export OPENAI_API_KEY — no config file needed.
"""

import os
import time
from typing import Generator

from openai import OpenAI, RateLimitError

def _find_config():
    """Look for config.env in: package dir, repo root (parent), cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "config.env"),
        os.path.join(os.path.dirname(here), "config.env"),
        os.path.join(os.getcwd(), "config.env"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


_CONFIG_PATH = _find_config()

_RETRY_DELAYS = [2, 5, 15]   # seconds to wait before each retry on a 429

_config_loaded = False


def _load_config():
    """Load KEY=VALUE lines from config.env into the environment.

    Real environment variables always win (setdefault).
    Only loads once (module-level cache).
    """
    global _config_loaded
    if _config_loaded:
        return
    if not os.path.exists(_CONFIG_PATH):
        _config_loaded = True
        return
    with open(_CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    _config_loaded = True


class LLMClient:
    """Talks to any OpenAI-compatible chat endpoint.

    OpenAI, Ollama, vLLM, llama.cpp, Groq, Together — anything that
    speaks the OpenAI chat format.

    tool_choice defaults to the MINIAGENT_TOOL_CHOICE env var (or "auto").
    Some local models (e.g. Qwen) need "required" to reliably call tools;
    OpenAI works fine with "auto".

    reasoning_effort defaults to the MINIAGENT_REASONING_EFFORT env var (or
    None). Reasoning models (e.g. gpt-5.6-luna) need "none" to use function
    tools via Chat Completions.

    On a 429 (rate limit) error, chat() retries with exponential backoff
    (2s, 5s, 15s) up to max_retries times before re-raising. max_retries
    defaults to the MINIAGENT_MAX_RETRIES env var (or 3).
    """

    def __init__(self, model=None, api_key=None, base_url=None,
                 temperature=None, max_tokens=None, tool_choice_default=None,
                 reasoning_effort=None, max_retries=None, **kwargs):
        _load_config()
        self.max_retries = (
            int(max_retries) if max_retries is not None
            else int(os.environ.get("MINIAGENT_MAX_RETRIES", len(_RETRY_DELAYS)))
        )
        self.model = model or os.environ.get("MINIAGENT_MODEL", "gpt-5.6-luna")
        self.temperature = float(
            temperature if temperature is not None
            else os.environ.get("MINIAGENT_TEMPERATURE", 0.7)
        )
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.environ.get("MINIAGENT_REASONING_EFFORT")
        )
        self.max_tokens = (
            int(max_tokens) if max_tokens
            else int(os.environ["MINIAGENT_MAX_TOKENS"]) if os.environ.get("MINIAGENT_MAX_TOKENS") else None
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
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
        """One LLM call. Returns the raw API response.

        If stream=True is in kwargs, returns a generator that yields
        (chunk, full_message) pairs. The full_message is the accumulated
        ChatCompletion object (only populated on the last chunk for
        tool_calls; content is built incrementally).
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or self.tool_choice
        params.update(kwargs)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if params.get("stream"):
                    return self._stream(params)
                return self._client.chat.completions.create(**params)
            except RateLimitError as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
        raise last_error

    def _stream(self, params: dict) -> Generator:
        """Stream a chat completion. Yields (delta_text, accumulated_message_dict).

        The accumulated_message_dict has the same shape as a non-streamed
        response's choices[0].message: {"role", "content", "tool_calls"}.
        Tool call arguments are accumulated across chunks.
        """
        stream = self._client.chat.completions.create(**params)

        # Accumulator state
        content_parts = []
        tool_calls = {}  # index -> {id, name, arguments_parts}

        for chunk in stream:
            if not chunk.choices:
                # Usage chunk (some providers send it last)
                continue
            delta = chunk.choices[0].delta

            # Accumulate content
            if delta.content:
                content_parts.append(delta.content)

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments_parts": [],
                        }
                    entry = tool_calls[idx]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function and tc.function.name:
                        entry["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["arguments_parts"].append(tc.function.arguments)

            # Build the accumulated message dict
            msg = {"role": "assistant", "content": "".join(content_parts) or None}
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tool_calls[i]["id"],
                        "type": "function",
                        "function": {
                            "name": tool_calls[i]["name"],
                            "arguments": "".join(tool_calls[i]["arguments_parts"]),
                        },
                    }
                    for i in sorted(tool_calls)
                ]
            else:
                msg["tool_calls"] = None

            yield delta, msg
