# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The model side: anything that turns a conversation into text.

The agent needs one method, `complete(messages) -> Completion`. `OpenAICompatible` speaks
the /v1/chat/completions API that llama.cpp, vLLM, Ollama, OpenAI and most gateways serve;
`ScriptedModel` replays fixed answers, for tests and for proving a harness without a model.
Nothing here knows about the database, and nothing here speaks MCP.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Model(Protocol):
    def complete(self, messages: list[dict]) -> Completion: ...


class ModelError(RuntimeError):
    """The model server failed or answered something that is not a completion."""


@dataclass
class OpenAICompatible:
    """A /v1/chat/completions endpoint.

    base_url     e.g. "http://127.0.0.1:8080/v1" or "https://api.openai.com/v1"
    model        the model name, when the server needs one
    api_key      sent as a bearer token, when the server needs one
    user_suffix  appended to every user message (e.g. " /no_think" for Qwen3)
    extra_body   merged into every request (e.g. a llama.cpp "lora" list)
    """

    base_url: str
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 300.0
    user_suffix: str = ""
    extra_body: dict = field(default_factory=dict)

    def complete(self, messages: list[dict]) -> Completion:
        if self.user_suffix:
            messages = [dict(m, content=m["content"] + self.user_suffix) if m["role"] == "user" else m
                        for m in messages]
        body = {"messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "stream": False, **self.extra_body}
        if self.model:
            body["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            raise ModelError(f"model server answered HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise ModelError(f"model server unreachable or not JSON: {e}") from e
        try:
            text = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"not a chat completion: {str(out)[:300]}") from e
        usage = out.get("usage") or {}
        return Completion(text, int(usage.get("prompt_tokens", 0)),
                          int(usage.get("completion_tokens", 0)))


class ScriptedModel:
    """Answers with the given replies, in order; then DONE. Records what it was asked."""

    def __init__(self, replies: Iterable[str]):
        self._replies = list(replies)
        self.seen: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return Completion(self._replies.pop(0) if self._replies else "DONE")
