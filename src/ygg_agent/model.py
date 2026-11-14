# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The model side: anything that turns a conversation into text.

The agent needs one method, `complete(messages) -> Completion`. `OpenAICompatible` speaks
the /v1/chat/completions API that llama.cpp, vLLM, Ollama, OpenAI and most gateways serve;
`LlamaCpp` loads a GGUF file and computes the tokens in this process, with no model server
and no HTTP; `ScriptedModel` replays fixed answers, for tests and for proving a harness
without a model. Nothing here knows about the database, and nothing here speaks MCP.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Model(Protocol):
    def complete(self, messages: list[dict]) -> Completion: ...


class ModelError(RuntimeError):
    """The model failed, or answered something that is not a completion."""


def _with_suffix(messages: list[dict], suffix: str) -> list[dict]:
    if not suffix:
        return messages
    return [dict(m, content=m["content"] + suffix) if m["role"] == "user" else m for m in messages]


def _completion_of(out: Any) -> Completion:
    """A chat completion in the /v1/chat/completions shape, from a server or from llama.cpp."""
    try:
        text = out["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise ModelError(f"not a chat completion: {str(out)[:300]}") from e
    usage = out.get("usage") or {}
    return Completion(text, int(usage.get("prompt_tokens", 0)),
                      int(usage.get("completion_tokens", 0)))


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
        messages = _with_suffix(messages, self.user_suffix)
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
        return _completion_of(out)


@dataclass
class LlamaCpp:
    """A GGUF model loaded in THIS process with llama.cpp. No model server, no HTTP.

    The weights are read from a file and the tokens are computed here, through
    llama-cpp-python (`pip install "ygg-agent[local]"`). With it the only connection the
    agent opens is PostgreSQL's own: the model writes SQL in this process and the statement
    travels over the wire protocol to the gate.

    model_path    a .gguf file
    n_ctx         the context window, in tokens
    n_gpu_layers  layers computed on the GPU: 0 is CPU only, -1 is all of them
    seed          the sampler's seed; with temperature 0 the answer does not depend on it
    chat_format   a llama-cpp-python chat format; None uses the template inside the GGUF
    lora_path     a LoRA adapter (.gguf) applied when the model is loaded
    grammar       a GBNF grammar the sampler must follow, token by token (pg_grammar_guard
                  compiles one from the live catalog)
    user_suffix   appended to every user message (e.g. " /no_think" for Qwen3)
    extra         more keyword arguments for llama_cpp.Llama

    The model is loaded on the first `complete()` (or by `load()`), once, and kept. One
    completion runs at a time: llama.cpp's context is not safe to share between threads.
    """

    model_path: str
    n_ctx: int = 8192
    n_gpu_layers: int = 0
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 0
    chat_format: str | None = None
    lora_path: str | None = None
    grammar: str | None = None
    user_suffix: str = ""
    verbose: bool = False
    extra: dict = field(default_factory=dict)
    _llm: Any = field(default=None, init=False, repr=False, compare=False)
    _grammar: Any = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def load(self) -> "LlamaCpp":
        """Load the model now instead of on the first completion. Returns self."""
        with self._lock:
            self._load()
        return self

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            import llama_cpp
        except ImportError as e:
            raise ModelError('LlamaCpp needs llama-cpp-python: pip install "ygg-agent[local]"') from e
        if not os.path.isfile(self.model_path):
            raise ModelError(f"no such model file: {self.model_path}")
        kw: dict = {"model_path": self.model_path, "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.n_gpu_layers, "seed": self.seed,
                    "verbose": self.verbose, **self.extra}
        if self.chat_format:
            kw["chat_format"] = self.chat_format
        if self.lora_path:
            kw["lora_path"] = self.lora_path
        try:
            grammar = (llama_cpp.LlamaGrammar.from_string(self.grammar, verbose=self.verbose)
                       if self.grammar else None)
            llm = llama_cpp.Llama(**kw)
        except Exception as e:
            raise ModelError(f"llama.cpp could not load {self.model_path}: {e}") from e
        self._grammar, self._llm = grammar, llm

    def complete(self, messages: list[dict]) -> Completion:
        messages = _with_suffix(messages, self.user_suffix)
        args: dict = {"messages": messages, "temperature": self.temperature,
                      "max_tokens": self.max_tokens, "stream": False}
        with self._lock:
            self._load()
            if self._grammar is not None:
                args["grammar"] = self._grammar
            try:
                out = self._llm.create_chat_completion(**args)
            except Exception as e:
                raise ModelError(f"llama.cpp failed while completing: {e}") from e
        return _completion_of(out)


class ScriptedModel:
    """Answers with the given replies, in order; then DONE. Records what it was asked."""

    def __init__(self, replies: Iterable[str]):
        self._replies = list(replies)
        self.seen: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return Completion(self._replies.pop(0) if self._replies else "DONE")
