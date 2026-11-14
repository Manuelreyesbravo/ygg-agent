# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The model side, without a database.

LlamaCpp is checked twice. With a stand-in for llama-cpp-python, so the suite needs neither
the package nor a model: what it loads, what it passes, what it does with a failure. And,
when YGG_AGENT_TEST_GGUF names a real .gguf and llama-cpp-python is installed, with the
real thing and every Internet socket refusing to connect: the claim is "no HTTP", so the
test makes HTTP impossible and asks for a completion.
"""
import os
import socket
import sys
import types

import pytest

from ygg_agent import Completion, LlamaCpp, ModelError


class _FakeLlama:
    loaded: list[dict] = []

    def __init__(self, **kw):
        if kw["model_path"].endswith("broken.gguf"):
            raise ValueError("bad magic")
        _FakeLlama.loaded.append(kw)
        self.calls: list[dict] = []

    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        if kw["messages"][-1]["content"].startswith("explode"):
            raise RuntimeError("out of context")
        return {"choices": [{"message": {"role": "assistant", "content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


class _FakeGrammar:
    def __init__(self, text):
        self.text = text

    @classmethod
    def from_string(cls, text, verbose=False):
        if "::=" not in text:
            raise ValueError("not a grammar")
        return cls(text)


@pytest.fixture
def fake_llama(monkeypatch):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama
    mod.LlamaGrammar = _FakeGrammar
    _FakeLlama.loaded = []
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return mod


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


def test_loads_once_and_completes(fake_llama, gguf):
    m = LlamaCpp(gguf, n_ctx=4096, n_gpu_layers=-1, max_tokens=64)
    assert _FakeLlama.loaded == []                      # nothing is loaded until it is asked
    out = m.complete([{"role": "user", "content": "one"}])
    m.complete([{"role": "user", "content": "two"}])
    assert out == Completion("SELECT 1", 7, 3)
    assert len(_FakeLlama.loaded) == 1                  # one load, kept
    kw = _FakeLlama.loaded[0]
    assert (kw["model_path"], kw["n_ctx"], kw["n_gpu_layers"]) == (gguf, 4096, -1)
    call = m._llm.calls[0]
    assert call["temperature"] == 0.0 and call["max_tokens"] == 64 and call["stream"] is False
    assert "grammar" not in call


def test_load_is_explicit_too(fake_llama, gguf):
    m = LlamaCpp(gguf)
    assert m.load() is m and len(_FakeLlama.loaded) == 1
    m.complete([{"role": "user", "content": "x"}])
    assert len(_FakeLlama.loaded) == 1


def test_user_suffix_only_on_user_messages(fake_llama, gguf):
    m = LlamaCpp(gguf, user_suffix=" /no_think")
    asked = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    m.complete(asked)
    sent = m._llm.calls[0]["messages"]
    assert [x["content"] for x in sent] == ["s", "u /no_think"]
    assert asked[1]["content"] == "u"                   # the caller's list is not changed


def test_grammar_lora_chat_format_and_extra_reach_llama(fake_llama, gguf):
    m = LlamaCpp(gguf, grammar='root ::= "SELECT 1"', lora_path="/a.gguf",
                 chat_format="chatml", extra={"n_threads": 4})
    m.complete([{"role": "user", "content": "x"}])
    kw = _FakeLlama.loaded[0]
    assert kw["lora_path"] == "/a.gguf" and kw["chat_format"] == "chatml" and kw["n_threads"] == 4
    assert m._llm.calls[0]["grammar"].text == 'root ::= "SELECT 1"'


def test_failures_are_model_errors(fake_llama, gguf, tmp_path):
    with pytest.raises(ModelError, match="no such model file"):
        LlamaCpp(str(tmp_path / "absent.gguf")).complete([{"role": "user", "content": "x"}])
    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"x")
    with pytest.raises(ModelError, match="could not load"):
        LlamaCpp(str(broken)).complete([{"role": "user", "content": "x"}])
    with pytest.raises(ModelError, match="could not load"):          # a grammar that does not parse
        LlamaCpp(gguf, grammar="nonsense").complete([{"role": "user", "content": "x"}])
    with pytest.raises(ModelError, match="failed while completing"):
        LlamaCpp(gguf).complete([{"role": "user", "content": "explode"}])


def test_without_the_package_it_says_what_to_install(monkeypatch, gguf):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)           # import llama_cpp -> ImportError
    with pytest.raises(ModelError, match=r"ygg-agent\[local\]"):
        LlamaCpp(gguf).complete([{"role": "user", "content": "x"}])


def test_real_model_with_the_internet_unplugged(monkeypatch):
    path = os.environ.get("YGG_AGENT_TEST_GGUF")
    if not path:
        pytest.skip("set YGG_AGENT_TEST_GGUF to a .gguf file to run a real model in process")
    pytest.importorskip("llama_cpp")

    tried: list = []
    real_connect = socket.socket.connect

    def no_internet(self, address, *a, **kw):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            tried.append(address)
            raise OSError("no TCP in this test: the model must run in this process")
        return real_connect(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", no_internet)
    m = LlamaCpp(path, n_ctx=2048, max_tokens=24,
                 user_suffix=os.environ.get("YGG_AGENT_USER_SUFFIX", ""))
    out = m.complete([{"role": "system", "content": "Answer with one SQL statement and nothing else."},
                      {"role": "user", "content": "Write a statement that returns the number 1."}])
    assert out.text.strip() and out.prompt_tokens > 0 and out.completion_tokens > 0
    assert tried == []                                  # it never even tried
