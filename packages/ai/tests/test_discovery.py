"""
Tests for model autodiscovery (pi_ai.discovery).

Unit tests mock the OpenAI client (no network). A gated live test (LIVE_TESTS=1 +
CUSTOM_*) hits the real configured endpoint.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import pi_ai.discovery as discovery
from pi_ai import EndpointDiscovery, discover_all_models, discover_models


class _FakeModels:
    def __init__(self, data):
        self._data = data

    async def list(self):
        return SimpleNamespace(data=self._data)


class _FakeAsyncOpenAI:
    """Records construction kwargs and returns canned model data."""
    last_kwargs: dict = {}
    data: list = []
    raise_exc: Exception | None = None

    def __init__(self, **kwargs):
        _FakeAsyncOpenAI.last_kwargs = kwargs
        if _FakeAsyncOpenAI.raise_exc is not None:
            raise _FakeAsyncOpenAI.raise_exc
        self.models = _FakeModels(_FakeAsyncOpenAI.data)

    async def close(self):
        pass


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeAsyncOpenAI.last_kwargs = {}
    _FakeAsyncOpenAI.data = []
    _FakeAsyncOpenAI.raise_exc = None
    monkeypatch.setattr(discovery._openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    return _FakeAsyncOpenAI


def test_openai_compatible_endpoints_map():
    endpoints = discovery._openai_compatible_endpoints()
    assert isinstance(endpoints, dict)
    # every value is an http(s) URL
    assert all(v.startswith("http") for v in endpoints.values())
    # the table includes OpenAI-compatible providers
    assert len(endpoints) >= 1


@pytest.mark.asyncio
async def test_discover_models_builds_models(fake_openai):
    fake_openai.data = [SimpleNamespace(id="qwen3.5-35b-a3b"), SimpleNamespace(id="llama-3")]
    models = await discover_models("https://example.test/v1/", "sk-key", "custom")

    assert [m.id for m in models] == ["qwen3.5-35b-a3b", "llama-3"]
    assert all(m.provider == "custom" for m in models)
    assert all(m.api == "openai-completions" for m in models)
    assert all(m.base_url == "https://example.test/v1/" for m in models)
    # key + base_url passed to the client
    assert fake_openai.last_kwargs["api_key"] == "sk-key"
    assert fake_openai.last_kwargs["base_url"] == "https://example.test/v1/"


@pytest.mark.asyncio
async def test_discover_models_canonical_openai_url_passes_none(fake_openai):
    fake_openai.data = [SimpleNamespace(id="gpt-x")]
    await discover_models("https://api.openai.com/v1", "sk", "openai")
    assert fake_openai.last_kwargs["base_url"] is None  # canonical → None


@pytest.mark.asyncio
async def test_discover_models_accepts_dict_entries(fake_openai):
    fake_openai.data = [{"id": "m-dict"}]
    models = await discover_models("https://x.test/v1", None, "custom")
    assert [m.id for m in models] == ["m-dict"]
    # missing key still satisfies the SDK
    assert fake_openai.last_kwargs["api_key"] == "none"


@pytest.mark.asyncio
async def test_discover_all_models_custom_only(fake_openai, monkeypatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://chat-ai.academiccloud.de/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "qwen3.5-35b-a3b")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    fake_openai.data = [SimpleNamespace(id="qwen3.5-35b-a3b")]

    results = await discover_all_models(providers=["custom"])
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EndpointDiscovery)
    assert r.provider == "custom" and r.error is None
    assert [m.id for m in r.models] == ["qwen3.5-35b-a3b"]


@pytest.mark.asyncio
async def test_discover_all_models_captures_errors(fake_openai, monkeypatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://bad.test/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "m")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk")
    fake_openai.raise_exc = RuntimeError("connection refused")

    results = await discover_all_models(providers=["custom"])
    assert len(results) == 1
    assert results[0].error is not None and "connection refused" in results[0].error
    assert results[0].models == []


# ─── gated live test ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_models_live_custom():
    if os.environ.get("LIVE_TESTS") != "1":
        pytest.skip("LIVE_TESTS!=1")
    endpoint = os.environ.get("CUSTOM_ENDPOINT")
    key = os.environ.get("CUSTOM_API_KEY")
    expected = os.environ.get("CUSTOM_MODEL")
    if not endpoint or not key:
        pytest.skip("CUSTOM_* not configured")

    models = await discover_models(endpoint, key, "custom")
    assert len(models) > 0
    ids = {m.id for m in models}
    if expected:
        assert expected in ids, f"{expected} not in discovered {sorted(ids)[:10]}..."
