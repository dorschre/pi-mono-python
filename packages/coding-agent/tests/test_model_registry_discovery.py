"""
Tests for ModelRegistry autodiscovery integration (mocked — no network).
"""
from __future__ import annotations

import json

import pytest

import pi_ai.discovery as discovery
from pi_ai.types import Model
from pi_coding_agent.core.model_registry import ModelRegistry


def _fake_discover_factory():
    """Return a fake discover_models that yields one model tagged by provider."""
    async def _fake(base_url, api_key=None, provider="custom", *, timeout_s=10.0, headers=None):
        return [Model(
            id=f"{provider}-model",
            name=f"{provider}-model",
            api="openai-completions",
            provider=provider,
            base_url=base_url,
            input=["text"],
        )]
    return _fake


@pytest.fixture
def no_models_json(tmp_path):
    # Point the registry at a non-existent models.json so only built-ins load.
    return str(tmp_path / "models.json")


@pytest.mark.asyncio
async def test_discover_registers_custom_models(monkeypatch, no_models_json):
    monkeypatch.setattr(discovery, "discover_models", _fake_discover_factory())
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://chat-ai.academiccloud.de/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "qwen3.5-35b-a3b")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")

    reg = ModelRegistry(models_json_path=no_models_json)
    results = await reg.discover()

    custom = next((r for r in results if r.provider == "custom"), None)
    assert custom is not None and custom.error is None
    assert any(m.id == "custom-model" for m in custom.models)

    # Registered into the registry and selectable.
    assert reg.find("custom", "custom-model") is not None
    available = await reg.get_available()
    assert any(m.provider == "custom" and m.id == "custom-model" for m in available)


@pytest.mark.asyncio
async def test_discover_dedupes(monkeypatch, no_models_json):
    monkeypatch.setattr(discovery, "discover_models", _fake_discover_factory())
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://x.test/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "m")
    monkeypatch.setenv("CUSTOM_API_KEY", "k")

    reg = ModelRegistry(models_json_path=no_models_json)
    await reg.discover()
    count1 = sum(1 for m in reg.get_all() if m.id == "custom-model")
    await reg.discover()
    count2 = sum(1 for m in reg.get_all() if m.id == "custom-model")
    assert count1 == 1 and count2 == 1


@pytest.mark.asyncio
async def test_discover_picks_up_models_json_provider(monkeypatch, tmp_path):
    # A models.json provider declaring a base_url + OpenAI-compatible api should be a
    # discovery candidate even with no built-in models.
    models_json = tmp_path / "models.json"
    models_json.write_text(json.dumps({
        "providers": {
            "myco": {
                "baseUrl": "https://myco.example/v1",
                "api": "openai-completions",
                "apiKey": "sk-myco-literal",
            }
        }
    }))
    monkeypatch.setattr(discovery, "discover_models", _fake_discover_factory())
    monkeypatch.delenv("CUSTOM_ENDPOINT", raising=False)

    reg = ModelRegistry(models_json_path=str(models_json))
    results = await reg.discover()

    myco = next((r for r in results if r.provider == "myco"), None)
    assert myco is not None and myco.error is None
    assert reg.find("myco", "myco-model") is not None
