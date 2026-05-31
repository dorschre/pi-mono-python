"""
Unit tests for custom OpenAI-compatible model support (get_custom_model).
No network — only validates Model construction from environment variables.
"""
from __future__ import annotations

import pytest

from pi_ai import get_custom_model
from pi_ai.env_api_keys import PROVIDER_ENV_VARS, get_env_api_key


def test_custom_provider_key_mapping():
    assert PROVIDER_ENV_VARS.get("custom") == "CUSTOM_API_KEY"


def test_get_custom_model_none_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CUSTOM_ENDPOINT", raising=False)
    monkeypatch.delenv("CUSTOM_MODEL", raising=False)
    assert get_custom_model() is None


def test_get_custom_model_none_when_partial(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://example.test/v1/")
    monkeypatch.delenv("CUSTOM_MODEL", raising=False)
    assert get_custom_model() is None


def test_get_custom_model_builds_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://chat-ai.academiccloud.de/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "qwen3.5-35b-a3b")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-test-123")
    monkeypatch.delenv("CUSTOM_CONTEXT_WINDOW", raising=False)

    model = get_custom_model()
    assert model is not None
    assert model.id == "qwen3.5-35b-a3b"
    assert model.name == "qwen3.5-35b-a3b"
    assert model.api == "openai-completions"
    assert model.provider == "custom"
    assert model.base_url == "https://chat-ai.academiccloud.de/v1/"
    assert model.input == ["text"]
    assert model.context_window == 128000

    # key is resolved via the provider->env mapping
    assert get_env_api_key("custom") == "sk-test-123"


def test_get_custom_model_context_window_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://example.test/v1/")
    monkeypatch.setenv("CUSTOM_MODEL", "m")
    monkeypatch.setenv("CUSTOM_CONTEXT_WINDOW", "32000")
    model = get_custom_model()
    assert model is not None and model.context_window == 32000

    # invalid value falls back to default
    monkeypatch.setenv("CUSTOM_CONTEXT_WINDOW", "not-an-int")
    model = get_custom_model()
    assert model is not None and model.context_window == 128000
