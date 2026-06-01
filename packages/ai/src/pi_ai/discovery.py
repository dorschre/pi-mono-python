"""
Model autodiscovery for OpenAI-compatible endpoints.

Every OpenAI-compatible provider implements `GET {base_url}/models`. This module
calls that listing (via the `openai` SDK's `AsyncOpenAI(...).models.list()`, the same
client the streaming provider uses) and turns the result into `Model` objects, so a
configured endpoint can report the models it actually serves instead of relying solely
on the static generated table.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import openai as _openai

from .env_api_keys import get_env_api_key
from .models import get_custom_model
from .models_generated import MODELS
from .types import Model

_CANONICAL_OPENAI_URL = "https://api.openai.com/v1"


def _openai_compatible_endpoints() -> dict[str, str]:
    """Derive ``{provider: base_url}`` from the built-in table for providers whose
    models use the OpenAI Chat Completions API. Keeps the map in sync with the
    generated models rather than hardcoding URLs."""
    endpoints: dict[str, str] = {}
    for model in MODELS.values():
        if model.api == "openai-completions" and model.provider not in endpoints:
            if model.base_url:
                endpoints[model.provider] = model.base_url
    return endpoints


@dataclass
class EndpointDiscovery:
    """Result of discovering models at a single endpoint."""
    provider: str
    base_url: str
    models: list[Model] = field(default_factory=list)
    error: str | None = None


async def discover_models(
    base_url: str,
    api_key: str | None = None,
    provider: str = "custom",
    *,
    timeout_s: float = 10.0,
    headers: dict[str, str] | None = None,
) -> list[Model]:
    """List the models served by an OpenAI-compatible endpoint.

    Calls ``GET {base_url}/models`` and builds a `Model` per returned id. The listing
    payload only carries ``id`` (and ``created``/``owned_by``), so other fields take
    sensible defaults. Raises on transport/HTTP errors — callers that sweep multiple
    endpoints should catch per endpoint (see `discover_all_models`).
    """
    # Mirror the streaming path: the canonical OpenAI URL is passed as None so the SDK
    # uses its built-in default.
    resolved_base = None if base_url == _CANONICAL_OPENAI_URL else base_url
    client = _openai.AsyncOpenAI(
        api_key=api_key or "none",
        base_url=resolved_base,
        default_headers=headers or None,
        timeout=timeout_s,
        max_retries=0,
    )
    try:
        page = await client.models.list()
        entries = list(getattr(page, "data", None) or [])
        models: list[Model] = []
        for entry in entries:
            model_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
            if not model_id:
                continue
            models.append(Model(
                id=model_id,
                name=model_id,
                api="openai-completions",
                provider=provider,
                base_url=base_url,
                input=["text"],
            ))
        return models
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def discover_all_models(
    *,
    providers: list[str] | None = None,
    timeout_s: float = 10.0,
) -> list[EndpointDiscovery]:
    """Discover models across all env-configured OpenAI-compatible endpoints + CUSTOM_*.

    Builds the candidate ``(provider, base_url, api_key)`` set from
    `_openai_compatible_endpoints()` (those with an env key) plus the `CUSTOM_*`
    endpoint, then lists each concurrently. Per-endpoint failures are captured in
    `EndpointDiscovery.error` rather than aborting the sweep.
    """
    candidates: list[tuple[str, str, str | None]] = []

    endpoints = _openai_compatible_endpoints()
    for provider, base_url in endpoints.items():
        if providers is not None and provider not in providers:
            continue
        key = get_env_api_key(provider)
        if key:
            candidates.append((provider, base_url, key))

    # CUSTOM_* endpoint (not in the generated table).
    custom = get_custom_model()
    if custom and (providers is None or "custom" in providers):
        candidates.append(("custom", custom.base_url, get_env_api_key("custom")))

    async def _one(provider: str, base_url: str, key: str | None) -> EndpointDiscovery:
        try:
            models = await discover_models(base_url, key, provider, timeout_s=timeout_s)
            return EndpointDiscovery(provider=provider, base_url=base_url, models=models)
        except Exception as e:
            return EndpointDiscovery(provider=provider, base_url=base_url, error=str(e))

    if not candidates:
        return []
    return list(await asyncio.gather(*(_one(p, b, k) for p, b, k in candidates)))
