"""
Gated live integration test: drive the AgentHarness against a real
OpenAI-compatible endpoint configured via CUSTOM_* env vars.

Opt-in only — skipped unless CUSTOM_ENDPOINT/CUSTOM_MODEL are set AND
LIVE_TESTS=1 (or pytest is run with --live). Sends data to the user's
configured external endpoint.

  LIVE_TESTS=1 uv run pytest packages/agent/tests/test_harness_live_custom.py
"""
from __future__ import annotations

import os

import pytest

from pi_ai import get_custom_model
from pi_agent import Agent, AgentOptions
from pi_agent.harness import AgentHarness, InMemorySessionStore


def _live_enabled() -> bool:
    return os.environ.get("LIVE_TESTS") == "1" or "--live" in (os.environ.get("PYTEST_ADDOPTS") or "")


pytestmark = pytest.mark.asyncio


async def test_harness_drives_custom_model_end_to_end():
    if not _live_enabled():
        pytest.skip("LIVE_TESTS!=1")
    model = get_custom_model()
    if model is None or not os.environ.get("CUSTOM_API_KEY"):
        pytest.skip("CUSTOM_* not configured")

    agent = Agent(AgentOptions())
    agent.set_model(model)
    agent.set_system_prompt("You are concise.")
    harness = AgentHarness(agent, InMemorySessionStore())

    await harness.prompt("Reply with exactly: HARNESS_LIVE_OK")
    await harness.agent.wait_for_idle()

    # Harness persisted the turn to its store and can rebuild it.
    ctx = harness.session_store.build_context()
    roles = [m["role"] for m in ctx.messages]
    assert roles[0] == "user"
    assert "assistant" in roles

    # The model actually answered.
    text = "".join(
        c.get("text", "")
        for m in ctx.messages if m.get("role") == "assistant"
        for c in (m.get("content") or [])
        if isinstance(c, dict) and c.get("type") == "text"
    )
    assert "HARNESS_LIVE_OK" in text or text.strip() != ""

    # Fork + switch_branch work against the live-produced session.
    first_entry = harness.session_store.get_branch()[0].id
    forked = await harness.fork()
    assert forked._session_id() != harness._session_id()
    switched = await harness.switch_branch(first_entry)
    assert harness.session_store.get_leaf_id() == first_entry
    assert [m["role"] for m in switched.messages] == ["user"]
