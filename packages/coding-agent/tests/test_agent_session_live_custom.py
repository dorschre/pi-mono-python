"""
Gated live test: AgentSession delegating persistence + actions to the AgentHarness,
driven against the CUSTOM_* OpenAI-compatible endpoint.

Opt-in only — skipped unless CUSTOM_ENDPOINT/CUSTOM_MODEL are set AND LIVE_TESTS=1.
Validates the Phase 3 delegation (harness persists messages; run_action dispatches)
which the standard key-gated unit tests cannot exercise in CI.

  LIVE_TESTS=1 uv run pytest packages/coding-agent/tests/test_agent_session_live_custom.py
"""
from __future__ import annotations

import os

import pytest

from pi_ai import get_custom_model
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import SessionManager

pytestmark = pytest.mark.asyncio


def _live_enabled() -> bool:
    return os.environ.get("LIVE_TESTS") == "1"


async def test_agent_session_delegates_to_harness_live(tmp_path):
    if not _live_enabled():
        pytest.skip("LIVE_TESTS!=1")
    model = get_custom_model()
    if model is None or not os.environ.get("CUSTOM_API_KEY"):
        pytest.skip("CUSTOM_* not configured")

    sm = SessionManager.create(str(tmp_path), session_dir=str(tmp_path))
    session = AgentSession(
        cwd=str(tmp_path),
        model=model,
        session_manager=sm,
    )

    events: list = []
    session.subscribe(lambda e: events.append(getattr(e, "type", None)))

    await session.prompt("Reply with exactly: HARNESS_OK")
    await session._agent.wait_for_idle()

    # Harness persisted the turn to the session store.
    ctx = sm.build_context()
    roles = [m["role"] for m in ctx.messages]
    assert "user" in roles and "assistant" in roles

    # Session events were forwarded to subscribers.
    assert "agent_start" in events and "agent_end" in events

    # Generic action dispatch works through the harness.
    assert "fork" in session.harness.get_actions()
    result = await session.run_action("fork")
    assert result.details.get("session_id")
