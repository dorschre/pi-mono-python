"""
Tests for AgentHarness: append via turns, build context, fork, switch_branch,
run_action, and event emission. Uses a mock stream_fn and an InMemorySessionStore.
Must NOT import pi_coding_agent (layering acceptance criterion).
"""
from __future__ import annotations

import time
from typing import AsyncGenerator

import pytest

from pi_ai import get_model
from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    TextContent,
    Usage,
)
from pi_agent import Agent, AgentOptions
from pi_agent.harness import (
    AgentHarness,
    FunctionAction,
    HarnessActionResult,
    InMemorySessionStore,
)


def _ts() -> int:
    return int(time.time() * 1000)


def _model():
    return get_model("anthropic", "claude-3-5-sonnet-20241022")


async def _mock_stream(model, context, options=None) -> AsyncGenerator:
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)
    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="ok")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)
    yield EventTextEnd(type="text_end", content_index=0, content="ok", partial=with_text)
    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text="ok")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


def _make_harness() -> AgentHarness:
    agent = Agent(AgentOptions(stream_fn=_mock_stream))
    agent.set_model(_model())
    return AgentHarness(agent, InMemorySessionStore())


@pytest.mark.asyncio
async def test_turn_persists_messages_to_store():
    h = _make_harness()
    await h.prompt("hello")
    await h.agent.wait_for_idle()

    ctx = h.session_store.build_context()
    roles = [m["role"] for m in ctx.messages]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_events_forwarded_and_lifecycle_emitted():
    h = _make_harness()
    seen: list = []
    h.subscribe(lambda e: seen.append(getattr(e, "type", None)))
    await h.prompt("hi")
    await h.agent.wait_for_idle()
    # agent events are forwarded
    assert "agent_start" in seen and "agent_end" in seen
    assert "message_end" in seen


@pytest.mark.asyncio
async def test_run_action_custom_and_events():
    h = _make_harness()
    events: list = []
    h.subscribe(lambda e: events.append(e))

    async def _echo(harness, args):
        return HarnessActionResult(text=args.get("msg", ""), details={"ok": True})

    h.register_action(FunctionAction("echo", _echo))
    result = await h.run_action("echo", {"msg": "hey"})
    assert result.text == "hey" and result.details == {"ok": True}

    types = [getattr(e, "type", None) for e in events]
    assert "action_start" in types and "action_end" in types


@pytest.mark.asyncio
async def test_run_action_unknown_raises():
    h = _make_harness()
    with pytest.raises(ValueError):
        await h.run_action("nope")


@pytest.mark.asyncio
async def test_fork_at_leaf_branches_before_it():
    # Faithful to AgentSession.fork: forking at the leaf assistant branches at its
    # parent (the user message), so the branch can be re-prompted.
    h = _make_harness()
    await h.prompt("first")
    await h.agent.wait_for_idle()

    forked = await h.fork()
    assert isinstance(forked, AgentHarness)
    assert forked._session_id() != h._session_id()
    assert [m["role"] for m in forked.session_store.build_context().messages] == ["user"]
    # original is unaffected
    assert [m["role"] for m in h.session_store.build_context().messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_fork_at_root_copies_full_session():
    # Forking at an entry with no parent (the first message) full-copies the session.
    h = _make_harness()
    await h.prompt("first")
    await h.agent.wait_for_idle()
    root_id = h.session_store.get_branch()[0].id

    forked = await h.fork(root_id)
    assert [m["role"] for m in forked.session_store.build_context().messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_switch_branch_restores_context():
    h = _make_harness()
    await h.prompt("q1")
    await h.agent.wait_for_idle()
    # entry id of the first (user) message
    first_entry = h.session_store.get_branch()[0].id
    await h.prompt("q2")
    await h.agent.wait_for_idle()

    ctx = await h.switch_branch(first_entry)
    assert h.session_store.get_leaf_id() == first_entry
    # context rebuilt to the branch point (just the first user message)
    assert [m["role"] for m in ctx.messages] == ["user"]
    # agent messages were reloaded as coerced model objects
    assert all(hasattr(m, "role") for m in h.agent.state.messages)


@pytest.mark.asyncio
async def test_builtin_actions_registered():
    h = _make_harness()
    for name in ("fork", "switch_branch", "resume"):
        assert name in h.get_actions()
