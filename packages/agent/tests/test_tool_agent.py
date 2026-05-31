"""
Tests for GenericToolCallingAgent — tool registry + harness pass-through.
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
from pi_agent import AgentTool, AgentToolResult, GenericToolCallingAgent
from pi_agent.harness import FunctionAction, HarnessActionResult, InMemorySessionStore


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
    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="done")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)
    yield EventTextEnd(type="text_end", content_index=0, content="done", partial=with_text)
    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text="done")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


def _dummy_tool(name: str = "noop") -> AgentTool:
    async def execute(tool_call_id, params, cancel=None, on_update=None):
        return AgentToolResult(content=[TextContent(type="text", text="ok")], details={})

    return AgentTool(
        name=name, label=name, description="dummy",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def test_tool_registry_and_active_selection():
    ta = GenericToolCallingAgent.create(
        model=_model(), stream_fn=_mock_stream,
        tools=[_dummy_tool("a"), _dummy_tool("b")],
    )
    assert set(ta.get_tool_names()) == {"a", "b"}
    assert set(ta.get_active_tools()) == {"a", "b"}
    assert {t.name for t in ta.agent.state.tools} == {"a", "b"}

    ta.set_active_tools(["a"])
    assert ta.get_active_tools() == ["a"]
    assert {t.name for t in ta.agent.state.tools} == {"a"}


@pytest.mark.asyncio
async def test_turn_persists_via_harness():
    ta = GenericToolCallingAgent.create(
        model=_model(), stream_fn=_mock_stream, store=InMemorySessionStore(),
    )
    await ta.prompt("hello")
    await ta.agent.wait_for_idle()
    roles = [m["role"] for m in ta.session_store.build_context().messages]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_register_and_run_action():
    ta = GenericToolCallingAgent.create(model=_model(), stream_fn=_mock_stream)

    async def _ping(harness, args):
        return HarnessActionResult(text="pong")

    ta.register_action(FunctionAction("ping", _ping))
    result = await ta.run_action("ping")
    assert result.text == "pong"


@pytest.mark.asyncio
async def test_fork_carries_tools():
    ta = GenericToolCallingAgent.create(
        model=_model(), stream_fn=_mock_stream,
        tools=[_dummy_tool("a")], store=InMemorySessionStore(),
    )
    await ta.prompt("x")
    await ta.agent.wait_for_idle()
    forked = await ta.fork()
    assert isinstance(forked, GenericToolCallingAgent)
    assert forked.get_tool_names() == ["a"]
