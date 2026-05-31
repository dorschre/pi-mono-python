"""
Tests for the state-transition adapter layer (pi_agent/state.py) and its integration
into the agent loop.

Covers:
1. AppendOnlyStateSystem unit behavior (seed/apply/materialize/new_messages/version + records).
2. Parity regression — the default loop (no factory) produces the same agent_end messages
   as the historical append-only behavior.
3. Injection — a recording stub observes apply() calls in the right order with actors/evidence.
4. Injection — a stub that trims materialize() changes what reaches convert_to_llm/stream_fn
   without altering the transition log.
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
    EventToolCallEnd,
    EventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AppendOnlyStateSystem,
    Observation,
    StateTransitionSystem,
    agent_loop,
)


def _ts() -> int:
    return int(time.time() * 1000)


def _user(text: str) -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=_ts())


def _assistant(text: str = "Hi!") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="test-model",
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )


def _keep_llm(msgs):
    return [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")]


# ─── 1. AppendOnlyStateSystem unit ──────────────────────────────────────────────


def test_append_only_seed_and_materialize():
    sys = AppendOnlyStateSystem()
    seed = [_user("prior")]
    sys.seed(seed)
    assert sys.materialize() == seed
    assert sys.new_messages() == []
    assert sys.version == 0

    # seed is copied, not aliased
    seed.append(_user("mutation after seed"))
    assert len(sys.materialize()) == 1


def test_append_only_apply_records_transitions():
    sys = AppendOnlyStateSystem()
    sys.seed([_user("prior")])

    m1 = _user("hello")
    t1 = sys.apply(Observation(type="prompt", actor="user", message=m1))
    m2 = _assistant("hi")
    t2 = sys.apply(Observation(type="assistant", actor="assistant", message=m2))

    # version is monotonic with from/to chaining
    assert (t1.from_version, t1.to_version) == (0, 1)
    assert (t2.from_version, t2.to_version) == (1, 2)
    assert sys.version == 2

    # ids are stable and unique
    assert t1.id and t2.id and t1.id != t2.id

    # materialize = seed + applied; new_messages = applied only
    assert sys.materialize() == [sys.materialize()[0], m1, m2]
    assert sys.new_messages() == [m1, m2]

    # the transition log preserves type/actor/message
    log = sys.transitions()
    assert [t.type for t in log] == ["prompt", "assistant"]
    assert [t.actor for t in log] == ["user", "assistant"]
    assert log[0].message is m1 and log[1].message is m2


def test_append_only_evidence_and_metadata_copied():
    sys = AppendOnlyStateSystem()
    evidence = ["tc1"]
    meta = {"confidence": 0.9}
    t = sys.apply(Observation(
        type="tool_result", actor="tool", message=None,
        evidence=evidence, metadata=meta,
    ))
    # message=None still records a transition (version advances)
    assert sys.version == 1
    assert t.evidence == ["tc1"] and t.metadata == {"confidence": 0.9}
    # copies, not aliases
    evidence.append("tc2")
    meta["confidence"] = 0.0
    assert t.evidence == ["tc1"] and t.metadata == {"confidence": 0.9}


def test_append_only_satisfies_protocol():
    assert isinstance(AppendOnlyStateSystem(), StateTransitionSystem)


# ─── shared mock streams ────────────────────────────────────────────────────────


async def _stream_text(model, context, options=None) -> AsyncGenerator:
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)
    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="Hi!")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)
    yield EventTextEnd(type="text_end", content_index=0, content="Hi!", partial=with_text)
    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text="Hi!")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


def _make_tool(executed: list):
    async def execute(tool_call_id, params, cancel=None, on_update=None):
        executed.append(params)
        return AgentToolResult(
            content=[TextContent(type="text", text="5")],
            details={"sum": 5},
        )

    return AgentTool(
        name="calculator", label="calculator", description="Add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        execute=execute,
    )


def _stream_toolcall_then_text():
    call_count = [0]

    async def _stream(m, ctx, opts=None):
        call_count[0] += 1
        partial = AssistantMessage(
            role="assistant", content=[], api=m.api, provider=m.provider,
            model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        if call_count[0] == 1:
            tc = ToolCall(type="toolCall", id="tc1", name="calculator", arguments={"a": 2, "b": 3})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant", content=[tc], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
        else:
            with_text = partial.model_copy(update={"content": [TextContent(type="text", text="5")]})
            yield EventTextStart(type="text_start", content_index=0, partial=with_text)
            yield EventTextEnd(type="text_end", content_index=0, content="5", partial=with_text)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="5")],
                api=m.api, provider=m.provider, model=m.id, usage=Usage(),
                stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

    return _stream


# ─── 2. Parity regression ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_loop_parity_returns_full_sequence():
    """Default loop (no factory) yields prompt → assistant(toolUse) → toolResult → assistant."""
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    executed: list = []
    config = AgentLoopConfig(model=model, convert_to_llm=_keep_llm)
    context = AgentContext(messages=[], tools=[_make_tool(executed)])

    stream = agent_loop([_user("What is 2+3?")], context, config,
                        stream_fn=_stream_toolcall_then_text())
    async for _ in stream:
        pass
    result = await stream.result()

    roles = [m.role for m in result]
    assert roles == ["user", "assistant", "toolResult", "assistant"]
    assert executed == [{"a": 2, "b": 3}]


@pytest.mark.asyncio
async def test_default_loop_seeds_prior_context():
    """Prior context is seeded (not in new_messages) but is visible to the model."""
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    seen_message_counts: list = []

    async def _stream_capture(m, ctx, opts=None):
        seen_message_counts.append(len(ctx.messages))
        async for ev in _stream_text(m, ctx, opts):
            yield ev

    config = AgentLoopConfig(model=model, convert_to_llm=_keep_llm)
    context = AgentContext(messages=[_user("old turn"), _assistant("old reply")])

    stream = agent_loop([_user("new")], context, config, stream_fn=_stream_capture)
    async for _ in stream:
        pass
    result = await stream.result()

    # Model saw seed (2) + new prompt (1) = 3 messages
    assert seen_message_counts == [3]
    # new_messages delta excludes the seed: just the new prompt + assistant reply
    assert [m.role for m in result] == ["user", "assistant"]


# ─── 3. Injection — observation order ───────────────────────────────────────────


class _RecordingSystem(AppendOnlyStateSystem):
    """Append-only, but records every Observation for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[Observation] = []

    def apply(self, observation: Observation):
        self.observations.append(observation)
        return super().apply(observation)


@pytest.mark.asyncio
async def test_injected_system_observes_transitions_in_order():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    executed: list = []
    recorder = _RecordingSystem()

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=_keep_llm,
        state_system=lambda: recorder,
    )
    context = AgentContext(messages=[], tools=[_make_tool(executed)])

    stream = agent_loop([_user("What is 2+3?")], context, config,
                        stream_fn=_stream_toolcall_then_text())
    async for _ in stream:
        pass

    types = [o.type for o in recorder.observations]
    actors = [o.actor for o in recorder.observations]
    assert types == ["prompt", "assistant", "tool_result", "assistant"]
    assert actors == ["user", "assistant", "tool", "assistant"]

    # the tool_result observation carries the tool_call_id as evidence
    tool_obs = recorder.observations[2]
    assert tool_obs.evidence == ["tc1"]


# ─── 4. Injection — materialize() controls model-visible context ────────────────


class _LastNSystem(AppendOnlyStateSystem):
    """Materializes only the last N messages (model-visible), full log unaffected."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def materialize(self):
        return super().materialize()[-self._n:]


@pytest.mark.asyncio
async def test_injected_system_trims_model_visible_context():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    seen_counts: list = []

    async def _stream_capture(m, ctx, opts=None):
        seen_counts.append(len(ctx.messages))
        async for ev in _stream_text(m, ctx, opts):
            yield ev

    trimmer = _LastNSystem(n=1)
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=_keep_llm,
        state_system=lambda: trimmer,
    )
    # seed two prior messages + one new prompt → materialize would be 3, trimmed to 1
    context = AgentContext(messages=[_user("a"), _assistant("b")])

    stream = agent_loop([_user("c")], context, config, stream_fn=_stream_capture)
    async for _ in stream:
        pass

    # Model only saw the last 1 message despite 3 in working state
    assert seen_counts == [1]
    # But the transition log / new_messages are unaffected by the view trimming
    assert trimmer.version == 2  # prompt + assistant
    assert [m.role for m in trimmer.new_messages()] == ["user", "assistant"]
