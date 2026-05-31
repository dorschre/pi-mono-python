"""
Tests for the generic harness session store + context reconstruction.
Must NOT import pi_coding_agent (layering acceptance criterion).
"""
from __future__ import annotations

import json

import pytest

from pi_agent.harness import (
    InMemorySessionStore,
    JsonlSessionStore,
    build_session_context,
    coerce_messages,
    migrate_to_current_version,
)


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _assistant_msg(text: str, provider: str = "anthropic", model: str = "m") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": provider,
        "model": model,
        "usage": {},
        "stop_reason": "stop",
        "timestamp": 2,
    }


def test_inmemory_append_branch_context():
    store = InMemorySessionStore()
    a = store.append_message(_user_msg("hello"))
    b = store.append_message(_assistant_msg("hi"))

    assert store.get_leaf_id() == b
    branch = store.get_branch()
    assert [e.id for e in branch] == [a, b]

    ctx = store.build_context()
    roles = [m["role"] for m in ctx.messages]
    assert roles == ["user", "assistant"]
    # model is recovered from the assistant message
    assert ctx.model == {"provider": "anthropic", "model_id": "m"}


def test_branching_creates_independent_leaf():
    store = InMemorySessionStore()
    store.append_message(_user_msg("q1"))
    a2 = store.append_message(_assistant_msg("a1"))
    store.append_message(_user_msg("q2"))

    # branch at a2 → new store ends at a2
    branched = store.branch(a2)
    assert branched.get_leaf_id() == a2
    assert [m["role"] for m in branched.build_context().messages] == ["user", "assistant"]
    # original is untouched
    assert [m["role"] for m in store.build_context().messages] == ["user", "assistant", "user"]


def test_thinking_and_model_change_entries():
    store = InMemorySessionStore()
    store.append_message(_user_msg("hi"))
    store.append_model_change("openai", "gpt-x")
    store.append_thinking_level_change("high")
    ctx = store.build_context()
    assert ctx.model == {"provider": "openai", "model_id": "gpt-x"}
    assert ctx.thinking_level == "high"


def test_compaction_boundary_in_context():
    store = InMemorySessionStore()
    store.append_message(_user_msg("old1"))
    keep = store.append_message(_assistant_msg("kept"))
    store.append_compaction(summary="did stuff", first_kept_entry_id=keep, tokens_before=123)
    store.append_message(_user_msg("after"))

    msgs = store.build_context().messages
    # first message is the compaction summary, then kept assistant, then the post message
    assert "[Context compacted." in msgs[0]["content"][0]["text"]
    assert any(m.get("role") == "assistant" for m in msgs)
    assert msgs[-1]["content"][0]["text"] == "after"


def test_jsonl_roundtrip_and_reopen(tmp_path):
    store = JsonlSessionStore.create(sessions_dir=str(tmp_path), metadata={"working_directory": "/x"})
    sid = store.get_session_id()
    store.append_message(_user_msg("persisted"))
    store.append_message(_assistant_msg("ok"))
    path = store.get_session_file()
    assert path is not None

    reopened = JsonlSessionStore.open(path)
    assert reopened.get_session_id() == sid
    assert reopened.get_metadata() == {"working_directory": "/x"}
    assert [m["role"] for m in reopened.build_context().messages] == ["user", "assistant"]


def test_v1_to_v3_migration():
    # v1 entries: flat list, no ids, hookMessage role
    entries = [
        {"type": "session", "id": "s", "version": 1},
        {"type": "message", "message": {"role": "user", "content": "hi"}},
        {"type": "message", "message": {"role": "hookMessage", "content": "note"}},
    ]
    applied = migrate_to_current_version(entries)
    assert applied is True
    assert entries[0]["version"] == 3
    # ids + parentId added
    assert "id" in entries[1] and "parentId" in entries[1]
    assert entries[1]["parentId"] is None
    assert entries[2]["parentId"] == entries[1]["id"]
    # hookMessage renamed to custom
    assert entries[2]["message"]["role"] == "custom"


def test_coerce_messages_dicts_to_models():
    coerced = coerce_messages([_user_msg("hi"), _assistant_msg("yo")])
    assert [m.role for m in coerced] == ["user", "assistant"]
    # unknown roles are passed through unchanged
    custom = {"role": "custom", "customType": "x", "content": "c"}
    out = coerce_messages([custom])
    assert out == [custom]
