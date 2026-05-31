"""
Generic session-tree models and context reconstruction for the agent harness.

Moved (and cwd-decoupled) from pi_coding_agent.core.session_manager so that any
agent — not just the coding agent — can persist messages, navigate branches, and
rebuild context. Nothing here knows about coding tools, cwd, or transports; the
session header carries an optional generic ``metadata`` dict instead of a ``cwd``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pi_ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage

CURRENT_SESSION_VERSION = 3

# Entry types (mirrors TypeScript session entry types)
SessionEntryType = Literal[
    "session",
    "message",
    "compaction",
    "branch_summary",
    "model_change",
    "thinking_level_change",
    "custom_message",
    "custom",
    "session_info",
    "label",
]


@dataclass
class SessionEntry:
    """A single entry in a session tree."""
    id: str
    type: str
    timestamp: int
    parent_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionHeader:
    """Session header (first record). Generic: domain data lives in ``metadata``."""
    type: str
    id: str
    timestamp: str
    version: int = CURRENT_SESSION_VERSION
    parent_session: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionTreeNode:
    """Tree node for get_tree() — defensive copy of session structure."""
    entry: SessionEntry
    children: list["SessionTreeNode"] = field(default_factory=list)
    label: str | None = None


@dataclass
class SessionContext:
    """Reconstructed conversation context for the agent."""
    messages: list[dict[str, Any]]
    thinking_level: str
    model: dict[str, str] | None  # {"provider": ..., "model_id": ...}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def generate_id(existing_ids: set[str]) -> str:
    """Generate a unique 8-hex-char ID, checking for collisions."""
    for _ in range(100):
        candidate = str(uuid.uuid4()).replace("-", "")[:8]
        if candidate not in existing_ids:
            return candidate
    return str(uuid.uuid4())


def migrate_v1_to_v2(entries: list[dict[str, Any]]) -> None:
    """Migrate v1 → v2: add id/parentId tree structure. Mutates in place."""
    ids: set[str] = set()
    prev_id: str | None = None

    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue

        entry["id"] = generate_id(ids)
        ids.add(entry["id"])
        entry["parentId"] = prev_id
        prev_id = entry["id"]

        # Convert firstKeptEntryIndex → firstKeptEntryId for compaction
        if entry.get("type") == "compaction":
            idx = entry.pop("firstKeptEntryIndex", None)
            if isinstance(idx, int) and 0 <= idx < len(entries):
                target = entries[idx]
                if target.get("type") != "session":
                    entry["firstKeptEntryId"] = target.get("id")


def migrate_v2_to_v3(entries: list[dict[str, Any]]) -> None:
    """Migrate v2 → v3: rename hookMessage role to custom. Mutates in place."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "hookMessage":
                msg["role"] = "custom"


def migrate_to_current_version(entries: list[dict[str, Any]]) -> bool:
    """Run all necessary migrations. Mutates in place. Returns True if any applied."""
    header = next((e for e in entries if e.get("type") == "session"), None)
    version = header.get("version", 1) if header else 1

    if version >= CURRENT_SESSION_VERSION:
        return False

    if version < 2:
        migrate_v1_to_v2(entries)
    if version < 3:
        migrate_v2_to_v3(entries)

    return True


def get_latest_compaction_entry(entries: list[SessionEntry]) -> SessionEntry | None:
    """Get the most recent compaction entry, if any."""
    for entry in reversed(entries):
        if entry.type == "compaction":
            return entry
    return None


def coerce_messages(messages: list[Any]) -> list[Message]:
    """Coerce a mix of dicts / pi_ai Message objects into Message objects.

    Restored ``SessionContext`` messages are plain dicts; converting the standard
    user/assistant/toolResult roles back into pi_ai models lets a plain ``Agent``
    (and real providers) consume them. Unknown roles (custom/branch/compaction
    message types) are left as-is for an application-level ``convert_to_llm`` to
    handle, and anything that fails validation is dropped from the model view.
    """
    out: list[Message] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        model_cls = {
            "user": UserMessage,
            "assistant": AssistantMessage,
            "toolResult": ToolResultMessage,
        }.get(role)
        if model_cls is None:
            out.append(m)  # custom/branch/compaction — leave for app convert_to_llm
            continue
        try:
            out.append(model_cls.model_validate(m))
        except Exception:
            # Not a model-valid message; skip from the coerced model view.
            continue
    return out


def build_session_context(
    entries: list[SessionEntry],
    leaf_id: str | None = None,
    by_id: dict[str, SessionEntry] | None = None,
) -> SessionContext:
    """Reconstruct context by walking from a leaf entry to the root.

    Handles compaction boundaries (emit the summary, then kept messages) and
    branch summaries. Generic — no coding/cwd assumptions.
    """
    if by_id is None:
        by_id = {e.id: e for e in entries}

    if leaf_id is None:
        # Explicitly None — navigated before the first entry.
        return SessionContext(messages=[], thinking_level="off", model=None)

    leaf: SessionEntry | None = by_id.get(leaf_id) if leaf_id else None
    if not leaf and entries:
        leaf = entries[-1]
    if not leaf:
        return SessionContext(messages=[], thinking_level="off", model=None)

    # Walk from leaf to root.
    path: list[SessionEntry] = []
    current: SessionEntry | None = leaf
    while current:
        path.insert(0, current)
        current = by_id.get(current.parent_id) if current.parent_id else None

    thinking_level = "off"
    model: dict[str, str] | None = None
    compaction: SessionEntry | None = None

    for entry in path:
        if entry.type == "thinking_level_change":
            thinking_level = entry.data.get("thinkingLevel") or entry.data.get("level", "off")
        elif entry.type == "model_change":
            model = {
                "provider": entry.data.get("provider", ""),
                "model_id": entry.data.get("modelId") or entry.data.get("model_id", ""),
            }
        elif entry.type == "message":
            msg = entry.data.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                provider = msg.get("provider", "")
                model_id = msg.get("model", "")
                if provider:
                    model = {"provider": provider, "model_id": model_id}
        elif entry.type == "compaction":
            compaction = entry

    messages: list[dict[str, Any]] = []

    def append_message(entry: SessionEntry) -> None:
        if entry.type == "message":
            msg = entry.data.get("message", {})
            if isinstance(msg, dict):
                messages.append(msg)
        elif entry.type == "custom_message":
            messages.append({
                "role": "custom",
                "customType": entry.data.get("customType", ""),
                "content": entry.data.get("content", ""),
                "display": entry.data.get("display", True),
                "timestamp": entry.timestamp,
            })
        elif entry.type == "branch_summary":
            summary = entry.data.get("summary", "")
            if summary:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"[Branch summary: {summary}]"}],
                    "timestamp": entry.timestamp,
                })

    if compaction:
        first_kept = compaction.data.get("firstKeptEntryId")
        summary = compaction.data.get("summary", "")
        tokens_before = compaction.data.get("tokensBefore", 0)

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"[Context compacted. Summary:\n{summary}]"}],
            "timestamp": compaction.timestamp,
            "_tokens_before": tokens_before,
        })

        comp_idx = next(
            (i for i, e in enumerate(path) if e.type == "compaction" and e.id == compaction.id),
            -1,
        )
        found_first_kept = False
        for i in range(comp_idx):
            entry = path[i]
            if entry.id == first_kept:
                found_first_kept = True
            if found_first_kept:
                append_message(entry)
        for i in range(comp_idx + 1, len(path)):
            append_message(path[i])
    else:
        for entry in path:
            append_message(entry)

    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)
