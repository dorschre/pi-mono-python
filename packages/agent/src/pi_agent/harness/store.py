"""
Generic session stores for the agent harness.

`AgentSessionStore` is the Protocol the harness depends on. `JsonlSessionStore`
is the cwd-free engine (entries, leaf/branch/tree, append, JSONL persistence with
v1→v3 migration) extracted from pi_coding_agent.core.session_manager.
`InMemorySessionStore` is a non-persistent variant for tests and ephemeral agents.

These take a resolved ``session_file``/``sessions_dir`` directly — no cwd hashing.
The coding agent layers cwd-aware factories on top (see pi_coding_agent).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .session import (
    CURRENT_SESSION_VERSION,
    SessionContext,
    SessionEntry,
    SessionTreeNode,
    build_session_context,
    generate_id,
    migrate_to_current_version,
)


@runtime_checkable
class AgentSessionStore(Protocol):
    """Minimal persistence contract the harness depends on."""

    def append_message(self, message: dict[str, Any], parent_id: str | None = None) -> str: ...
    def append_entry(self, entry_type: str, data: dict[str, Any], parent_id: str | None = None) -> str: ...
    def get_entry(self, entry_id: str) -> SessionEntry | None: ...
    def get_leaf_id(self) -> str | None: ...
    def set_leaf_id(self, entry_id: str | None) -> None: ...
    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]: ...
    def build_context(self, leaf_id: str | None = None) -> SessionContext: ...


class JsonlSessionStore:
    """Tree-structured session store persisted as JSONL (cwd-free).

    The first line is a generic header; subsequent lines are entries linked by
    ``parentId``. Domain metadata (e.g. a working directory) goes in the header's
    ``metadata`` dict — the store itself never interprets it.
    """

    def __init__(self, session_file: str | None = None, sessions_dir: str | None = None) -> None:
        if session_file:
            self._session_file_path: str | None = os.path.abspath(session_file)
            self.sessions_dir = os.path.dirname(self._session_file_path)
        else:
            self.sessions_dir = sessions_dir or self._default_sessions_dir()
            if self.sessions_dir:
                os.makedirs(self.sessions_dir, exist_ok=True)
            self._session_file_path = None

        self._entries: list[dict[str, Any]] = []
        self._header: dict[str, Any] | None = None
        self._leaf_id: str | None = None
        self._by_id: dict[str, dict[str, Any]] = {}

        if self._session_file_path and os.path.exists(self._session_file_path):
            self._load_file()

    @staticmethod
    def _default_sessions_dir() -> str:
        return os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_file(self) -> None:
        if not self._session_file_path:
            return
        self._entries = []
        self._by_id = {}
        try:
            with open(self._session_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "session":
                        self._header = obj
                    else:
                        self._entries.append(obj)
                        if "id" in obj:
                            self._by_id[obj["id"]] = obj
        except OSError:
            pass
        if self._entries:
            self._leaf_id = self._entries[-1].get("id")

        all_entries = ([self._header] if self._header else []) + self._entries
        if migrate_to_current_version(all_entries):
            self._persist_all()

    def _persist_all(self) -> None:
        if not self._session_file_path:
            return
        lines: list[str] = []
        if self._header:
            lines.append(json.dumps(self._header, ensure_ascii=False))
        for entry in self._entries:
            lines.append(json.dumps(entry, ensure_ascii=False))
        with open(self._session_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _append_raw(self, obj: dict[str, Any]) -> None:
        if not self._session_file_path:
            return
        with open(self._session_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        sessions_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_session: str | None = None,
    ) -> "JsonlSessionStore":
        resolved_dir = sessions_dir or cls._default_sessions_dir()
        os.makedirs(resolved_dir, exist_ok=True)
        session_id = str(uuid.uuid4())
        session_file = os.path.join(resolved_dir, f"{session_id}.jsonl")
        store = cls(session_file=session_file)
        header: dict[str, Any] = {
            "type": "session",
            "id": session_id,
            "version": CURRENT_SESSION_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            header["metadata"] = dict(metadata)
        if parent_session:
            header["parentSession"] = parent_session
        store._header = header
        store._append_raw(header)
        return store

    @classmethod
    def open(cls, path: str) -> "JsonlSessionStore":
        return cls(session_file=os.path.abspath(path))

    @classmethod
    def fork_from(
        cls,
        source_path: str,
        sessions_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "JsonlSessionStore":
        source = cls.open(source_path)
        target = cls.create(sessions_dir, metadata=metadata, parent_session=source_path)
        for entry in source._entries:
            target._entries.append(entry)
            if "id" in entry:
                target._by_id[entry["id"]] = entry
            target._append_raw(entry)
        if target._entries:
            target._leaf_id = target._entries[-1].get("id")
        return target

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_session_id(self) -> str:
        return self._header.get("id", "") if self._header else ""

    def get_session_file(self) -> str | None:
        return self._session_file_path

    def get_session_dir(self) -> str:
        return self.sessions_dir

    def get_header(self) -> dict[str, Any] | None:
        return self._header

    def get_metadata(self) -> dict[str, Any]:
        return (self._header or {}).get("metadata", {})

    def get_entries(self) -> list[SessionEntry]:
        return [
            SessionEntry(
                id=e.get("id", str(uuid.uuid4())),
                type=e.get("type", "unknown"),
                timestamp=e.get("timestamp", 0),
                parent_id=e.get("parentId") or e.get("parent_id"),
                data=e,
            )
            for e in self._entries
        ]

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        raw = self._by_id.get(entry_id)
        if not raw:
            return None
        return SessionEntry(
            id=raw.get("id", entry_id),
            type=raw.get("type", "unknown"),
            timestamp=raw.get("timestamp", 0),
            parent_id=raw.get("parentId") or raw.get("parent_id"),
            data=raw,
        )

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def set_leaf_id(self, entry_id: str | None) -> None:
        self._leaf_id = entry_id

    def get_leaf_entry(self) -> SessionEntry | None:
        if self._leaf_id:
            return self.get_entry(self._leaf_id)
        entries = self.get_entries()
        return entries[-1] if entries else None

    def get_label(self, entry_id: str) -> str | None:
        label: str | None = None
        for raw in self._entries:
            if raw.get("type") == "label" and (raw.get("targetId") or raw.get("data", {}).get("targetId")) == entry_id:
                label = raw.get("label") or raw.get("data", {}).get("label")
        return label

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        leaf = self.get_entry(from_id) if from_id else self.get_leaf_entry()
        if not leaf:
            return []
        path: list[SessionEntry] = []
        current: SessionEntry | None = leaf
        while current:
            path.insert(0, current)
            current = self.get_entry(current.parent_id) if current.parent_id else None
        return path

    def get_tree(self) -> list[SessionTreeNode]:
        entries = self.get_entries()
        nodes: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []
        for entry in entries:
            nodes[entry.id] = SessionTreeNode(entry=entry, label=self.get_label(entry.id))
        for entry in entries:
            node = nodes[entry.id]
            if entry.parent_id and entry.parent_id in nodes:
                nodes[entry.parent_id].children.append(node)
            else:
                roots.append(node)
        return roots

    def build_context(self, leaf_id: str | None = None) -> SessionContext:
        entries = self.get_entries()
        by_id = {e.id: e for e in entries}
        return build_session_context(entries, leaf_id or self._leaf_id, by_id)

    # ── Append ────────────────────────────────────────────────────────────────

    def _new_id(self) -> str:
        return generate_id({e.get("id", "") for e in self._entries})

    def _store_entry(self, entry: dict[str, Any]) -> str:
        self._entries.append(entry)
        if "id" in entry:
            self._by_id[entry["id"]] = entry
        self._leaf_id = entry.get("id")
        self._append_raw(entry)
        return entry["id"]

    def append_entry(self, entry_type: str, data: dict[str, Any], parent_id: str | None = None) -> str:
        leaf = self.get_leaf_entry()
        entry: dict[str, Any] = {
            "id": self._new_id(),
            "type": entry_type,
            "timestamp": int(time.time() * 1000),
            "parentId": parent_id or (leaf.id if leaf else None),
        }
        entry.update(data)
        return self._store_entry(entry)

    def append_message(self, message: dict[str, Any], parent_id: str | None = None) -> str:
        return self.append_entry("message", {"message": message}, parent_id)

    def append_model_change(self, provider: str, model_id: str) -> str:
        return self.append_entry("model_change", {"provider": provider, "modelId": model_id})

    def append_thinking_level_change(self, level: str) -> str:
        return self.append_entry("thinking_level_change", {"thinkingLevel": level})

    def append_compaction(self, summary: str, first_kept_entry_id: str, tokens_before: int = 0) -> str:
        return self.append_entry("compaction", {
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        })

    def append_branch_summary(self, summary: str, from_id: str) -> str:
        return self.append_entry("branch_summary", {"fromId": from_id, "summary": summary})

    def append_custom_message_entry(self, custom_type: str, content: Any, display: bool = True) -> str:
        return self.append_entry("custom_message", {
            "customType": custom_type,
            "content": content,
            "display": display,
        })

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        extra: dict[str, Any] = {"customType": custom_type}
        if data is not None:
            extra["data"] = data
        return self.append_entry("custom", extra)

    def append_label_change(self, target_id: str, label: str | None) -> str:
        return self.append_entry("label", {"targetId": target_id, "label": label})

    def append_session_info(self, name: str | None = None) -> str:
        return self.append_entry("session_info", {"name": name} if name is not None else {})

    # ── Branching ─────────────────────────────────────────────────────────────

    def branch(self, branch_point_id: str | None = None, metadata: dict[str, Any] | None = None) -> "JsonlSessionStore":
        new_store = type(self).create(
            sessions_dir=self.sessions_dir,
            metadata=metadata if metadata is not None else self.get_metadata(),
            parent_session=self._session_file_path or "",
        )
        for raw in self._entries:
            new_store._entries.append(raw)
            if "id" in raw:
                new_store._by_id[raw["id"]] = raw
            new_store._append_raw(raw)
            if branch_point_id and raw.get("id") == branch_point_id:
                break
        new_store._leaf_id = branch_point_id or (new_store._entries[-1].get("id") if new_store._entries else None)
        return new_store


class InMemorySessionStore(JsonlSessionStore):
    """Non-persistent store — keeps the tree in memory only (no disk I/O)."""

    def __init__(self) -> None:
        super().__init__(session_file=None, sessions_dir=None)
        self._header = {
            "type": "session",
            "id": str(uuid.uuid4()),
            "version": CURRENT_SESSION_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def create(cls, sessions_dir=None, metadata=None, parent_session=None) -> "InMemorySessionStore":  # type: ignore[override]
        store = cls()
        if metadata:
            store._header["metadata"] = dict(metadata)
        if parent_session:
            store._header["parentSession"] = parent_session
        return store

    def _append_raw(self, obj: dict[str, Any]) -> None:
        return  # in memory only

    def _persist_all(self) -> None:
        return
