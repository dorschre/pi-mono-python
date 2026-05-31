"""
Harness lifecycle events and phase enum.

The harness forwards the wrapped Agent's events unchanged and additionally emits
these lifecycle events (as lightweight objects) to subscribers. Hook names for
features built on top of the harness (compaction, branch summaries) are declared
here but not implemented in this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Phase state machine (mirrors the TS harness phases).
HarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]

# Extension/recovery hook points other features attach to. Declared, not implemented here.
HookName = Literal["before_compact", "compact", "before_tree", "tree"]


@dataclass
class HarnessSessionStart:
    type: Literal["session_start"] = "session_start"
    session_id: str = ""


@dataclass
class HarnessSessionSwitch:
    session_id: str
    type: Literal["session_switch"] = "session_switch"


@dataclass
class HarnessSessionFork:
    source_session_id: str
    new_session_id: str
    entry_id: str | None = None
    type: Literal["session_fork"] = "session_fork"


@dataclass
class HarnessActionStart:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    type: Literal["action_start"] = "action_start"


@dataclass
class HarnessActionEnd:
    name: str
    result: Any = None
    error: str | None = None
    type: Literal["action_end"] = "action_end"


HarnessLifecycleEvent = (
    HarnessSessionStart
    | HarnessSessionSwitch
    | HarnessSessionFork
    | HarnessActionStart
    | HarnessActionEnd
)
