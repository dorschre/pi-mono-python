"""
pi_agent.harness — reusable session/branch/action runtime around an Agent.

Generic by design: no coding tools, cwd, or transport dependencies. The coding
agent (and other products) layer their specifics on top.
"""
from .actions import FunctionAction, HarnessAction, HarnessActionResult
from .agent_harness import AgentHarness
from .events import (
    HarnessActionEnd,
    HarnessActionStart,
    HarnessPhase,
    HarnessSessionFork,
    HarnessSessionStart,
    HarnessSessionSwitch,
)
from .session import (
    CURRENT_SESSION_VERSION,
    SessionContext,
    SessionEntry,
    SessionEntryType,
    SessionHeader,
    SessionTreeNode,
    build_session_context,
    coerce_messages,
    generate_id,
    get_latest_compaction_entry,
    migrate_to_current_version,
)
from .store import AgentSessionStore, InMemorySessionStore, JsonlSessionStore

__all__ = [
    # harness
    "AgentHarness",
    # actions
    "HarnessAction",
    "HarnessActionResult",
    "FunctionAction",
    # events
    "HarnessPhase",
    "HarnessSessionStart",
    "HarnessSessionSwitch",
    "HarnessSessionFork",
    "HarnessActionStart",
    "HarnessActionEnd",
    # stores
    "AgentSessionStore",
    "JsonlSessionStore",
    "InMemorySessionStore",
    # session models + utils
    "SessionEntry",
    "SessionEntryType",
    "SessionHeader",
    "SessionTreeNode",
    "SessionContext",
    "build_session_context",
    "coerce_messages",
    "generate_id",
    "get_latest_compaction_entry",
    "migrate_to_current_version",
    "CURRENT_SESSION_VERSION",
]
