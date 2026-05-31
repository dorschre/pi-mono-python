"""
pi_agent — Agent loop and state management
Python mirror of @mariozechner/pi-agent-core
"""

from .agent import Agent, AgentOptions
from .agent_loop import agent_loop, agent_loop_continue
from .proxy import stream_proxy
from .state import (
    AppendOnlyStateSystem,
    Observation,
    StateSystemFactory,
    StateTransitionSystem,
    Transition,
    TransitionType,
)
from .types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventToolEnd,
    AgentEventToolStart,
    AgentEventToolUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CustomAgentMessages,
    StreamFn,
    ThinkingLevel,
)

__all__ = [
    # Agent class
    "Agent",
    "AgentOptions",
    # Loop functions
    "agent_loop",
    "agent_loop_continue",
    # Proxy
    "stream_proxy",
    # State-transition adapter layer
    "AppendOnlyStateSystem",
    "Observation",
    "StateSystemFactory",
    "StateTransitionSystem",
    "Transition",
    "TransitionType",
    # Types
    "AgentContext",
    "AgentEvent",
    "AgentEventAgentEnd",
    "AgentEventAgentStart",
    "AgentEventMessageEnd",
    "AgentEventMessageStart",
    "AgentEventMessageUpdate",
    "AgentEventToolEnd",
    "AgentEventToolStart",
    "AgentEventToolUpdate",
    "AgentEventTurnEnd",
    "AgentEventTurnStart",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "CustomAgentMessages",
    "StreamFn",
    "ThinkingLevel",
]
