"""
State-transition adapter layer for the agent loop.

This module is a **Python-side extension** with no TypeScript counterpart yet. It
introduces a pluggable seam — `StateTransitionSystem` — that owns the per-run
working-state message list the agent loop reads from (`materialize()`) and writes to
(`apply()`). Different transition systems (append-only today; event-sourcing,
belief-revision, checkpoint, etc. later) can be injected via a factory passed through
`AgentLoopConfig` / `AgentOptions` without forking `agent_loop`.

The default `AppendOnlyStateSystem` reproduces the loop's historical behavior exactly,
so routing through this layer is behaviorally identical to the pre-existing code and the
"mirrors TypeScript" contract is preserved at the observable level.

Each state change is recorded as a lightweight first-class `Transition` (stable id,
from/to version, type, actor, timestamp, evidence pointers) — enough to support audit and
event-sourcing adapters without pulling in a full PROV/RDF stack.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from .types import AgentMessage

# ─── Transition vocabulary ────────────────────────────────────────────────────

TransitionType = Literal["prompt", "steering", "assistant", "tool_result", "follow_up"]


@dataclass
class Observation:
    """Input to a single state transition.

    `type` classifies the observation, `actor` records who produced it
    ("user" | "assistant" | "tool" | "system"), `message` is the message added to the
    working state (if any), `evidence` carries pointers such as tool_call_ids this
    observation derives from, and `metadata` is a free-form bag for adapters.
    """
    type: TransitionType
    actor: str
    message: AgentMessage | None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """A lightweight, first-class record of one applied state change."""
    id: str
    type: TransitionType
    actor: str
    from_version: int
    to_version: int
    timestamp: int  # ms epoch
    message: AgentMessage | None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── StateTransitionSystem protocol ─────────────────────────────────────────────


@runtime_checkable
class StateTransitionSystem(Protocol):
    """Pluggable authority over the agent loop's working-state message list."""

    def seed(self, messages: list[AgentMessage]) -> None:
        """Install prior context. Seeded messages are NOT counted as run-delta."""
        ...

    def materialize(self) -> list[AgentMessage]:
        """Return the working message list to feed the model this step."""
        ...

    def apply(self, observation: Observation) -> Transition:
        """Evolve the state with one observation and return the transition record."""
        ...

    def new_messages(self) -> list[AgentMessage]:
        """Return the messages added during this run (for the agent_end result)."""
        ...

    def transitions(self) -> list[Transition]:
        """Return the append-only log of transitions applied this run."""
        ...

    @property
    def version(self) -> int:
        """Current state version (monotonically increasing per applied transition)."""
        ...


StateSystemFactory = Callable[[], StateTransitionSystem]


# ─── Default adapter ─────────────────────────────────────────────────────────


class AppendOnlyStateSystem:
    """Default transition system — append-only accumulation.

    Reproduces the agent loop's historical semantics exactly: `seed` stores the prior
    context, `apply` appends the observation's message and records a transition,
    `materialize` returns seed + applied messages, and `new_messages` returns only the
    messages applied during this run.
    """

    def __init__(self) -> None:
        self._seed: list[AgentMessage] = []
        self._applied: list[AgentMessage] = []
        self._transitions: list[Transition] = []
        self._version: int = 0

    def seed(self, messages: list[AgentMessage]) -> None:
        self._seed = list(messages)

    def materialize(self) -> list[AgentMessage]:
        return [*self._seed, *self._applied]

    def apply(self, observation: Observation) -> Transition:
        from_version = self._version
        to_version = from_version + 1
        self._version = to_version
        if observation.message is not None:
            self._applied.append(observation.message)
        transition = Transition(
            id=str(uuid.uuid4()),
            type=observation.type,
            actor=observation.actor,
            from_version=from_version,
            to_version=to_version,
            timestamp=int(time.time() * 1000),
            message=observation.message,
            evidence=list(observation.evidence),
            metadata=dict(observation.metadata),
        )
        self._transitions.append(transition)
        return transition

    def new_messages(self) -> list[AgentMessage]:
        return list(self._applied)

    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    @property
    def version(self) -> int:
        return self._version
