"""
Generic harness actions.

An action is a named, awaitable unit of work over a harness — the unit that
product transports (CLI/TUI/RPC) call via ``harness.run_action(name, args)``
instead of re-defining the behavior per transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .agent_harness import AgentHarness


@dataclass
class HarnessActionResult:
    text: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HarnessAction(Protocol):
    name: str

    async def run(self, harness: "AgentHarness", args: dict[str, Any]) -> HarnessActionResult: ...


@dataclass
class FunctionAction:
    """Adapter wrapping an async function as a HarnessAction."""
    name: str
    fn: Callable[["AgentHarness", dict[str, Any]], Awaitable[HarnessActionResult]]

    async def run(self, harness: "AgentHarness", args: dict[str, Any]) -> HarnessActionResult:
        return await self.fn(harness, args)
