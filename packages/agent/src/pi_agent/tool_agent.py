"""
GenericToolCallingAgent — a harness-backed agent with a generic tool registry.

Sits between AgentHarness and product-specific agents (e.g. the coding agent). It
owns a tool registry and active-tool selection and exposes the harness's session /
action / branching surface — without any CLI/TUI/RPC or coding-tool dependency.
"""
from __future__ import annotations

from typing import Any, Callable

from .agent import Agent, AgentOptions
from .harness import AgentHarness, AgentSessionStore, InMemorySessionStore, HarnessAction, HarnessActionResult
from .types import AgentTool, ThinkingLevel


class GenericToolCallingAgent:
    """A reusable tool-calling agent built on :class:`AgentHarness`."""

    def __init__(self, harness: AgentHarness) -> None:
        self.harness = harness
        self._tools: dict[str, AgentTool] = {}
        self._active: list[str] = []
        # Seed registry from whatever the wrapped agent already has.
        for tool in harness.agent.state.tools or []:
            self._tools[tool.name] = tool
            self._active.append(tool.name)

    # ── Convenience constructor ─────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        model: Any | None = None,
        store: AgentSessionStore | None = None,
        tools: list[AgentTool] | None = None,
        stream_fn: Any | None = None,
        convert_to_llm: Callable | None = None,
        get_api_key: Callable | None = None,
        system_prompt: str = "",
        thinking_level: ThinkingLevel = "off",
    ) -> "GenericToolCallingAgent":
        opts = AgentOptions(
            stream_fn=stream_fn,
            convert_to_llm=convert_to_llm,
            get_api_key=get_api_key,
        )
        agent = Agent(opts)
        if model is not None:
            agent.set_model(model)
        agent.set_system_prompt(system_prompt)
        agent.set_thinking_level(thinking_level)
        if tools:
            agent.set_tools(list(tools))
        harness = AgentHarness(agent, store or InMemorySessionStore())
        return cls(harness)

    # ── Tool registry ────────────────────────────────────────────────────────────

    def set_tools(self, tools: list[AgentTool], active_tool_names: list[str] | None = None) -> None:
        self._tools = {t.name: t for t in tools}
        self._active = list(active_tool_names) if active_tool_names is not None else list(self._tools)
        self._apply_active_tools()

    def get_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        return list(self._tools)

    def get_active_tools(self) -> list[str]:
        return list(self._active)

    def set_active_tools(self, names: list[str]) -> None:
        self._active = [n for n in names if n in self._tools]
        self._apply_active_tools()

    def _apply_active_tools(self) -> None:
        self.harness.agent.set_tools([self._tools[n] for n in self._active if n in self._tools])

    # ── Harness pass-through ──────────────────────────────────────────────────────

    @property
    def agent(self) -> Agent:
        return self.harness.agent

    @property
    def session_store(self) -> AgentSessionStore:
        return self.harness.session_store

    def subscribe(self, fn: Callable[[Any], None]) -> Callable[[], None]:
        return self.harness.subscribe(fn)

    def register_action(self, action: HarnessAction) -> None:
        self.harness.register_action(action)

    async def run_action(self, name: str, args: dict[str, Any] | None = None) -> HarnessActionResult:
        return await self.harness.run_action(name, args)

    async def prompt(self, *args: Any, **kwargs: Any) -> None:
        await self.harness.prompt(*args, **kwargs)

    async def fork(self, entry_id: str | None = None) -> "GenericToolCallingAgent":
        forked_harness = await self.harness.fork(entry_id)
        forked = GenericToolCallingAgent(forked_harness)
        forked.set_tools(self.get_tools(), self.get_active_tools())
        return forked

    async def switch_branch(self, entry_id: str):
        return await self.harness.switch_branch(entry_id)

    async def resume(self):
        return await self.harness.resume()
