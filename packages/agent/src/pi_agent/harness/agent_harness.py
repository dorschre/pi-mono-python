"""
AgentHarness — a reusable runtime around an Agent.

Owns session-tree persistence, branch/leaf navigation, context reconstruction, a
generic action registry, lifecycle events, and recovery/extension hook points. It
deliberately knows nothing about coding tools, cwd, or transports, so any agent can
reuse it (see the issue's four-layer model: Agent → AgentHarness → tool agent →
coding agent).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from pi_ai import get_model

from ..agent import Agent, AgentOptions
from ..types import AgentEvent
from .actions import FunctionAction, HarnessAction, HarnessActionResult
from .events import (
    HarnessActionEnd,
    HarnessActionStart,
    HarnessSessionFork,
    HarnessSessionStart,
    HarnessSessionSwitch,
)
from .session import SessionContext, coerce_messages
from .store import AgentSessionStore


def _default_serialize(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {"role": getattr(msg, "role", "unknown")}


class AgentHarness:
    """Reusable runtime wrapping an :class:`Agent` and an :class:`AgentSessionStore`."""

    def __init__(
        self,
        agent: Agent,
        session_store: AgentSessionStore,
        *,
        serialize_message: Callable[[Any], dict[str, Any]] | None = None,
        agent_factory: Callable[[], Agent] | None = None,
        register_builtin_actions: bool = True,
    ) -> None:
        self.agent = agent
        self.session_store = session_store
        self._serialize = serialize_message or _default_serialize
        self._agent_factory = agent_factory

        self.phase: str = "idle"
        self._listeners: list[Callable[[Any], None]] = []
        self._actions: dict[str, HarnessAction] = {}
        self._hooks: dict[str, list[Callable[..., Any]]] = {}

        self._unsubscribe_agent = self.agent.subscribe(self._on_agent_event)

        if register_builtin_actions:
            self._register_builtin_actions()

        self._emit(HarnessSessionStart(session_id=self._session_id()))

    # ── Event bus ─────────────────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[Any], None]) -> Callable[[], None]:
        self._listeners.append(fn)

        def unsub() -> None:
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsub

    def _emit(self, event: Any) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass

    def _on_agent_event(self, event: AgentEvent) -> None:
        # Persist messages as they finalize (generic — mirrors AgentSession._on_agent_event).
        if getattr(event, "type", None) == "message_end":
            msg = getattr(event, "message", None)
            if msg is not None and getattr(msg, "role", "") in ("user", "assistant", "toolResult"):
                self.session_store.append_message(self._serialize(msg))
        self._emit(event)

    # ── Hooks (extension points; no behavior implemented here) ─────────────────

    def register_hook(self, name: str, fn: Callable[..., Any]) -> Callable[[], None]:
        self._hooks.setdefault(name, []).append(fn)
        return lambda: self._hooks.get(name, []).remove(fn) if fn in self._hooks.get(name, []) else None

    async def run_hook(self, name: str, payload: Any = None) -> list[Any]:
        results: list[Any] = []
        for fn in list(self._hooks.get(name, [])):
            out = fn(self, payload)
            if hasattr(out, "__await__"):
                out = await out
            results.append(out)
        return results

    # ── Actions ───────────────────────────────────────────────────────────────

    def register_action(self, action: HarnessAction) -> None:
        self._actions[action.name] = action

    def get_actions(self) -> list[str]:
        return list(self._actions)

    async def run_action(self, name: str, args: dict[str, Any] | None = None) -> HarnessActionResult:
        args = args or {}
        action = self._actions.get(name)
        if action is None:
            raise ValueError(f"Unknown action: {name}")
        self._emit(HarnessActionStart(name=name, args=args))
        try:
            result = await action.run(self, args)
        except Exception as e:
            self._emit(HarnessActionEnd(name=name, error=str(e)))
            raise
        self._emit(HarnessActionEnd(name=name, result=result))
        return result

    def _register_builtin_actions(self) -> None:
        async def _fork(h: "AgentHarness", args: dict[str, Any]) -> HarnessActionResult:
            forked = await h.fork(args.get("entry_id"))
            return HarnessActionResult(
                text="forked",
                details={"session_id": forked._session_id()},
            )

        async def _switch_branch(h: "AgentHarness", args: dict[str, Any]) -> HarnessActionResult:
            ctx = await h.switch_branch(args["entry_id"])
            return HarnessActionResult(text="switched", details={"message_count": len(ctx.messages)})

        async def _resume(h: "AgentHarness", args: dict[str, Any]) -> HarnessActionResult:
            ctx = await h.resume()
            return HarnessActionResult(text="resumed", details={"message_count": len(ctx.messages)})

        self.register_action(FunctionAction("fork", _fork))
        self.register_action(FunctionAction("switch_branch", _switch_branch))
        self.register_action(FunctionAction("resume", _resume))

    # ── Context restore / navigation ───────────────────────────────────────────

    def _session_id(self) -> str:
        getter = getattr(self.session_store, "get_session_id", None)
        return getter() if callable(getter) else ""

    def _restore(self, leaf_id: str | None = None) -> SessionContext:
        """Rebuild context from the store and load it into the Agent."""
        context = self.session_store.build_context(leaf_id)
        self.agent.replace_messages(coerce_messages(context.messages))
        if context.model:
            model = get_model(context.model["provider"], context.model["model_id"])
            if model is not None:
                self.agent.set_model(model)
        if context.thinking_level:
            self.agent.set_thinking_level(context.thinking_level)  # type: ignore[arg-type]
        return context

    async def resume(self) -> SessionContext:
        """Rebuild context from the current leaf so the agent can continue."""
        self.agent.abort()
        await self.agent.wait_for_idle()
        return self._restore(self.session_store.get_leaf_id())

    async def switch_branch(self, entry_id: str) -> SessionContext:
        """Move the leaf to ``entry_id`` and reload context from that position."""
        self.agent.abort()
        await self.agent.wait_for_idle()
        self.session_store.set_leaf_id(entry_id)
        context = self._restore(entry_id)
        self._emit(HarnessSessionSwitch(session_id=self._session_id()))
        return context

    def _clone_agent(self) -> Agent:
        if self._agent_factory is not None:
            return self._agent_factory()
        src = self.agent
        opts = AgentOptions(
            convert_to_llm=getattr(src, "_convert_to_llm", None),
            transform_context=getattr(src, "_transform_context", None),
            stream_fn=getattr(src, "stream_fn", None),
            get_api_key=getattr(src, "get_api_key", None),
            state_system=getattr(src, "_state_system", None),
            on_payload=getattr(src, "_on_payload", None),
            thinking_budgets=getattr(src, "_thinking_budgets", None),
            transport=getattr(src, "_transport", "sse"),
            max_retry_delay_ms=getattr(src, "_max_retry_delay_ms", None),
        )
        clone = Agent(opts)
        if src.state.model:
            clone.set_model(src.state.model)
        clone.set_system_prompt(src.state.system_prompt)
        clone.set_tools(src.state.tools)
        clone.set_thinking_level(src.state.thinking_level)
        return clone

    async def fork(self, entry_id: str | None = None) -> "AgentHarness":
        """Fork into a new harness branched at ``entry_id`` (or the current leaf)."""
        self.agent.abort()
        await self.agent.wait_for_idle()

        branch_point = entry_id
        if not branch_point:
            leaf = self.session_store.get_leaf_id()
            branch_point = leaf

        branch_fn = getattr(self.session_store, "branch", None)
        if branch_fn is None:
            raise TypeError("session_store does not support branch()")

        # Mirror AgentSession.fork: if the branch point has a parent, branch at the
        # parent (dropping the leaf so it can be re-prompted); otherwise (root or no
        # branch point) full-copy the session — branch(None) copies all entries.
        target: str | None = None
        if branch_point is not None:
            entry = self.session_store.get_entry(branch_point)
            if entry is not None and entry.parent_id:
                target = entry.parent_id

        forked_store = branch_fn(target)
        forked_agent = self._clone_agent()
        forked = AgentHarness(
            forked_agent,
            forked_store,
            serialize_message=self._serialize,
            agent_factory=self._agent_factory,
        )
        forked._restore(forked_store.get_leaf_id())
        self._emit(HarnessSessionFork(
            source_session_id=self._session_id(),
            new_session_id=forked._session_id(),
            entry_id=entry_id,
        ))
        return forked

    # ── Pass-through conveniences ───────────────────────────────────────────────

    async def prompt(self, *args: Any, **kwargs: Any) -> None:
        self.phase = "turn"
        try:
            await self.agent.prompt(*args, **kwargs)
        finally:
            self.phase = "idle"

    def append_message(self, message: Any) -> None:
        self.agent.append_message(message)

    def build_context(self, leaf_id: str | None = None) -> SessionContext:
        return self.session_store.build_context(leaf_id)
