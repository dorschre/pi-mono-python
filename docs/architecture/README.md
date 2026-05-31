# Architecture Documentation — pi-mono-python

A [C4 model](https://c4model.com/) of the `pi` LM-agent harness, plus UML **sequence** and
**activity** diagrams of the runtime behavior. All diagrams are **PlantUML source** (`.puml`);
no images are committed — render them yourself (see [Rendering](#rendering)).

The harness is a uv workspace of four packages:

| Package | Role |
|---|---|
| `pi_ai` | Unified LLM streaming layer (provider registry + adapters, models, OAuth, event stream) |
| `pi_agent` | Agent loop, `Agent` wrapper, pluggable `StateTransitionSystem`, agent types/events |
| `pi_coding_agent` | CLI app: modes, `AgentSession`, `SessionManager`, tools, compaction, extensions, settings/model/auth |
| `pi_tui` | Terminal UI: differential renderer, editor, components, autocomplete, key bindings |

## C4 model

Read top-down: **Context → Container → Component**.

| Level | Diagram | What it shows |
|---|---|---|
| L1 Context | [`c4/01-context.puml`](c4/01-context.puml) | The harness, the developer, and external systems (LLM APIs, OAuth, filesystem, session/settings store) |
| L2 Container | [`c4/02-container.puml`](c4/02-container.puml) | The four packages, their dependencies, and the JSONL/JSON data stores |
| L3 Component | [`c4/03-component-pi_agent.puml`](c4/03-component-pi_agent.puml) | Inside `pi_agent`: `Agent`, `agent_loop`/`_run_loop`, `_stream_assistant_response`, `_execute_tool_calls`, `StateTransitionSystem`, types |
| L3 Component | [`c4/03-component-pi_ai.puml`](c4/03-component-pi_ai.puml) | Inside `pi_ai`: `stream_simple`, `api_registry`, provider adapters, models, OAuth, `EventStream` |
| L3 Component | [`c4/03-component-pi_coding_agent.puml`](c4/03-component-pi_coding_agent.puml) | Inside `pi_coding_agent`: CLI/modes, `AgentSession`, `SessionManager`, tools, compaction, extensions, settings/model/auth |
| L3 Component | [`c4/03-component-pi_tui.puml`](c4/03-component-pi_tui.puml) | Inside `pi_tui`: runtime, renderer, input pipeline, editor, components |

The C4 macro library is vendored under [`c4/_standard/`](c4/_standard/) so the diagrams render
fully offline (sourced from [plantuml-stdlib/C4-PlantUML](https://github.com/plantuml-stdlib/C4-PlantUML)).

## Sequence diagrams (lowest-level dynamic views)

| Diagram | Flow |
|---|---|
| [`sequence/seq-prompt-turn.puml`](sequence/seq-prompt-turn.puml) | One prompt turn end to end: mode → `AgentSession` → `Agent` → `agent_loop` → `stream_simple` → provider → tools → state → persistence |
| [`sequence/seq-tool-execution.puml`](sequence/seq-tool-execution.puml) | `_execute_tool_calls`: validation, `AgentTool.execute`, updates, mid-run steering, skipped calls |
| [`sequence/seq-state-transition.puml`](sequence/seq-state-transition.puml) | The pluggable `StateTransitionSystem`: `seed` / `materialize` / `apply` / `new_messages` / `transitions` |
| [`sequence/seq-session-restore.puml`](sequence/seq-session-restore.puml) | `switch_session` → `build_session_context` tree walk → rehydrate `Agent` |
| [`sequence/seq-compaction.puml`](sequence/seq-compaction.puml) | Post-turn retry-vs-compaction; overflow detection → `compact_context` → new context |

## Activity diagrams (lowest-level control flow)

| Diagram | Logic |
|---|---|
| [`activity/act-agent-loop.puml`](activity/act-agent-loop.puml) | `_run_loop`: turn loop, tool-call branch, steering, follow-up gate |
| [`activity/act-stream-assistant.puml`](activity/act-stream-assistant.puml) | `_stream_assistant_response`: streaming event switch + partial/abort fallback |
| [`activity/act-post-turn-checks.puml`](activity/act-post-turn-checks.puml) | `_post_turn_checks`: retry precedence + overflow-guarded compaction |
| [`activity/act-build-session-context.puml`](activity/act-build-session-context.puml) | `build_session_context`: leaf→root walk + compaction-boundary reconstruction |

Diagram node/participant labels carry the real symbol and source file so each diagram stays
traceable to the code.

## Rendering

Diagrams use the PlantUML toolchain; Graphviz (`dot`) is required for C4 layouts.

```bash
# Render everything to SVG (requires PlantUML 1.2020+ on Java 11+ and Graphviz)
java -jar plantuml.jar -tsvg -o rendered docs/architecture/**/*.puml

# Single diagram to PNG
java -jar plantuml.jar -tpng docs/architecture/c4/02-container.puml
```

IDE plugins (VS Code "PlantUML", JetBrains "PlantUML integration") render `.puml` on the fly.
The C4 includes are local (`c4/_standard/`), so no network access is needed.
