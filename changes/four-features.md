# Four feature additions

## 1. Extended metadata response: `failure` and `failure_reason`

**Intent:** The original metadata response only had `success` (the task is done) with no way to
distinguish "not done yet, keep going" from "hopelessly stuck, stop retrying." Adding `failure`
and `failure_reason` lets the metadata LLM signal an unrecoverable state so the loop terminates
immediately instead of burning through all remaining attempts.

**Diffs:**

`_parse_metadata` in `loop.py` — always normalises the four standard fields with safe defaults,
and spreads any extra task-defined keys through unchanged:
```python
return {
    "progress": parsed.get("progress", text if not parsed else ""),
    "success": bool(parsed.get("success", False)),
    "failure": bool(parsed.get("failure", False)),
    "failure_reason": parsed.get("failure_reason", ""),
    **{k: v for k, v in parsed.items() if k not in (...)},
}
```

`_metadata_step` in `loop.py` — injects field documentation into every metadata prompt so the
LLM knows to return them:
```
  - failure (boolean): True when the agent is stuck in an unrecoverable or repeatedly bad state...
  - failure_reason (string): Human-readable explanation... Empty string when failure is false.
```

`run()` in `loop.py` — checks `failure` before `success` each turn; if true, sets
`terminal_failure = True`, breaks the turn loop, then breaks the attempt loop entirely:
```python
if metadata.get("failure", False):
    terminal_failure = True
    break
...
if terminal_failure:
    if on_attempt is not None:
        on_attempt(False, artifact)
    break
```

`conftest.py` — `MockMetadataResponse` updated to include the new fields in its JSON output.

---

## 2. Task-defined metadata fields (`METADATA_FIELDS`)

**Intent:** Different tasks need to track different progress dimensions. The gnome explorer needs
separate `explore_progress` and `extract_progress` fields; a skill-writer needs `skill_progress`.
`METADATA_FIELDS` lets each task declare its own named fields, which are injected into the
metadata prompt alongside the standard ones and returned in the metadata dict.

**Diffs:**

`Task` in `task.py` — new class attribute with schema:
```python
METADATA_FIELDS: list[dict] = []
# Each entry: {"name": str, "description": str, "type": str}
```

`_metadata_step` in `loop.py` — reads `self.task.METADATA_FIELDS`, appends them to the
standard field documentation block, and includes all of them in the metadata user message:
```python
extra_fields = [
    (f['name'], f.get('type', 'string'), f['description'])
    for f in (self.task.METADATA_FIELDS or [])
]
all_fields = standard_fields + extra_fields
field_docs = "\n".join(f"  - {name} ({typ}): {desc}" for name, typ, desc in all_fields)
```

Because `_parse_metadata` already spreads unknown keys through, no further parsing changes
were needed — the extra fields come back in the metadata dict automatically.

---

## 3. Context objects passed through the loop

**Intent:** Tools and tasks often need access to runtime objects that aren't part of the task
prompt — server handles, bot IDs, HTTP endpoints, file paths. Adding a `context` dict to
`AgentLoop` lets the caller inject these at construction time, and threads them through to
every tool call and task method, without requiring callers to subclass or monkey-patch.

**Diffs:**

`AgentLoop.__init__` in `loop.py`:
```python
def __init__(self, task, tools, config, context: dict | None = None):
    ...
    self.context = context
```

`Tool.call` signature in `tool.py`:
```python
def call(self, input: dict, context: dict | None = None) -> dict:
```

`_execute_tools` in `loop.py` — passes context on every tool invocation:
```python
result = self.tools[name].call(inputs, self.context)
```

`Task.produce_artifact` and `Task.side_effects` in `task.py` — both gain `context` as an
optional parameter so tasks can use runtime objects during artifact production and side effects:
```python
def produce_artifact(self, task_inputs, tool_results, context: dict | None = None): ...
def side_effects(self, artifact, context: dict | None = None): ...
```

`run()` in `loop.py` — passes context to both:
```python
artifact = self.task.produce_artifact(..., self.context)
self.task.side_effects(artifact, self.context)
```

`conftest.py` — `MockTool.call`, `MockTask.produce_artifact`, `MockFailingTask.produce_artifact`
all updated to accept the new optional parameter.

---

## 4. Between-turns hook on `Task`

**Intent:** Some tasks need to perform work between each agent turn — for example, resetting a
game scenario to its initial state so the agent always sees a clean starting point when testing
a skill script. The `between_turns` hook makes this a first-class concept on the task rather
than requiring callers to wrap the loop externally.

**Diffs:**

`Task.between_turns` in `task.py` — new method with a no-op default:
```python
def between_turns(self, turn_number: int, tool_results: list[dict], context: dict | None = None) -> None:
    """Called after each turn's tools execute, before produce_artifact."""
```

`run()` in `loop.py` — called after `_execute_tools`, before `produce_artifact`, so the hook
runs in the correct position (world is mutated by tool, then reset, then artifact is assessed):
```python
# Between-turn hook — e.g. reset scenario state
self.task.between_turns(_turn + 1, list(all_tool_results), self.context)
```

The 1-based `turn_number` is passed so tasks can make first-turn vs subsequent-turn decisions
(e.g. skip the reset on turn 1 since the scenario is already fresh).
