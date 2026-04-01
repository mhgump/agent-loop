# Agent Loop Framework

A Python framework for building reliable AI agent loops. The framework alternates between a **progress step** (the LLM calls tools to make progress) and a **metadata step** (a second LLM call evaluates whether the task is complete), retrying across multiple attempts until success or exhaustion.

## Installation

```bash
pip install -e ".[dev]"
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The framework automatically loads this file when using the default client.

---

## Core Concepts

| Concept | Role |
|---|---|
| `Tool` | A capability the LLM can call. Returns JSON with an `errors` field. |
| `Task` | Defines the goal: prompt template, stopping criteria, system prompts, and artifact production logic. |
| `AgentLoopConfig` | Configures models, retry limits, and acceptance policy. |
| `AgentLoop` | Orchestrates the loop: progress → tools → artifact → metadata → repeat. |

---

## Defining a Tool

Subclass `Tool`, set a class-level `spec` dict (Anthropic tool format), and implement `call()`.

```python
from agent_loop import Tool

class GetWeatherTool(Tool):
    spec = {
        "name": "get_weather",
        "description": "Get current weather for a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }

    def call(self, location: str, unit: str = "celsius") -> dict:
        # Your implementation here
        data = fetch_weather_api(location, unit)
        return {
            "temperature": data["temp"],
            "condition": data["condition"],
            "errors": None,           # None = success
        }
```

### The `errors` field contract

Every `call()` return value **must** contain an `errors` field:

- `"errors": None` — tool succeeded
- `"errors": "description of what went wrong"` — tool failed

The framework uses this field to evaluate whether `accept_attempts_if_no_error` applies.

---

## Defining a Task

Subclass `Task` and define four class-level string constants plus `produce_artifact()`.

```python
from agent_loop import Task

class WeatherReportTask(Task):
    PROMPT_TEMPLATE = (
        "Generate a weather comparison report for {city_a} and {city_b}. "
        "Use the get_weather tool to fetch current conditions for each city."
    )

    STOPPING_CRITERIA = (
        "A complete report has been generated containing weather data for both cities."
    )

    PROGRESS_SYSTEM_PROMPT = (
        "You are a helpful assistant. Use the available tools to gather data "
        "needed to complete the user's task."
    )

    METADATA_SYSTEM_PROMPT = (
        "You evaluate whether a task has been completed. "
        "Respond with JSON: {\"progress\": \"<what has been done>\", \"success\": true/false}."
    )

    def produce_artifact(self, tool_results: list[dict]):
        """Build the final artifact from all tool results collected this attempt.

        Raise an exception if the artifact cannot be produced — the framework
        will treat this as a failed attempt and retry.
        """
        weather_data = [r for r in tool_results if r.get("errors") is None]
        if len(weather_data) < 2:
            raise ValueError("Need weather data for at least two cities")
        return {
            "report": f"Comparison: {weather_data[0]} vs {weather_data[1]}",
            "cities_compared": len(weather_data),
        }
```

### Instantiating a Task

Pass a dict of values to fill the `PROMPT_TEMPLATE` placeholders:

```python
task = WeatherReportTask({"city_a": "San Francisco, CA", "city_b": "New York, NY"})
```

---

## Configuring the Agent Loop

```python
from agent_loop import AgentLoopConfig

config = AgentLoopConfig(
    # client defaults to Anthropic() loaded from ANTHROPIC_API_KEY in .env
    model_progress="claude-haiku-4-5-20251001",   # model for progress (tool-calling) steps
    model_metadata="claude-haiku-4-5-20251001",   # model for metadata (evaluation) steps
    max_attempts=3,                                # outer retry limit
    max_turns=10,                                  # max progress/metadata cycles per attempt
    accept_attempts_if_no_error=False,             # see below
)
```

### `accept_attempts_if_no_error`

When `True`, an attempt that exhausts `max_turns` is still accepted if:
- All tool calls returned `"errors": None`
- The task's `produce_artifact()` did not raise

This lets you accept partial completion when the LLM ran out of turns but produced clean results.

---

## Running the Agent Loop

```python
from agent_loop import AgentLoop

loop = AgentLoop(task=task, tools=[GetWeatherTool()], config=config)
artifact = loop.run()

if artifact is not None:
    print("Task completed:", artifact)
else:
    print("All attempts exhausted without success")
```

### Return value

`run()` returns the artifact produced by `Task.produce_artifact()`, or `None` if every attempt failed.

---

## Callbacks

`run()` accepts three optional callbacks for observability:

```python
def on_progress(tool_call_specs: list[dict]):
    """Called after each progress step. Receives tool calls the LLM made."""
    for spec in tool_call_specs:
        print(f"  Tool: {spec['name']}({spec['input']})")

def on_metadata(metadata: dict):
    """Called after each metadata step. Receives {progress, success}."""
    print(f"  Evaluation: {metadata['progress']}")
    print(f"  Success: {metadata['success']}")

def on_attempt(success: bool, artifact):
    """Called after each attempt completes."""
    status = "succeeded" if success else "failed"
    print(f"Attempt {status}: {artifact}")

artifact = loop.run(
    on_progress=on_progress,
    on_metadata=on_metadata,
    on_attempt=on_attempt,
)
```

---

## How the Loop Works

```
for each attempt (up to max_attempts):
    for each turn (up to max_turns):
        1. Progress step  — LLM receives task prompt + conversation history,
                            returns a set of tool calls
        2. Execute tools  — all tool calls are run; results collected
        3. Produce artifact — Task.produce_artifact(all_tool_results) is called
        4. Metadata step  — separate LLM call evaluates {progress, success}
                            given the task, stopping criteria, and current artifact
        5. If metadata returns success: true → accept attempt, return artifact

    After max_turns exhausted (without success):
        - if Task raised → attempt fails (no artifact)
        - if accept_attempts_if_no_error and all tools error-free → accept
        - otherwise → attempt fails, retry outer loop

return artifact, or None if all attempts failed
```

The conversation history for progress steps accumulates turn by turn within an attempt:
each turn appends the LLM's tool calls (assistant message) and the tool results plus
evaluation context (user message). A new attempt starts with a fresh conversation.

### Metadata step response format

The metadata LLM must return JSON in this shape:

```json
{
  "progress": "Description of what has been accomplished so far.",
  "success": true
}
```

Set `"success": false` to indicate the task is not yet complete.

---

## Complete Example

```python
from agent_loop import AgentLoop, AgentLoopConfig, Task, Tool


class SearchTool(Tool):
    spec = {
        "name": "search",
        "description": "Search for information on a topic",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }

    def call(self, query: str) -> dict:
        # Replace with a real search implementation
        return {"results": [f"Result for: {query}"], "errors": None}


class ResearchTask(Task):
    PROMPT_TEMPLATE = "Research the topic: {topic}. Summarise your findings."
    STOPPING_CRITERIA = "A summary of findings has been produced."
    PROGRESS_SYSTEM_PROMPT = "You are a research assistant. Use the search tool to gather information."
    METADATA_SYSTEM_PROMPT = (
        'Evaluate whether the research task is complete. '
        'Respond with JSON only: {"progress": "...", "success": true/false}'
    )

    def produce_artifact(self, tool_results: list[dict]) -> dict:
        results = [r["results"] for r in tool_results if r.get("errors") is None]
        if not results:
            raise ValueError("No search results collected")
        return {"summary": results, "sources_checked": len(results)}


if __name__ == "__main__":
    task = ResearchTask({"topic": "quantum computing"})
    config = AgentLoopConfig(max_attempts=3, max_turns=5)

    artifact = AgentLoop(task=task, tools=[SearchTool()], config=config).run(
        on_metadata=lambda m: print(f"[eval] {m['progress']} | done={m['success']}"),
    )

    if artifact:
        print("Research complete:", artifact)
    else:
        print("Research did not complete in the allotted attempts.")
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Test journeys

| File | Journey |
|---|---|
| `test_journey_success.py` | Task succeeds and metadata returns `success: true` → artifact returned |
| `test_journey_metadata_fails.py` | Metadata always returns `false` → `None` returned after all attempts |
| `test_journey_task_fails.py` | `produce_artifact` always raises → `None` returned despite metadata success |
| `test_journey_accept_no_error.py` | Metadata always `false` but flag set + no errors → artifact accepted |

### Available mocks (in `tests/conftest.py`)

| Mock | Behaviour |
|---|---|
| `MockMetadataResponse(bools)` | Returns `success` values from the provided list in order |
| `MockProgressResponse(tool_name)` | Always returns one tool call for `tool_name` with sequential inputs |
| `MockTask` | `produce_artifact` returns `{"status": "complete", "tool_count": N}` |
| `MockFailingTask` | `produce_artifact` always raises `RuntimeError` |
| `MockTool` | `call()` always returns `{"result": "executed", "errors": None}` |
| `MockClient` | Routes calls to `MockProgressResponse` (when tools present) or `MockMetadataResponse` |
