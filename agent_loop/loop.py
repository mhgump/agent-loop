import json
import re
import traceback
from typing import Any, Callable, Optional

from .config import AgentLoopConfig
from .task import Task
from .tool import Tool


def _parse_metadata(text: str) -> dict:
    """Parse {progress, success} JSON from an LLM response string."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"progress": text, "success": False}


def _content_to_dicts(content) -> list[dict]:
    """Convert API response content blocks to plain dicts for reuse in messages."""
    result = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
        elif block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return result


class AgentLoop:
    """Orchestrates progress/metadata LLM calls around a set of tools and a task."""

    def __init__(self, task: Task, tools: list[Tool], config: AgentLoopConfig):
        self.task = task
        self.tools = {t.name: t for t in tools}
        self.config = config
        self._last_turn_tool_errors: list[dict] = []
        self._produce_artifact_error: Optional[str] = None

    def _progress_step(self, messages: list[dict]) -> Any:
        return self.config.client.messages.create(
            model=self.config.model_progress,
            max_tokens=4096,
            system=self.task.PROGRESS_SYSTEM_PROMPT,
            tools=[t.to_api_def() for t in self.tools.values()],
            messages=messages,
        )

    def _metadata_step(self, artifact) -> dict:
        if artifact is not None:
            try:
                artifact_str = json.dumps(artifact, indent=2)
            except (TypeError, ValueError):
                artifact_str = str(artifact)
        else:
            artifact_str = "No artifact produced."

        user_msg = (
            f"Task: {self.task.get_prompt()}\n\n"
            f"Stopping criteria: {self.task.STOPPING_CRITERIA}\n\n"
            f"Current artifact:\n{artifact_str}"
        )
        response = self.config.client.messages.create(
            model=self.config.model_metadata,
            max_tokens=1024,
            system=self.task.METADATA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        return _parse_metadata(text)

    def _execute_tools(self, tool_uses) -> tuple[list[dict], list[dict]]:
        """Run all tool_use blocks. Returns (api_results_for_messages, return_values)."""
        api_results = []
        return_values = []
        for tu in tool_uses:
            name = tu.name if hasattr(tu, "name") else tu.get("name", "")
            tool_id = tu.id if hasattr(tu, "id") else tu.get("id", "")
            inputs = tu.input if hasattr(tu, "input") else tu.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {}

            if name not in self.tools:
                result = {"errors": f"Unknown tool: {name}"}
            else:
                try:
                    result = self.tools[name].call(inputs)
                except Exception as exc:
                    result = {"errors": str(exc)}

            return_values.append(result)
            api_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result),
            })
        return api_results, return_values

    def run(
        self,
        on_progress: Optional[Callable] = None,
        on_metadata: Optional[Callable] = None,
        on_attempt: Optional[Callable] = None,
    ):
        """Execute the agent loop.

        Args:
            on_progress: Called after each progress step with a list of tool call specs
                         [{"name": ..., "input": ...}].
            on_metadata: Called after each metadata step with the {progress, success} dict.
            on_attempt:  Called after each attempt with (success: bool, artifact).

        Returns:
            The final artifact produced by the task, or None if all attempts failed.
        """
        self._last_turn_tool_errors = []
        self._produce_artifact_error = None

        for _attempt in range(self.config.max_attempts):
            messages = [{"role": "user", "content": self.task.get_prompt()}]
            all_tool_results: list[dict] = []
            artifact = None
            attempt_success = False

            for _turn in range(self.config.max_turns):
                # Progress step
                progress_resp = self._progress_step(messages)

                tool_uses = [
                    b for b in progress_resp.content
                    if (hasattr(b, "type") and b.type == "tool_use")
                    or (isinstance(b, dict) and b.get("type") == "tool_use")
                ]

                if on_progress is not None:
                    on_progress([
                        {
                            "name": tu.name if hasattr(tu, "name") else tu["name"],
                            "input": tu.input if hasattr(tu, "input") else tu["input"],
                        }
                        for tu in tool_uses
                    ])

                messages.append({
                    "role": "assistant",
                    "content": _content_to_dicts(progress_resp.content),
                })

                # Execute tools
                api_results, return_values = self._execute_tools(tool_uses)
                all_tool_results.extend(return_values)
                self._last_turn_tool_errors = [
                    r for r in return_values if r.get("errors") is not None
                ]

                # Produce artifact
                try:
                    artifact = self.task.produce_artifact(
                        self.task.prompt_fields, list(all_tool_results)
                    )
                    self._produce_artifact_error = None
                except Exception:
                    self._produce_artifact_error = traceback.format_exc()
                    artifact = None

                # Metadata step
                metadata = self._metadata_step(artifact)

                if on_metadata is not None:
                    on_metadata(metadata)

                # Append user message with tool results and evaluation context
                if tool_uses:
                    user_content: Any = list(api_results)
                    if artifact is not None:
                        user_content.append({
                            "type": "text",
                            "text": f"Task artifact: {json.dumps(artifact)}",
                        })
                    user_content.append({
                        "type": "text",
                        "text": f"Evaluation: {json.dumps(metadata)}",
                    })
                else:
                    user_content = f"Evaluation: {json.dumps(metadata)}"

                messages.append({"role": "user", "content": user_content})

                if metadata.get("success", False):
                    attempt_success = True
                    break

            # Evaluate attempt outcome
            has_artifact = artifact is not None
            all_no_errors = all(r.get("errors") is None for r in all_tool_results)

            if not has_artifact:
                if on_attempt is not None:
                    on_attempt(False, None)
                continue

            if attempt_success:
                if on_attempt is not None:
                    on_attempt(True, artifact)
                self.task.side_effects(artifact)
                return artifact

            if self.config.accept_attempts_if_no_error and all_no_errors:
                if on_attempt is not None:
                    on_attempt(True, artifact)
                self.task.side_effects(artifact)
                return artifact

            if on_attempt is not None:
                on_attempt(False, artifact)

        return None

    def last_turn_tool_errors(self) -> list[dict]:
        """Return tool call results that contained errors from the most recent turn.

        Each entry is a dict with at least an ``"errors"`` key containing the error
        string. Returns an empty list if the last turn had no errors or ``run()``
        has not been called yet.
        """
        return list(self._last_turn_tool_errors)

    def produce_artifact_error(self) -> Optional[str]:
        """Return the traceback string from the most recent ``produce_artifact`` failure.

        Returns ``None`` if the last call to ``produce_artifact`` succeeded or
        ``run()`` has not been called yet.
        """
        return self._produce_artifact_error
