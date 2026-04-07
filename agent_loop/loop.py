import json
import os
import re
import traceback
from typing import Any, Callable, Optional

from .config import AgentLoopConfig
from .task import Task
from .tool import Tool


def _parse_metadata(text: str) -> dict:
    """Parse metadata JSON from an LLM response string.

    Guarantees the returned dict always contains the standard fields:
      - progress (str): description of overall progress so far.
      - success (bool): True when the task is complete.
      - failure (bool): True when the agent is in an unrecoverable / repeatedly bad state.
      - failure_reason (str): Human-readable explanation for failure (empty when not failing).
    Any additional task-defined fields present in the JSON are preserved as-is.
    """
    parsed: dict = {}
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {
        "progress": parsed.get("progress", text if not parsed else ""),
        "success": bool(parsed.get("success", False)),
        "failure": bool(parsed.get("failure", False)),
        "failure_reason": parsed.get("failure_reason", ""),
        **{k: v for k, v in parsed.items() if k not in ("progress", "success", "failure", "failure_reason")},
    }


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

    def __init__(self, task: Task, tools: list[Tool], config: AgentLoopConfig, context: dict | None = None):
        self.task = task
        self.tools = {t.name: t for t in tools}
        self.config = config
        self.context = context
        self._last_turn_tool_errors: list[dict] = []
        self._produce_artifact_error: Optional[str] = None
        self._last_metadata: Optional[dict] = None

    def _write_turn_log(self, turn: int, step: str, direction: str, content: str) -> None:
        if not self.config.log_dir:
            return
        turns_dir = os.path.join(self.config.log_dir, "turns")
        os.makedirs(turns_dir, exist_ok=True)
        filename = f"{turn:02d}_{step}_{direction}.txt"
        with open(os.path.join(turns_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    def _progress_step(self, messages: list[dict], turn: int = 0) -> Any:
        if self.config.log_dir:
            self._write_turn_log(turn, "progress", "request", json.dumps({
                "system": self.task.PROGRESS_SYSTEM_PROMPT,
                "messages": messages,
            }, indent=2))

        resp = self.config.client.messages.create(
            model=self.config.model_progress,
            max_tokens=4096,
            system=self.task.PROGRESS_SYSTEM_PROMPT,
            tools=[t.to_api_def() for t in self.tools.values()],
            messages=messages,
        )

        if self.config.log_dir:
            self._write_turn_log(turn, "progress", "response", json.dumps(
                _content_to_dicts(resp.content), indent=2
            ))

        return resp

    def _metadata_step(self, artifact, tool_results: list[dict] | None = None, turn: int = 0) -> dict:
        if artifact is not None:
            try:
                artifact_str = json.dumps(artifact, indent=2)
            except (TypeError, ValueError):
                artifact_str = str(artifact)
        else:
            artifact_str = "No artifact produced."

        if tool_results:
            try:
                tool_results_str = json.dumps(tool_results, indent=2)
            except (TypeError, ValueError):
                tool_results_str = str(tool_results)
            tool_section = f"Tool results:\n{tool_results_str}\n\n"
        else:
            tool_section = ""

        # Build the required-fields documentation for the metadata response.
        standard_fields = [
            ('progress',       'string',  'Overall description of progress toward the task so far.'),
            ('success',        'boolean', 'True when the task is fully complete and the artifact satisfies the stopping criteria.'),
            ('failure',        'boolean', 'True when the agent is stuck in an unrecoverable or repeatedly bad state and cannot make progress.'),
            ('failure_reason', 'string',  'Human-readable explanation of why the agent considers itself failed. Empty string when failure is false.'),
        ]
        extra_fields = [
            (f['name'], f.get('type', 'string'), f['description'])
            for f in (self.task.METADATA_FIELDS or [])
        ]
        all_fields = standard_fields + extra_fields
        field_docs = "\n".join(
            f"  - {name} ({typ}): {desc}" for name, typ, desc in all_fields
        )

        user_msg = (
            f"Task: {self.task.get_prompt()}\n\n"
            f"Stopping criteria: {self.task.STOPPING_CRITERIA}\n\n"
            f"{tool_section}"
            f"Current artifact:\n{artifact_str}\n\n"
            f"Respond with a JSON object containing exactly these fields:\n{field_docs}"
        )
        if self.config.log_dir:
            self._write_turn_log(turn, "metadata", "request", json.dumps({
                "system": self.task.METADATA_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            }, indent=2))

        response = self.config.client.messages.create(
            model=self.config.model_metadata,
            max_tokens=1024,
            system=self.task.METADATA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))

        if self.config.log_dir:
            self._write_turn_log(turn, "metadata", "response", text)

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
                    result = self.tools[name].call(inputs, self.context)
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
        self._last_metadata = None

        terminal_failure = False

        for _attempt in range(self.config.max_attempts):
            messages = [{"role": "user", "content": self.task.get_prompt()}]
            all_tool_results: list[dict] = []
            artifact = None
            attempt_success = False

            for _turn in range(self.config.max_turns):
                turn_num = _attempt * self.config.max_turns + _turn + 1

                # Progress step
                progress_resp = self._progress_step(messages, turn=turn_num)

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

                # Between-turn hook — e.g. reset scenario state
                self.task.between_turns(_turn + 1, list(all_tool_results), self.context)

                # Produce artifact
                try:
                    artifact = self.task.produce_artifact(
                        self.task.prompt_fields, list(all_tool_results), self.context
                    )
                    self._produce_artifact_error = None
                except Exception:
                    self._produce_artifact_error = traceback.format_exc()
                    artifact = None

                # Metadata step
                metadata = self._metadata_step(artifact, list(all_tool_results), turn=turn_num)
                self._last_metadata = metadata

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

                if metadata.get("failure", False):
                    terminal_failure = True
                    break

                if metadata.get("success", False):
                    attempt_success = True
                    break

            if terminal_failure:
                if on_attempt is not None:
                    on_attempt(False, artifact)
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
                self.task.side_effects(artifact, self.context)
                return artifact

            if self.config.accept_attempts_if_no_error and all_no_errors:
                if on_attempt is not None:
                    on_attempt(True, artifact)
                self.task.side_effects(artifact, self.context)
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

    def last_metadata(self) -> Optional[dict]:
        """Return the metadata dict from the most recent metadata step.

        Returns ``None`` if ``run()`` has not been called yet or no metadata step
        has completed.
        """
        return self._last_metadata
