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

    def _metadata_step(self, messages: list[dict], api_results: list[dict], turn: int = 0) -> dict:
        """Evaluate the current conversation state and return metadata.

        Shares the same conversation history as the progress step — the evaluator
        sees the full multi-turn context (tool calls and their results) rather than
        a summarised artifact. ``api_results`` are the tool_result blocks from the
        current turn; they are appended alongside the evaluation request so the
        evaluator sees the latest tool outputs without polluting the real messages list.
        """
        # Build the required-fields documentation for the metadata response.
        standard_fields = [
            ('progress',       'string',  'Overall description of progress toward the task so far.'),
            ('success',        'boolean',
             'True ONLY when you are fully confident the agent has completed the task and ALL '
             'stopping criteria are satisfied. Default to false when there is any uncertainty.'),
            ('failure',        'boolean',
             'True ONLY when you are fully confident the agent has been stuck across multiple '
             'turns and cannot make further progress toward the goal. A single failed step, '
             'partial progress, or recoverable error does NOT qualify. Default to false when '
             'there is any uncertainty.'),
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

        eval_text = (
            f"Task: {self.task.get_prompt()}\n\n"
            f"Stopping criteria: {self.task.STOPPING_CRITERIA}\n\n"
            f"Evaluate the conversation above and respond with a JSON object containing "
            f"exactly these fields:\n{field_docs}\n\n"
            f"Be conservative: prefer continuing the conversation over declaring success or "
            f"failure prematurely. Only set success=true or failure=true when you are fully "
            f"confident — if in doubt, leave both false."
        )

        # Combine tool results (if any) with the evaluation request into a single user
        # message so the conversation remains valid (tool_result blocks must immediately
        # follow the assistant turn that produced the corresponding tool_use blocks).
        if api_results:
            eval_content: Any = list(api_results) + [{"type": "text", "text": eval_text}]
        else:
            eval_content = eval_text

        metadata_messages = messages + [{"role": "user", "content": eval_content}]

        if self.config.log_dir:
            self._write_turn_log(turn, "metadata", "request", json.dumps({
                "system": self.task.METADATA_SYSTEM_PROMPT,
                "messages": metadata_messages,
            }, indent=2))

        response = self.config.client.messages.create(
            model=self.config.model_metadata,
            max_tokens=1024,
            system=self.task.METADATA_SYSTEM_PROMPT,
            messages=metadata_messages,
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
            extra = self.task.extra_progress_messages(self.context)
            messages = extra + [{"role": "user", "content": self.task.get_prompt()}]
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

                # Metadata step — evaluates the conversation as-is; produce_artifact is
                # deferred until success/failure is confirmed below.
                metadata = self._metadata_step(messages, api_results, turn=turn_num)
                self._last_metadata = metadata

                if on_metadata is not None:
                    on_metadata(metadata)

                # Append user message with tool results and evaluation context
                if tool_uses:
                    user_content: Any = list(api_results)
                    user_content.append({
                        "type": "text",
                        "text": f"Evaluation: {json.dumps(metadata)}",
                    })
                else:
                    user_content = f"Evaluation: {json.dumps(metadata)}"

                messages.append({"role": "user", "content": user_content})

                if metadata.get("failure", False) or metadata.get("success", False):
                    # Store terminal metadata in context so produce_artifact can react to it,
                    # then re-run produce_artifact once to let tasks perform terminal actions
                    # (e.g. a final clean replay for timing data) before the artifact is returned.
                    if self.context is not None:
                        self.context["_terminal_metadata"] = metadata
                    try:
                        artifact = self.task.produce_artifact(
                            self.task.prompt_fields, list(all_tool_results), self.context
                        )
                        self._produce_artifact_error = None
                    except Exception:
                        self._produce_artifact_error = traceback.format_exc()
                        artifact = None
                    if metadata.get("failure", False):
                        terminal_failure = True
                    else:
                        attempt_success = True
                    break

            if terminal_failure:
                if on_attempt is not None:
                    on_attempt(False, artifact)
                break

            # Evaluate attempt outcome
            all_no_errors = all(r.get("errors") is None for r in all_tool_results)

            if attempt_success:
                # produce_artifact was already called in the terminal block above.
                if artifact is not None:
                    if on_attempt is not None:
                        on_attempt(True, artifact)
                    self.task.side_effects(artifact, self.context)
                    return artifact
                # produce_artifact raised even though metadata declared success; treat as failure.
                if on_attempt is not None:
                    on_attempt(False, None)
                continue

            if self.config.accept_attempts_if_no_error and all_no_errors:
                # No success/failure from metadata — accept the attempt if tools all ran cleanly.
                # produce_artifact is called here (only time it runs without a terminal signal).
                try:
                    artifact = self.task.produce_artifact(
                        self.task.prompt_fields, list(all_tool_results), self.context
                    )
                    self._produce_artifact_error = None
                except Exception:
                    self._produce_artifact_error = traceback.format_exc()
                    artifact = None
                if artifact is not None:
                    if on_attempt is not None:
                        on_attempt(True, artifact)
                    self.task.side_effects(artifact, self.context)
                    return artifact

            if on_attempt is not None:
                on_attempt(False, None)

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
