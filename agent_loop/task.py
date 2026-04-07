from abc import ABC, abstractmethod


class Task(ABC):
    """Base class for all tasks.

    Each task serves two purposes:
    1. **Produce an artifact** — `produce_artifact()` assembles the final result from tool
       call outputs. The framework calls this after every turn; returning ``None`` or raising
       an exception causes the attempt to be treated as failed. Exceptions are caught
       silently — retrieve the traceback via ``AgentLoop.produce_artifact_error()``.
    2. **Apply side effects** — `side_effects()` is called once after `produce_artifact()`
       returns without error and the attempt is accepted. Override it to persist the artifact,
       send notifications, or take any other action that should happen exactly once on success.
       The default implementation does nothing.

    Define four class-level string constants and implement `produce_artifact()`. Optionally
    override `side_effects()`.
    """

    PROMPT_TEMPLATE: str = ""
    """Template string for the task prompt. Use {field_name} placeholders."""

    STOPPING_CRITERIA: str = ""
    """Plain-text description of when the task is considered complete."""

    PROGRESS_SYSTEM_PROMPT: str = ""
    """System prompt for the progress (worker) LLM calls."""

    METADATA_SYSTEM_PROMPT: str = ""
    """System prompt for the metadata (evaluator) LLM calls."""

    METADATA_FIELDS: list[dict] = []
    """Additional fields the task wants included in every metadata response.

    Each entry is a dict with keys:
      - "name"        (str): JSON field name the LLM must return.
      - "description" (str): What the field should contain.
      - "type"        (str): JSON type, e.g. "string", "number", "boolean".

    These fields are injected into the metadata prompt alongside the standard
    ``progress``, ``success``, ``failure``, and ``failure_reason`` fields.
    """

    def __init__(self, prompt_fields: dict):
        self.prompt_fields = prompt_fields

    def get_prompt(self) -> str:
        """Render the prompt template with the provided fields."""
        return self.PROMPT_TEMPLATE.format(**self.prompt_fields)

    @abstractmethod
    def produce_artifact(self, task_inputs: dict, tool_results: list[dict], context: dict | None = None):
        """Produce the final artifact from task inputs and tool call results.

        Args:
            task_inputs: The dict passed to the task constructor (i.e. ``prompt_fields``).
                         Use these to drive artifact construction programmatically — for
                         example, to include the original query, target entity, or any
                         other task-level parameter in the returned artifact.
            tool_results: All dicts returned by tool calls so far in this attempt.
            context: Optional context dict from the AgentLoop (e.g. server/bot references).

        Returns:
            The final artifact (any JSON-serialisable value), or ``None`` to signal
            that the artifact could not be produced. Any exception raised here is
            caught by the framework, treated as a ``None`` return, and stored
            internally — retrieve it via ``AgentLoop.produce_artifact_error()``.
        """

    def side_effects(self, artifact, context: dict | None = None) -> None:
        """Apply side effects after a successful artifact has been produced.

        The framework calls this exactly once, after ``produce_artifact()`` has returned
        without raising and the attempt has been accepted. Override to persist the artifact,
        emit notifications, write files, call external APIs, etc.

        Args:
            artifact: The value returned by ``produce_artifact()``.
            context: Optional context dict from the AgentLoop (e.g. server/bot references).
        """

    def extra_progress_messages(self, context: dict | None = None) -> list[dict]:
        """Return extra messages to inject at the start of the progress conversation.

        These are prepended before the initial task prompt on every attempt, allowing
        tasks to provide persistent reference material (API docs, schemas, etc.) that
        should be visible throughout the entire conversation.

        Must return a valid alternating sequence of user/assistant messages (the Anthropic
        API requires turns to alternate). Typically one user message followed by one
        assistant acknowledgment. Defaults to an empty list (no extra context).

        Args:
            context: Optional context dict from the AgentLoop.
        """
        return []

    def between_turns(self, turn_number: int, tool_results: list[dict], context: dict | None = None) -> None:
        """Called between each progress turn, after tools have executed.

        Override to perform per-turn housekeeping — for example, resetting the game
        scenario state so the agent always sees the initial world at the start of the
        next turn. The default implementation does nothing.

        Args:
            turn_number: 1-based index of the turn that just completed.
            tool_results: All tool-call result dicts accumulated so far in this attempt.
            context: Optional context dict from the AgentLoop (e.g. server/bot references).
        """
