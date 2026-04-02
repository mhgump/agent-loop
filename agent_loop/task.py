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

    def __init__(self, prompt_fields: dict):
        self.prompt_fields = prompt_fields

    def get_prompt(self) -> str:
        """Render the prompt template with the provided fields."""
        return self.PROMPT_TEMPLATE.format(**self.prompt_fields)

    @abstractmethod
    def produce_artifact(self, task_inputs: dict, tool_results: list[dict]):
        """Produce the final artifact from task inputs and tool call results.

        Args:
            task_inputs: The dict passed to the task constructor (i.e. ``prompt_fields``).
                         Use these to drive artifact construction programmatically — for
                         example, to include the original query, target entity, or any
                         other task-level parameter in the returned artifact.
            tool_results: All dicts returned by tool calls so far in this attempt.

        Returns:
            The final artifact (any JSON-serialisable value), or ``None`` to signal
            that the artifact could not be produced. Any exception raised here is
            caught by the framework, treated as a ``None`` return, and stored
            internally — retrieve it via ``AgentLoop.produce_artifact_error()``.
        """

    def side_effects(self, artifact) -> None:
        """Apply side effects after a successful artifact has been produced.

        The framework calls this exactly once, after ``produce_artifact()`` has returned
        without raising and the attempt has been accepted. Override to persist the artifact,
        emit notifications, write files, call external APIs, etc.

        Args:
            artifact: The value returned by ``produce_artifact()``.
        """
