from abc import ABC, abstractmethod


class Task(ABC):
    """Base class for all tasks. Define class-level constants and implement produce_artifact()."""

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
    def produce_artifact(self, tool_results: list[dict]):
        """Produce the final artifact from all tool call results collected this attempt.

        Args:
            tool_results: All dicts returned by tool calls so far in this attempt.

        Returns:
            The final artifact (any JSON-serialisable value).

        Raises:
            Exception: If the artifact cannot be produced. The framework treats any
                       exception here as a failed attempt.
        """
