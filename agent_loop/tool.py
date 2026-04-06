from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools. Define spec as a class attribute and implement call()."""

    spec: dict = {}
    """Anthropic tool definition JSON (name, description, input_schema)."""

    @abstractmethod
    def call(self, input: dict, context: dict | None = None) -> dict:
        """Execute the tool.

        Args:
            input: Tool inputs as a dict matching the input_schema defined in spec.
            context: Optional context dict passed down from the AgentLoop. Contains any
                     Python objects the caller registered (e.g. server/bot references).

        Returns a JSON-serialisable dict that MUST contain an 'errors' field:
          - None  when the tool succeeded
          - str   describing the error when the tool failed
        """

    @property
    def name(self) -> str:
        return self.spec.get("name", "")

    def to_api_def(self) -> dict:
        """Return the spec dict suitable for the Anthropic tools parameter."""
        return self.spec
