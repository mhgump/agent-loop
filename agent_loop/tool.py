from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools. Define spec as a class attribute and implement call()."""

    spec: dict = {}
    """Anthropic tool definition JSON (name, description, input_schema)."""

    @abstractmethod
    def call(self, **kwargs) -> dict:
        """Execute the tool.

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
