from dataclasses import dataclass, field
from typing import Any


def _default_client() -> Any:
    """Create a default Anthropic client, loading ANTHROPIC_API_KEY from .env if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from anthropic import Anthropic
    return Anthropic()


@dataclass
class AgentLoopConfig:
    """Runtime configuration for an agent loop execution."""

    client: Any = field(default_factory=_default_client)
    """Anthropic client instance. Defaults to Anthropic() with ANTHROPIC_API_KEY from .env."""

    model_progress: str = "claude-haiku-4-5-20251001"
    """Model ID for progress (worker) calls."""

    model_metadata: str = "claude-haiku-4-5-20251001"
    """Model ID for metadata (evaluator) calls."""

    max_attempts: int = 3
    """Maximum number of outer retry attempts."""

    max_turns: int = 10
    """Maximum progress/metadata turns per attempt."""

    accept_attempts_if_no_error: bool = False
    """When True, accept an attempt that reaches max_turns if all tools had no errors
    and the task produced an artifact, even if metadata did not return success: true."""
