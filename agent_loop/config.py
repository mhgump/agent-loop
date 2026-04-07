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

    max_tokens_progress: int = 16384
    """Max tokens for progress (worker) calls."""

    max_tokens_metadata: int = 1024
    """Max tokens for metadata (evaluator) calls."""

    accept_attempts_if_no_error: bool = False
    """When True, accept an attempt that reaches max_turns if all tools had no errors
    and the task produced an artifact, even if metadata did not return success: true."""

    cache_control: dict | None = None
    """If set, applied as cache_control on the system prompt in all LLM calls.
    Example: {"type": "ephemeral", "ttl": "1h"}"""

    log_dir: str | None = None
    """If set, per-turn LLM request/response logs are written to {log_dir}/turns/
    as NN_progress_request.txt, NN_progress_response.txt, NN_metadata_request.txt,
    NN_metadata_response.txt."""
