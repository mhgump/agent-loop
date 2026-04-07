"""Tests that verify conversation threading and system-prompt placement of format instructions.

Goal: ensure each progress/metadata API call receives a messages list that is a proper
continuation of the previous call, so the API's KV cache can be reused across turns
instead of paying the full input-token price on every call.
"""
from agent_loop import AgentLoop, AgentLoopConfig
from conftest import MockMetadataResponse, MockProgressResponse, MockTask, MockTool


def _make_capturing_client(progress_mock, metadata_mock):
    """Return (client, progress_calls, metadata_calls).

    Each list entry is a dict with the ``messages`` and ``system`` kwargs captured
    from the corresponding API call.
    """
    progress_calls = []
    metadata_calls = []

    class CapturingMessages:
        def create(self, **kwargs):
            entry = {
                "messages": list(kwargs.get("messages", [])),
                "system": kwargs.get("system", ""),
            }
            if kwargs.get("tools"):
                progress_calls.append(entry)
                return progress_mock(**kwargs)
            else:
                metadata_calls.append(entry)
                return metadata_mock(**kwargs)

    class CapturingClient:
        def __init__(self):
            self.messages = CapturingMessages()

    return CapturingClient(), progress_calls, metadata_calls


def test_each_progress_call_extends_previous():
    """Turn N+1 progress messages start with all of turn N's progress messages."""
    client, progress_calls, _ = _make_capturing_client(
        MockProgressResponse("mock_tool"),
        MockMetadataResponse([False, True]),  # two turns
    )
    config = AgentLoopConfig(client=client, max_attempts=1, max_turns=5)

    AgentLoop(task=MockTask({"goal": "test"}), tools=[MockTool()], config=config).run()

    assert len(progress_calls) == 2, "Expected exactly two progress calls"
    prev = progress_calls[0]["messages"]
    curr = progress_calls[1]["messages"]
    assert len(curr) > len(prev), (
        f"Turn 2 progress messages ({len(curr)}) should be longer than turn 1 ({len(prev)})"
    )
    assert curr[: len(prev)] == prev, (
        "Turn 2 progress messages should start with all of turn 1's messages"
    )


def test_each_metadata_call_extends_same_turn_progress_call():
    """Metadata messages for turn N start with all of turn N's progress messages."""
    client, progress_calls, metadata_calls = _make_capturing_client(
        MockProgressResponse("mock_tool"),
        MockMetadataResponse([False, True]),  # two turns
    )
    config = AgentLoopConfig(client=client, max_attempts=1, max_turns=5)

    AgentLoop(task=MockTask({"goal": "test"}), tools=[MockTool()], config=config).run()

    assert len(progress_calls) == len(metadata_calls) == 2
    for i, (prog, meta) in enumerate(zip(progress_calls, metadata_calls)):
        prog_msgs = prog["messages"]
        meta_msgs = meta["messages"]
        assert len(meta_msgs) > len(prog_msgs), (
            f"Turn {i + 1} metadata messages ({len(meta_msgs)}) should be longer "
            f"than progress messages ({len(prog_msgs)})"
        )
        assert meta_msgs[: len(prog_msgs)] == prog_msgs, (
            f"Turn {i + 1} metadata messages should start with the same-turn progress messages"
        )


def test_metadata_system_prompt_contains_format_instructions():
    """Field docs and stopping criteria appear in the metadata system prompt, not inline."""
    client, _, metadata_calls = _make_capturing_client(
        MockProgressResponse("mock_tool"),
        MockMetadataResponse([True]),  # one turn
    )
    config = AgentLoopConfig(client=client, max_attempts=1, max_turns=5)
    task = MockTask({"goal": "test"})

    AgentLoop(task=task, tools=[MockTool()], config=config).run()

    assert metadata_calls, "Expected at least one metadata call"
    system = metadata_calls[0]["system"]

    # All standard field names must appear in the system prompt.
    for field in ("progress", "success", "failure", "failure_reason"):
        assert field in system, f"Field '{field}' missing from metadata system prompt"

    # Stopping criteria must appear in the system prompt.
    assert MockTask.STOPPING_CRITERIA in system, (
        "STOPPING_CRITERIA missing from metadata system prompt"
    )

    # The per-turn eval message should be a minimal trigger, not contain format instructions.
    last_user_msg = metadata_calls[0]["messages"][-1]
    content = last_user_msg.get("content", "")
    if isinstance(content, list):
        eval_text = " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        eval_text = content
    assert "Stopping criteria" not in eval_text, (
        "Stopping criteria should not be repeated in the per-turn eval message"
    )
    assert "failure_reason" not in eval_text, (
        "Field docs should not be repeated in the per-turn eval message"
    )
