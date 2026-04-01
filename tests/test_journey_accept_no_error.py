"""Journey 4: MockTask + MockMetadataResponse([False * 10]) + accept_attempts_if_no_error -> artifact produced."""
from agent_loop import AgentLoop, AgentLoopConfig
from conftest import MockClient, MockMetadataResponse, MockProgressResponse, MockTask, MockTool


def test_artifact_produced_with_accept_attempts_if_no_error():
    client = MockClient(
        progress_mock=MockProgressResponse("mock_tool"),
        metadata_mock=MockMetadataResponse([False] * 10),
    )
    config = AgentLoopConfig(
        client=client,
        max_attempts=2,
        max_turns=2,
        accept_attempts_if_no_error=True,
    )
    task = MockTask({"goal": "test"})

    artifact = AgentLoop(task=task, tools=[MockTool()], config=config).run()

    assert artifact is not None
    assert artifact["status"] == "complete"
