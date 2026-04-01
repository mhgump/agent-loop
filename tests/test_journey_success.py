"""Journey 1: MockTask + MockMetadataResponse([True * 10]) -> artifact is produced by framework."""
from agent_loop import AgentLoop, AgentLoopConfig
from conftest import MockClient, MockMetadataResponse, MockProgressResponse, MockTask, MockTool


def test_artifact_produced_when_metadata_succeeds():
    client = MockClient(
        progress_mock=MockProgressResponse("mock_tool"),
        metadata_mock=MockMetadataResponse([True] * 10),
    )
    config = AgentLoopConfig(client=client, max_attempts=3, max_turns=5)
    task = MockTask({"goal": "test"})

    artifact = AgentLoop(task=task, tools=[MockTool()], config=config).run()

    assert artifact is not None
    assert artifact["status"] == "complete"
