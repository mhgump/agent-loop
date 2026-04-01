"""Journey 2: MockTask + MockMetadataResponse([False * 10]) -> artifact is NOT produced by framework."""
from agent_loop import AgentLoop, AgentLoopConfig
from conftest import MockClient, MockMetadataResponse, MockProgressResponse, MockTask, MockTool


def test_no_artifact_when_metadata_never_succeeds():
    client = MockClient(
        progress_mock=MockProgressResponse("mock_tool"),
        metadata_mock=MockMetadataResponse([False] * 10),
    )
    config = AgentLoopConfig(client=client, max_attempts=2, max_turns=2)
    task = MockTask({"goal": "test"})

    artifact = AgentLoop(task=task, tools=[MockTool()], config=config).run()

    assert artifact is None
