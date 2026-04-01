"""Journey 3: MockFailingTask + MockMetadataResponse([True * 10]) -> artifact is NOT produced by framework."""
from agent_loop import AgentLoop, AgentLoopConfig
from conftest import MockClient, MockFailingTask, MockMetadataResponse, MockProgressResponse, MockTool


def test_no_artifact_when_task_always_raises():
    client = MockClient(
        progress_mock=MockProgressResponse("mock_tool"),
        metadata_mock=MockMetadataResponse([True] * 10),
    )
    config = AgentLoopConfig(client=client, max_attempts=2, max_turns=2)
    task = MockFailingTask({"goal": "test"})

    artifact = AgentLoop(task=task, tools=[MockTool()], config=config).run()

    assert artifact is None
