"""Shared mock utilities for agent-loop tests."""

import json

from agent_loop import Task, Tool


class MockTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class MockResponse:
    def __init__(self, content: list):
        self.content = content


class MockProgressResponse:
    """Mock progress LLM: always returns a single tool call with sequential inputs."""

    def __init__(self, tool_name: str = "mock_tool"):
        self.tool_name = tool_name
        self._call_count = 0

    def __call__(self, **kwargs):
        self._call_count += 1
        return MockResponse(content=[
            MockToolUseBlock(
                id=f"tu_{self._call_count}",
                name=self.tool_name,
                input={"value": f"call_{self._call_count}"},
            )
        ])


class MockMetadataResponse:
    """Mock metadata LLM: returns success based on a configured list of bools."""

    def __init__(self, successes: list):
        self._successes = successes
        self._call_count = 0

    def __call__(self, **kwargs):
        success = (
            self._successes[self._call_count]
            if self._call_count < len(self._successes)
            else False
        )
        self._call_count += 1
        return MockResponse(content=[
            MockTextBlock(json.dumps({"progress": "In progress.", "success": success}))
        ])


class MockMessages:
    def __init__(self, progress_mock, metadata_mock):
        self._progress = progress_mock
        self._metadata = metadata_mock

    def create(self, **kwargs):
        if kwargs.get("tools"):
            return self._progress(**kwargs)
        return self._metadata(**kwargs)


class MockClient:
    def __init__(self, progress_mock, metadata_mock):
        self.messages = MockMessages(progress_mock, metadata_mock)


class MockTool(Tool):
    spec = {
        "name": "mock_tool",
        "description": "A mock tool for testing.",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    }

    def call(self, **kwargs) -> dict:
        return {"result": "executed", "errors": None}


class MockTask(Task):
    PROMPT_TEMPLATE = "Complete the task: {goal}"
    STOPPING_CRITERIA = "Task is complete when an artifact is returned."
    PROGRESS_SYSTEM_PROMPT = "You are a helpful assistant."
    METADATA_SYSTEM_PROMPT = "Evaluate whether the task is complete."

    def produce_artifact(self, tool_results: list[dict]):
        return {"status": "complete", "tool_count": len(tool_results)}


class MockFailingTask(Task):
    PROMPT_TEMPLATE = "Complete the task: {goal}"
    STOPPING_CRITERIA = "Task is complete when an artifact is returned."
    PROGRESS_SYSTEM_PROMPT = "You are a helpful assistant."
    METADATA_SYSTEM_PROMPT = "Evaluate whether the task is complete."

    def produce_artifact(self, tool_results: list[dict]):
        raise RuntimeError("This task always fails")
