"""Tests for the external MCP tool registry (async, with mocked sessions)."""

import pytest

from tool_registry import (
    ExternalServerConfig,
    RegisteredTool,
    ToolRegistry,
    ToolNotFoundError,
    ToolExecutionError,
)


class _FakeTool:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {}


class _FakeToolsResponse:
    def __init__(self, tools):
        self.tools = tools


class _FakeResult:
    def __init__(self, text, is_error=False):
        self.isError = is_error

        class _Content:
            def __init__(self, t):
                self.text = t

        self.content = [_Content(text)] if text is not None else []


class _FakeSession:
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def list_tools(self):
        return _FakeToolsResponse(self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "boom":
            return _FakeResult("error detail", is_error=True)
        return _FakeResult(f"ran {name}")


class TestExternalServerConfig:
    def test_from_dict(self):
        c = ExternalServerConfig(**{"name": "fs", "transport": "stdio", "command": "npx", "args": ["x"]})
        assert c.name == "fs"
        assert c.transport == "stdio"
        assert c.args == ["x"]


class TestToolRegistry:
    @pytest.fixture
    def registry_with_session(self):
        reg = ToolRegistry()
        session = _FakeSession([_FakeTool("search", "search tool"), _FakeTool("read", "read tool")])
        reg._sessions["srv"] = session
        reg._tools = {
            "search": RegisteredTool("search", "search tool", {}, "srv"),
            "read": RegisteredTool("read", "read tool", {}, "srv"),
        }
        return reg, session

    async def test_list_tools(self, registry_with_session):
        reg, _ = registry_with_session
        tools = await reg.list_tools()
        names = {t.name for t in tools}
        assert names == {"search", "read"}

    async def test_call_tool_success(self, registry_with_session):
        reg, session = registry_with_session
        result = await reg.call_tool("search", {"q": "hi"})
        assert result == "ran search"
        assert session.calls == [("search", {"q": "hi"})]

    async def test_call_unknown_tool_raises(self, registry_with_session):
        reg, _ = registry_with_session
        with pytest.raises(ToolNotFoundError):
            await reg.call_tool("ghost", {})

    async def test_call_tool_error_result_raises(self, registry_with_session):
        reg, _ = registry_with_session
        reg._tools["boom"] = RegisteredTool("boom", "", {}, "srv")
        with pytest.raises(ToolExecutionError):
            await reg.call_tool("boom", {})

    async def test_call_tool_no_session_raises(self):
        reg = ToolRegistry()
        reg._tools["orphan"] = RegisteredTool("orphan", "", {}, "gone")
        with pytest.raises(ToolExecutionError):
            await reg.call_tool("orphan", {})

    async def test_refresh_replaces_tools(self, registry_with_session):
        reg, session = registry_with_session
        session._tools = [_FakeTool("newtool", "fresh")]
        await reg.refresh("srv")
        names = {t.name for t in await reg.list_tools()}
        assert names == {"newtool"}

    async def test_refresh_unknown_server_noop(self, registry_with_session):
        reg, _ = registry_with_session
        before = {t.name for t in await reg.list_tools()}
        await reg.refresh("does-not-exist")
        after = {t.name for t in await reg.list_tools()}
        assert before == after

    async def test_initialize_discovers_tools(self):
        reg = ToolRegistry()
        session = _FakeSession([_FakeTool("t1"), _FakeTool("t2")])

        async def _fake_connect(config):
            return session

        reg._connect = _fake_connect
        await reg.initialize([ExternalServerConfig(name="srv", transport="stdio", command="x")])
        names = {t.name for t in await reg.list_tools()}
        assert names == {"t1", "t2"}

    async def test_initialize_handles_connect_failure(self):
        reg = ToolRegistry()

        async def _fail(config):
            raise ConnectionError("cannot connect")

        reg._connect = _fail
        # Should not raise — failures are logged and skipped
        await reg.initialize([ExternalServerConfig(name="bad", transport="stdio", command="x")])
        assert await reg.list_tools() == []

    async def test_connect_rejects_unknown_transport(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError):
            await reg._connect(ExternalServerConfig(name="x", transport="carrier-pigeon"))
