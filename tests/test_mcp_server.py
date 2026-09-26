"""Tests for the MCP server wiring: lifespan lifecycle, tool discovery, sync bridge."""

import asyncio

import pytest

import mcp_server
from tool_registry import RegisteredTool


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset mcp_server module globals between tests."""
    mcp_server._cached_graph = None
    mcp_server._tool_registry = None
    mcp_server._external_server_configs = None
    mcp_server._server_loop = None
    yield
    mcp_server._cached_graph = None
    mcp_server._tool_registry = None
    mcp_server._external_server_configs = None
    mcp_server._server_loop = None


async def test_discover_tools_empty_when_no_registry():
    assert await mcp_server._discover_tools() == []


async def test_discover_tools_returns_metadata():
    class _Reg:
        async def list_tools(self):
            return [RegisteredTool("fs", "filesystem", {"type": "object"}, "srv")]

    mcp_server._tool_registry = _Reg()
    tools = await mcp_server._discover_tools()
    assert tools == [{"name": "fs", "description": "filesystem", "input_schema": {"type": "object"}}]


async def test_sync_tool_executor_bridges_to_loop():
    """The sync bridge must run the async call_tool on the captured server loop."""
    class _Reg:
        async def call_tool(self, name, args):
            await asyncio.sleep(0)
            return f"executed {name} {args}"

    mcp_server._tool_registry = _Reg()
    mcp_server._server_loop = asyncio.get_running_loop()

    result = await asyncio.to_thread(mcp_server._sync_tool_executor, "search", {"q": 1})
    assert result == "executed search {'q': 1}"


async def test_sync_tool_executor_without_registry_raises():
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(mcp_server._sync_tool_executor, "x", {})


async def test_sync_tool_executor_deadlock_guard():
    """Calling on the server loop thread must fail fast, not deadlock."""
    class _Reg:
        async def call_tool(self, name, args):
            return "never reached"

    mcp_server._tool_registry = _Reg()
    mcp_server._server_loop = asyncio.get_running_loop()

    # Called directly on the event-loop thread (not via to_thread) -> must raise.
    with pytest.raises(RuntimeError, match="event loop thread"):
        mcp_server._sync_tool_executor("x", {})


def test_get_graph_builds_without_tools_when_no_registry():
    mcp_server._tool_registry = None
    graph = mcp_server._get_graph()
    from graph import TOOL_NODE
    assert TOOL_NODE not in graph.get_graph().nodes
    assert mcp_server._get_graph() is graph  # cached


def test_get_graph_builds_with_tool_node_when_registry_present():
    class _Reg:
        pass

    mcp_server._tool_registry = _Reg()
    graph = mcp_server._get_graph()
    from graph import TOOL_NODE
    assert TOOL_NODE in graph.get_graph().nodes


class TestLifespan:
    async def test_lifespan_no_configs_sets_loop_no_registry(self):
        mcp_server._external_server_configs = None
        async with mcp_server._lifespan(mcp_server.mcp):
            assert mcp_server._server_loop is asyncio.get_running_loop()
            assert mcp_server._tool_registry is None

    async def test_lifespan_initializes_and_shuts_down_registry(self, monkeypatch):
        events = []

        class _FakeRegistry:
            async def initialize(self, configs):
                events.append(("init", configs))

            async def list_tools(self):
                return []

            async def shutdown(self):
                events.append(("shutdown",))

        monkeypatch.setattr(mcp_server, "ToolRegistry", _FakeRegistry)
        mcp_server._external_server_configs = [object()]  # non-empty triggers init

        async with mcp_server._lifespan(mcp_server.mcp):
            assert mcp_server._tool_registry is not None
            assert events[0][0] == "init"

        # After exiting the context, shutdown must have been awaited (same task).
        assert ("shutdown",) in events

    async def test_lifespan_shutdown_error_is_swallowed(self, monkeypatch):
        class _BadRegistry:
            async def initialize(self, configs):
                pass

            async def list_tools(self):
                return []

            async def shutdown(self):
                raise RuntimeError("teardown boom")

        monkeypatch.setattr(mcp_server, "ToolRegistry", _BadRegistry)
        mcp_server._external_server_configs = [object()]

        # Must not raise despite shutdown() failing.
        async with mcp_server._lifespan(mcp_server.mcp):
            pass
