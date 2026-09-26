"""Tests for the MCP server wiring: tool discovery and the sync tool bridge."""

import asyncio

import pytest

import mcp_server
from tool_registry import RegisteredTool


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset mcp_server module globals between tests."""
    mcp_server._cached_graph = None
    mcp_server._tool_registry = None
    mcp_server._registry_initialized = False
    mcp_server._external_server_configs = None
    mcp_server._server_loop = None
    yield
    mcp_server._cached_graph = None
    mcp_server._tool_registry = None
    mcp_server._registry_initialized = False
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
            await asyncio.sleep(0)  # prove it runs as a coroutine on the loop
            return f"executed {name} {args}"

    mcp_server._tool_registry = _Reg()
    mcp_server._server_loop = asyncio.get_running_loop()

    # The bridge blocks on future.result(), so it must run off the event loop thread.
    result = await asyncio.to_thread(mcp_server._sync_tool_executor, "search", {"q": 1})
    assert result == "executed search {'q': 1}"


async def test_sync_tool_executor_without_registry_raises():
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(mcp_server._sync_tool_executor, "x", {})


async def test_get_graph_builds_without_tools():
    # No external configs -> registry stays None -> graph built without tool executor.
    graph = await mcp_server._get_graph()
    from graph import TOOL_NODE
    assert TOOL_NODE not in graph.get_graph().nodes
    # Cached on second call
    assert await mcp_server._get_graph() is graph
