"""Tool Registry for managing external MCP server connections and tools."""

import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


@dataclass
class ExternalServerConfig:
    """Configuration for an external MCP server connection."""
    name: str
    transport: str  # "stdio" or "sse"
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None


@dataclass
class RegisteredTool:
    """A tool discovered from an external MCP server."""
    name: str
    description: str
    input_schema: dict
    server_name: str


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not in the registry."""


class ToolExecutionError(Exception):
    """Raised when an external MCP server returns an error during tool execution."""


class ToolRegistry:
    """Registry managing external MCP server connections and their tools."""

    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()

    async def initialize(self, configs: list[ExternalServerConfig]) -> None:
        """Connect to all configured external MCP servers and discover tools."""
        for config in configs:
            try:
                session = await self._connect(config)
                self._sessions[config.name] = session
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    self._tools[tool.name] = RegisteredTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        server_name=config.name,
                    )
                logger.info(f"Connected to '{config.name}': {len(tools_response.tools)} tools discovered")
            except Exception as e:
                logger.warning(f"Failed to connect to '{config.name}': {e}")

    async def _connect(self, config: ExternalServerConfig) -> ClientSession:
        """Establish connection to an external MCP server using managed exit stack."""
        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session
        elif config.transport == "sse":
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(config.url)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    async def list_tools(self) -> list[RegisteredTool]:
        """Return all discovered tools from all connected external servers."""
        return list(self._tools.values())

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke a tool on its owning external MCP server."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")

        session = self._sessions.get(tool.server_name)
        if not session:
            raise ToolExecutionError(f"No active session for server '{tool.server_name}'")

        try:
            start = time.time()
            result = await session.call_tool(tool_name, arguments)
            duration_ms = int((time.time() - start) * 1000)
            logger.info(f"Tool '{tool_name}' executed in {duration_ms}ms")

            if result.isError:
                raise ToolExecutionError(f"Tool '{tool_name}' returned error: {result.content}")
            return result.content[0].text if result.content else ""
        except (ToolNotFoundError, ToolExecutionError):
            raise
        except Exception as e:
            raise ToolExecutionError(f"Failed to call tool '{tool_name}': {e}") from e

    async def refresh(self, server_name: str) -> None:
        """Re-discover tools from a specific external server."""
        session = self._sessions.get(server_name)
        if not session:
            logger.warning(f"No session for server '{server_name}' to refresh")
            return
        # Remove old tools from this server
        self._tools = {k: v for k, v in self._tools.items() if v.server_name != server_name}
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            self._tools[tool.name] = RegisteredTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                server_name=server_name,
            )

    async def shutdown(self) -> None:
        """Close all external MCP server connections via the exit stack."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
