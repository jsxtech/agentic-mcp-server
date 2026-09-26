"""MCP Server exposing the multi-agent system via Model Context Protocol."""

import asyncio
import json
import logging

from mcp.server.fastmcp import FastMCP

from config import MCP_SERVER_NAME, MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS, MCP_REQUEST_TIMEOUT
from agent_registry import AGENTS, run_agent
from graph import build_agent_graph, AgentState
from tool_registry import ToolRegistry, ExternalServerConfig

logger = logging.getLogger(__name__)

mcp = FastMCP(MCP_SERVER_NAME)

# Cache compiled graph — it's stateless and reusable across requests.
# Note: this graph is built with a tool_executor bound to the server's event loop,
# so it is distinct from the CLI graph cache in runners.py (which has no external
# tools). The two caches intentionally differ; they are never used in the same process.
_cached_graph = None

# External tool registry — lazy-initialized on first use within the server's event loop
_tool_registry: ToolRegistry | None = None
_registry_initialized = False
_external_server_configs: list[ExternalServerConfig] | None = None

# The event loop running the FastMCP server. Captured on first tool call so that
# synchronous graph nodes (offloaded to worker threads) can call back into async
# tool sessions that live on this loop.
_server_loop: asyncio.AbstractEventLoop | None = None


def set_tool_registry(configs: list[ExternalServerConfig]):
    """Store configs for deferred initialization within the server's event loop."""
    global _external_server_configs
    _external_server_configs = configs


async def _ensure_registry():
    """Lazy-initialize tool registry within the current event loop on first use."""
    global _tool_registry, _registry_initialized, _server_loop
    if _registry_initialized:
        return
    _registry_initialized = True
    _server_loop = asyncio.get_running_loop()
    if _external_server_configs:
        _tool_registry = ToolRegistry()
        await _tool_registry.initialize(_external_server_configs)
        logger.info(f"Tool registry initialized: {len(await _tool_registry.list_tools())} external tools")


def _sync_tool_executor(tool_name: str, arguments: dict) -> str:
    """Synchronous bridge to execute an async external MCP tool from a worker thread.

    Graph nodes run synchronously inside ``asyncio.to_thread``. The external MCP
    sessions live on the server's event loop, so we schedule the coroutine there
    and block for the result.
    """
    if _tool_registry is None or _server_loop is None:
        raise RuntimeError("Tool registry is not initialized")
    future = asyncio.run_coroutine_threadsafe(
        _tool_registry.call_tool(tool_name, arguments), _server_loop
    )
    return future.result(timeout=MCP_REQUEST_TIMEOUT)


async def _get_graph():
    """Return the cached compiled graph, building it once on first use.

    Ensures the tool registry is initialized first so the graph can be built with a
    tool_executor when external servers are configured.
    """
    global _cached_graph
    await _ensure_registry()
    if _cached_graph is None:
        tool_executor = _sync_tool_executor if _tool_registry is not None else None
        _cached_graph = build_agent_graph(tool_executor=tool_executor)
    return _cached_graph


async def _discover_tools() -> list[dict]:
    """Return metadata for all discovered external tools (name/description/schema)."""
    if _tool_registry is None:
        return []
    tools = await _tool_registry.list_tools()
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


@mcp.tool()
async def run_multi_agent(query: str) -> str:
    """Run the full multi-agent orchestration pipeline on a user query.

    Args:
        query: The user's request to be processed by the agent team.

    Returns:
        The final synthesized answer from the agent orchestration.
    """
    graph = await _get_graph()
    available_tools = await _discover_tools()
    initial_state: AgentState = {
        "user_input": query,
        "agent_outputs": {},
        "next_agent": "",
        "final_answer": "",
        "iteration": 0,
        "tool_calls": [],
        "available_tools": available_tools,
        "pending_tool": "",
        "pending_args": {},
    }
    # graph.invoke() is blocking (calls Ollama HTTP) — run in thread to avoid blocking the event loop
    final_state = await asyncio.to_thread(graph.invoke, initial_state)
    return final_state.get("final_answer", "No response generated.")


@mcp.tool()
async def list_agents() -> str:
    """List all available agents with their descriptions and capabilities."""
    return json.dumps(
        {name: {"description": info["description"], "temperature": info["temperature"]}
         for name, info in AGENTS.items()},
        indent=2,
    )


@mcp.resource("agents://list")
async def get_agents_resource() -> str:
    """Return agent configurations as a readable resource."""
    return json.dumps(
        [{"name": n, "description": d["description"], "temperature": d["temperature"]}
         for n, d in AGENTS.items()],
        indent=2,
    )


@mcp.resource("config://system")
async def get_config_resource() -> str:
    """Return current system configuration as a readable resource."""
    return json.dumps({
        "model_name": MODEL_NAME,
        "ollama_base_url": OLLAMA_BASE_URL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "agent_count": len(AGENTS),
    }, indent=2)


# Dynamically register a tool for each agent
def _register_agent_tools():
    for name, info in AGENTS.items():
        # Create closure with correct binding
        def make_tool(agent_name, description):
            async def tool_fn(task: str) -> str:
                # run_agent() is blocking (Ollama HTTP call) — offload to thread
                return await asyncio.to_thread(run_agent, agent_name, task)
            tool_fn.__name__ = f"run_{agent_name}"
            tool_fn.__doc__ = f"""{description}

    Args:
        task: The task to execute.

    Returns:
        The agent's response.
"""
            return tool_fn

        mcp.tool()(make_tool(name, info["description"]))


_register_agent_tools()


def start_server(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8080):
    """Start the MCP server with the specified transport.

    The tool registry is lazy-initialized on first tool call, ensuring it runs
    within the same event loop that FastMCP creates for its server.
    """
    logger.info(f"Starting MCP server '{MCP_SERVER_NAME}' with {transport} transport ({len(AGENTS)} agents)")
    if transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")
