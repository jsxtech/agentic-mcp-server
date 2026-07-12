"""MCP Server exposing the multi-agent system via Model Context Protocol."""

import asyncio
import json
import logging

from mcp.server.fastmcp import FastMCP

from config import MCP_SERVER_NAME, MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS
from agent_registry import AGENTS, run_agent
from graph import build_agent_graph, AgentState
from tool_registry import ToolRegistry, ExternalServerConfig

logger = logging.getLogger(__name__)

mcp = FastMCP(MCP_SERVER_NAME)

# Cache compiled graph — it's stateless and reusable across requests
_cached_graph = None

# External tool registry — lazy-initialized on first use within the server's event loop
_tool_registry: ToolRegistry | None = None
_registry_initialized = False
_external_server_configs: list[ExternalServerConfig] | None = None


def _get_graph():
    """Return the cached compiled graph, building it once on first use."""
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_agent_graph()
    return _cached_graph


def set_tool_registry(configs: list[ExternalServerConfig]):
    """Store configs for deferred initialization within the server's event loop."""
    global _external_server_configs
    _external_server_configs = configs


async def _ensure_registry():
    """Lazy-initialize tool registry within the current event loop on first use."""
    global _tool_registry, _registry_initialized
    if _registry_initialized:
        return
    _registry_initialized = True
    if _external_server_configs:
        _tool_registry = ToolRegistry()
        await _tool_registry.initialize(_external_server_configs)
        logger.info(f"Tool registry initialized: {len(await _tool_registry.list_tools())} external tools")


@mcp.tool()
async def run_multi_agent(query: str) -> str:
    """Run the full multi-agent orchestration pipeline on a user query.

    Args:
        query: The user's request to be processed by the agent team.

    Returns:
        The final synthesized answer from the agent orchestration.
    """
    await _ensure_registry()
    graph = _get_graph()
    initial_state: AgentState = {
        "user_input": query,
        "agent_outputs": {},
        "next_agent": "",
        "final_answer": "",
        "iteration": 0,
        "tool_calls": [],
        "available_tools": [],
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
