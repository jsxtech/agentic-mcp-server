"""Multi-agent orchestration graph using LangGraph."""

import json
import logging
from typing import Callable, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agent_registry import run_agent, AGENT_NAMES
import config as _config

logger = logging.getLogger(__name__)

# Context-window protection: cap how much accumulated agent/tool output is fed
# forward into subsequent nodes. Mirrors the safeguard in runners.py so the
# graph path can't blow past the model's context window on long supervisor loops.
MAX_CONTEXT_CHARS = 8000
MAX_TOOL_RESULT_CHARS = 4000

# Sentinel used by the supervisor to request execution of an external MCP tool.
USE_TOOL = "use_tool"
TOOL_NODE = "tool_executor"

# A synchronous callable that executes an external MCP tool: (name, args) -> str.
ToolExecutor = Callable[[str, dict], str]


class AgentState(TypedDict, total=False):
    """State that flows through the agent graph."""
    user_input: str
    agent_outputs: dict[str, str]
    next_agent: str
    final_answer: str
    iteration: int
    tool_calls: list[dict]
    available_tools: list[dict]
    pending_tool: str
    pending_args: dict


def _truncate(text: str, limit: int) -> str:
    """Truncate text to a character budget with a visible marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated — {len(text)} chars total, showing first {limit}]"


def _format_previous_outputs(agent_outputs: dict[str, str]) -> str:
    """Serialize prior agent/tool outputs for a follow-on prompt, bounded in size."""
    serialized = json.dumps(agent_outputs, indent=2)
    return _truncate(serialized, MAX_CONTEXT_CHARS)


def _tool_usage_instructions(available_tools: list[dict]) -> str:
    """Extra supervisor instructions enabling external tool use (only when tools exist)."""
    if not available_tools:
        return ""
    tool_lines = "\n".join(f'  - "{t["name"]}": {t.get("description", "")}' for t in available_tools)
    return (
        "\n\nYou may also invoke an external tool instead of an agent. To do so, respond with:\n"
        '{"next": "use_tool", "tool": "tool_name", "args": {"key": "value"}}\n'
        "Available external tools:\n"
        f"{tool_lines}\n"
    )


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor decides which agent to call next, whether to use a tool, or finishes."""
    llm = ChatOllama(
        model=_config.MODEL_NAME, base_url=_config.OLLAMA_BASE_URL,
        temperature=0.2, num_predict=_config.MAX_TOKENS, timeout=_config.MCP_REQUEST_TIMEOUT,
    )

    context = f"User request: {state['user_input']}\n\n"

    available_tools = state.get("available_tools") or []
    if available_tools:
        context += "Available external tools:\n"
        for tool in available_tools:
            context += f"  - {tool['name']}: {tool.get('description', '')}\n"
        context += "\n"

    if state["agent_outputs"]:
        context += "Previous agent outputs:\n"
        for name, output in state["agent_outputs"].items():
            context += f"\n--- {name} ---\n{output}\n"
        context += "\nBased on the above, decide what to do next."
    else:
        context += "No agents have been called yet. Decide which agent should handle this first."

    system_prompt = _config.SUPERVISOR_SYSTEM_PROMPT + _tool_usage_instructions(available_tools)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ])

    decision = _parse_decision(response.content)

    next_choice = decision.get("next")
    if next_choice == "FINISH":
        state["final_answer"] = decision.get("final_answer", "Task completed.")
        state["next_agent"] = "FINISH"
    elif next_choice == USE_TOOL and available_tools:
        tool_name = decision.get("tool", "")
        known_tools = {t["name"] for t in available_tools}
        if tool_name in known_tools:
            state["pending_tool"] = tool_name
            state["pending_args"] = decision.get("args", {}) or {}
            state["next_agent"] = USE_TOOL
        else:
            # Hallucinated tool name — finish gracefully rather than loop.
            state["final_answer"] = f"Unable to route: no external tool named '{tool_name}'."
            state["next_agent"] = "FINISH"
    else:
        next_name = next_choice or "FINISH"
        if next_name not in AGENT_NAMES:
            # Supervisor hallucinated an invalid agent name — finish gracefully
            state["final_answer"] = decision.get(
                "reason",
                decision.get("final_answer", f"Unable to route: no agent named '{next_name}'."),
            )
            state["next_agent"] = "FINISH"
        else:
            state["next_agent"] = next_name

    state["iteration"] = state.get("iteration", 0) + 1
    return state


def _parse_decision(content: str) -> dict:
    """Parse the supervisor's JSON decision, tolerating surrounding prose."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        return {"next": "FINISH", "final_answer": content}


def _make_agent_node(agent_name: str):
    """Create a graph node for any agent."""
    def node(state: AgentState) -> AgentState:
        task = state["user_input"]
        if state["agent_outputs"]:
            task += f"\n\nContext from previous steps:\n{_format_previous_outputs(state['agent_outputs'])}"
        try:
            state["agent_outputs"][agent_name] = run_agent(agent_name, task)
        except Exception as e:
            state["agent_outputs"][agent_name] = f"[Agent '{agent_name}' failed: {e}]"
        return state
    return node


def _make_tool_node(tool_executor: ToolExecutor):
    """Create a node that executes the supervisor's requested external MCP tool."""
    def node(state: AgentState) -> AgentState:
        tool_name = state.get("pending_tool", "")
        args = state.get("pending_args", {}) or {}
        try:
            result = tool_executor(tool_name, args)
        except Exception as e:
            result = f"[Tool '{tool_name}' failed: {e}]"
        result = _truncate(str(result), MAX_TOOL_RESULT_CHARS)
        state["agent_outputs"][f"tool:{tool_name}"] = result
        state.setdefault("tool_calls", []).append(
            {"tool": tool_name, "args": args, "result": result}
        )
        # Clear the pending request so we don't re-run the same tool.
        state["pending_tool"] = ""
        state["pending_args"] = {}
        return state
    return node


def route_from_supervisor(state: AgentState) -> str:
    """Route to the next agent/tool based on supervisor's decision."""
    if state.get("iteration", 0) >= 5:
        # Max iterations reached — ensure we have a final answer
        if not state.get("final_answer"):
            # Synthesize from the last agent output
            if state["agent_outputs"]:
                last_agent = list(state["agent_outputs"].keys())[-1]
                state["final_answer"] = state["agent_outputs"][last_agent]
            else:
                state["final_answer"] = "Max iterations reached without producing a result."
        return "__end__"
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == USE_TOOL:
        return TOOL_NODE
    if next_agent in AGENT_NAMES:
        return next_agent
    return "__end__"


def build_agent_graph(tool_executor: Optional[ToolExecutor] = None) -> StateGraph:
    """Construct and compile the multi-agent graph.

    Args:
        tool_executor: Optional synchronous callable ``(tool_name, args) -> str`` that
            executes an external MCP tool. When provided, the supervisor may route to a
            tool-execution node. When ``None`` (e.g. the CLI path with no external
            servers configured), tool routing is disabled entirely.
    """
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")

    for name in AGENT_NAMES:
        graph.add_node(name, _make_agent_node(name))
        graph.add_edge(name, "supervisor")

    routing_map = {name: name for name in AGENT_NAMES}
    routing_map["__end__"] = END

    if tool_executor is not None:
        graph.add_node(TOOL_NODE, _make_tool_node(tool_executor))
        graph.add_edge(TOOL_NODE, "supervisor")
        routing_map[TOOL_NODE] = TOOL_NODE

    graph.add_conditional_edges("supervisor", route_from_supervisor, routing_map)

    return graph.compile()
