"""Multi-agent orchestration graph using LangGraph."""

import json
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agent_registry import run_agent, AGENT_NAMES
import config as _config


class AgentState(TypedDict):
    """State that flows through the agent graph."""
    user_input: str
    agent_outputs: dict[str, str]
    next_agent: str
    final_answer: str
    iteration: int
    tool_calls: list[dict]
    available_tools: list[dict]


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor decides which agent to call next or finishes."""
    llm = ChatOllama(
        model=_config.MODEL_NAME, base_url=_config.OLLAMA_BASE_URL,
        temperature=0.2, num_predict=_config.MAX_TOKENS, timeout=_config.MCP_REQUEST_TIMEOUT,
    )

    context = f"User request: {state['user_input']}\n\n"

    if state.get("available_tools"):
        context += "Available external tools:\n"
        for tool in state["available_tools"]:
            context += f"  - {tool['name']}: {tool['description']}\n"
        context += "\n"

    if state["agent_outputs"]:
        context += "Previous agent outputs:\n"
        for name, output in state["agent_outputs"].items():
            context += f"\n--- {name} ---\n{output}\n"
        context += "\nBased on the above, decide what to do next."
    else:
        context += "No agents have been called yet. Decide which agent should handle this first."

    response = llm.invoke([
        SystemMessage(content=_config.SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                decision = json.loads(content[start:end])
            except json.JSONDecodeError:
                decision = {"next": "FINISH", "final_answer": content}
        else:
            decision = {"next": "FINISH", "final_answer": content}

    if decision.get("next") == "FINISH":
        state["final_answer"] = decision.get("final_answer", "Task completed.")
        state["next_agent"] = "FINISH"
    else:
        next_name = decision.get("next", "FINISH")
        if next_name not in AGENT_NAMES:
            # Supervisor hallucinated an invalid agent name — finish gracefully
            state["final_answer"] = decision.get("reason", decision.get("final_answer", f"Unable to route: no agent named '{next_name}'."))
            state["next_agent"] = "FINISH"
        else:
            state["next_agent"] = next_name

    state["iteration"] = state.get("iteration", 0) + 1
    return state


def _make_agent_node(agent_name: str):
    """Create a graph node for any agent."""
    def node(state: AgentState) -> AgentState:
        task = state["user_input"]
        if state["agent_outputs"]:
            task += f"\n\nContext from previous steps:\n{json.dumps(state['agent_outputs'], indent=2)}"
        try:
            state["agent_outputs"][agent_name] = run_agent(agent_name, task)
        except Exception as e:
            state["agent_outputs"][agent_name] = f"[Agent '{agent_name}' failed: {e}]"
        return state
    return node


def route_from_supervisor(state: AgentState) -> str:
    """Route to the next agent based on supervisor's decision."""
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
    if next_agent in AGENT_NAMES:
        return next_agent
    return "__end__"


def build_agent_graph(tool_registry=None) -> StateGraph:
    """Construct and compile the multi-agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")

    for name in AGENT_NAMES:
        graph.add_node(name, _make_agent_node(name))
        graph.add_edge(name, "supervisor")

    routing_map = {name: name for name in AGENT_NAMES}
    routing_map["__end__"] = END
    graph.add_conditional_edges("supervisor", route_from_supervisor, routing_map)

    return graph.compile()
