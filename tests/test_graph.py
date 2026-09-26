"""Tests for the LangGraph orchestration: routing, decision parsing, nodes, tools."""

import graph
from graph import (
    AgentState,
    route_from_supervisor,
    _parse_decision,
    _truncate,
    _make_agent_node,
    _make_tool_node,
    _tool_usage_instructions,
    build_agent_graph,
    USE_TOOL,
    TOOL_NODE,
    MAX_TOOL_RESULT_CHARS,
)


def _state(**over) -> AgentState:
    base: AgentState = {
        "user_input": "hi",
        "agent_outputs": {},
        "next_agent": "",
        "final_answer": "",
        "iteration": 0,
        "tool_calls": [],
        "available_tools": [],
        "pending_tool": "",
        "pending_args": {},
    }
    base.update(over)
    return base


class TestTruncate:
    def test_short_unchanged(self):
        assert _truncate("abc", 10) == "abc"

    def test_long_truncated_with_marker(self):
        out = _truncate("x" * 100, 10)
        assert out.startswith("x" * 10)
        assert "truncated" in out


class TestParseDecision:
    def test_clean_json(self):
        assert _parse_decision('{"next": "coder"}') == {"next": "coder"}

    def test_json_embedded_in_prose(self):
        content = 'Sure, here is my decision: {"next": "coder", "reason": "x"} done.'
        assert _parse_decision(content)["next"] == "coder"

    def test_unparseable_falls_back_to_finish(self):
        d = _parse_decision("no json here at all")
        assert d["next"] == "FINISH"
        assert d["final_answer"] == "no json here at all"


class TestRouteFromSupervisor:
    def test_routes_to_valid_agent(self):
        assert route_from_supervisor(_state(next_agent="coder")) == "coder"

    def test_finish_routes_to_end(self):
        assert route_from_supervisor(_state(next_agent="FINISH")) == "__end__"

    def test_invalid_agent_routes_to_end(self):
        assert route_from_supervisor(_state(next_agent="bogus")) == "__end__"

    def test_use_tool_routes_to_tool_node(self):
        assert route_from_supervisor(_state(next_agent=USE_TOOL)) == TOOL_NODE

    def test_max_iterations_ends_and_synthesizes(self):
        st = _state(iteration=5, agent_outputs={"coder": "the answer"})
        assert route_from_supervisor(st) == "__end__"
        assert st["final_answer"] == "the answer"

    def test_max_iterations_no_outputs(self):
        st = _state(iteration=5)
        route_from_supervisor(st)
        assert "Max iterations" in st["final_answer"]


class TestAgentNode:
    def test_node_stores_output(self, monkeypatch):
        monkeypatch.setattr(graph, "run_agent", lambda n, t: f"out-{n}")
        node = _make_agent_node("coder")
        st = node(_state())
        assert st["agent_outputs"]["coder"] == "out-coder"

    def test_node_captures_exception(self, monkeypatch):
        def _boom(n, t):
            raise RuntimeError("fail")

        monkeypatch.setattr(graph, "run_agent", _boom)
        st = _make_agent_node("coder")(_state())
        assert "failed" in st["agent_outputs"]["coder"]

    def test_node_truncates_prior_context(self, monkeypatch):
        captured = {}

        def _fake(n, t):
            captured["task"] = t
            return "ok"

        monkeypatch.setattr(graph, "run_agent", _fake)
        big_outputs = {"prev": "z" * 20000}
        _make_agent_node("coder")(_state(agent_outputs=big_outputs))
        # The prior context fed into the task must be bounded.
        assert "truncated" in captured["task"]
        assert len(captured["task"]) < 20000 + 1000


class TestToolNode:
    def test_tool_executes_and_records(self):
        node = _make_tool_node(lambda name, args: f"result of {name}")
        st = _state(pending_tool="search", pending_args={"q": "x"})
        out = node(st)
        assert out["agent_outputs"]["tool:search"] == "result of search"
        assert out["tool_calls"][0]["tool"] == "search"
        assert out["pending_tool"] == ""  # cleared

    def test_tool_failure_captured(self):
        def _boom(name, args):
            raise RuntimeError("nope")

        st = _state(pending_tool="broken", pending_args={})
        out = _make_tool_node(_boom)(st)
        assert "failed" in out["agent_outputs"]["tool:broken"]

    def test_tool_result_truncated(self):
        node = _make_tool_node(lambda n, a: "q" * (MAX_TOOL_RESULT_CHARS + 5000))
        out = node(_state(pending_tool="big"))
        assert "truncated" in out["agent_outputs"]["tool:big"]


class TestToolUsageInstructions:
    def test_empty_when_no_tools(self):
        assert _tool_usage_instructions([]) == ""

    def test_lists_tools(self):
        instr = _tool_usage_instructions([{"name": "fs", "description": "files"}])
        assert "use_tool" in instr
        assert "fs" in instr


class TestBuildGraph:
    def test_no_tool_node_without_executor(self):
        g = build_agent_graph()
        assert TOOL_NODE not in g.get_graph().nodes

    def test_tool_node_present_with_executor(self):
        g = build_agent_graph(tool_executor=lambda n, a: "ok")
        assert TOOL_NODE in g.get_graph().nodes


class TestSupervisorNode:
    """Supervisor node with the LLM mocked out."""

    def _mock_llm(self, monkeypatch, response_content):
        class _Resp:
            content = response_content

        class _LLM:
            def __init__(self, *a, **k):
                pass

            def invoke(self, messages):
                return _Resp()

        monkeypatch.setattr(graph, "ChatOllama", _LLM)

    def test_finish_decision(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "FINISH", "final_answer": "done"}')
        st = graph.supervisor_node(_state())
        assert st["next_agent"] == "FINISH"
        assert st["final_answer"] == "done"

    def test_route_to_agent(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "coder", "reason": "code task"}')
        st = graph.supervisor_node(_state())
        assert st["next_agent"] == "coder"

    def test_hallucinated_agent_finishes(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "wizard", "reason": "magic"}')
        st = graph.supervisor_node(_state())
        assert st["next_agent"] == "FINISH"

    def test_use_tool_decision(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "use_tool", "tool": "fs", "args": {"p": "/tmp"}}')
        st = graph.supervisor_node(_state(available_tools=[{"name": "fs", "description": "files"}]))
        assert st["next_agent"] == USE_TOOL
        assert st["pending_tool"] == "fs"
        assert st["pending_args"] == {"p": "/tmp"}

    def test_use_tool_unknown_tool_finishes(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "use_tool", "tool": "ghost", "args": {}}')
        st = graph.supervisor_node(_state(available_tools=[{"name": "fs", "description": "files"}]))
        assert st["next_agent"] == "FINISH"

    def test_iteration_increments(self, monkeypatch):
        self._mock_llm(monkeypatch, '{"next": "FINISH", "final_answer": "x"}')
        st = graph.supervisor_node(_state(iteration=2))
        assert st["iteration"] == 3
