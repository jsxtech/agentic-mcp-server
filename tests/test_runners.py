"""Tests for runners: routing, workflows, chain/parallel/compare, memory, file context.

``run_agent`` is patched so tests never touch Ollama.
"""

import pytest

import runners
from runners import (
    WORKFLOWS,
    run_direct,
    run_chain,
    run_parallel,
    run_compare,
    run_workflow,
    run_batch,
    run_conditional,
    run_feedback_loop,
    load_file_context,
    load_multi_file_context,
    AgentMemory,
    invalidate_graph_cache,
    _is_approved,
)
from session import ConversationHistory


@pytest.fixture
def fake_agent(monkeypatch):
    """Patch run_agent to echo the agent name and task deterministically."""
    calls = []

    def _fake(name, task):
        calls.append((name, task))
        return f"[{name}] handled: {task[:40]}"

    monkeypatch.setattr(runners, "run_agent", _fake)
    return calls


@pytest.fixture
def history():
    return ConversationHistory()


class TestRunDirect:
    def test_unknown_agent(self, fake_agent, history):
        out = run_direct("nonexistent", "task", history)
        assert out.startswith("❌ Unknown agent")

    def test_valid_agent(self, fake_agent, history):
        out = run_direct("coder", "write code", history)
        assert "[coder]" in out


class TestRunChain:
    def test_invalid_agent_in_chain(self, fake_agent, history):
        out = run_chain("coder|bogus", "task", history)
        assert "❌ Unknown agent(s)" in out
        assert "bogus" in out

    def test_chain_sequential(self, fake_agent, history):
        out = run_chain("coder|reviewer", "build", history)
        # Final output is the last agent in the chain
        assert "[reviewer]" in out
        # Both agents ran
        names = [c[0] for c in fake_agent]
        assert names == ["coder", "reviewer"]

    def test_chain_truncates_long_input(self, monkeypatch, history):
        seen = {}

        def _fake(name, task):
            seen[name] = task
            # Return a huge output to force truncation on the next hop
            return "y" * 20000

        monkeypatch.setattr(runners, "run_agent", _fake)
        run_chain("coder|reviewer", "start", history)
        # reviewer's input must have been truncated below the ~8000 budget + marker
        assert "truncated" in seen["reviewer"]


class TestRunParallel:
    def test_empty_list(self, fake_agent, history):
        assert run_parallel([], "task", history) == {}

    def test_parallel_all_run(self, fake_agent, history):
        results = run_parallel(["coder", "tester"], "task", history)
        assert set(results.keys()) == {"coder", "tester"}
        assert all("handled" in v for v in results.values())

    def test_parallel_captures_errors(self, monkeypatch, history):
        def _boom(name, task):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(runners, "run_agent", _boom)
        results = run_parallel(["coder"], "task", history)
        assert "Error:" in results["coder"]


class TestRunCompare:
    def test_invalid(self, fake_agent, history):
        out = run_compare("coder,bogus", "task", history)
        assert "❌ Unknown agent(s)" in out

    def test_compare_formats_sections(self, fake_agent, history):
        out = run_compare("coder,tester", "task", history)
        assert "━━━ CODER ━━━" in out
        assert "━━━ TESTER ━━━" in out


class TestRunWorkflow:
    def test_unknown_workflow(self, fake_agent, history):
        out = run_workflow("no_such_flow", "task", history)
        assert "❌ Unknown workflow" in out

    def test_known_workflow_runs_all_agents(self, fake_agent, history):
        run_workflow("bug_fix", "fix it", history)
        names = [c[0] for c in fake_agent]
        assert names == WORKFLOWS["bug_fix"]  # debugger, coder, tester

    def test_all_workflow_agents_are_valid(self):
        from agent_registry import AGENT_NAMES
        for wf, agents in WORKFLOWS.items():
            for a in agents:
                assert a in AGENT_NAMES, f"workflow {wf} references unknown agent {a}"


class TestRunBatch:
    def test_unknown_agent(self, fake_agent, history):
        out = run_batch("bogus", ["t1"], history)
        assert out[0].startswith("❌ Unknown agent")

    def test_empty_tasks(self, fake_agent, history):
        out = run_batch("coder", [], history)
        assert out == ["❌ No tasks provided."]

    def test_batch_runs_each_task(self, fake_agent, history):
        out = run_batch("coder", ["a", "b", "c"], history)
        assert len(out) == 3


class TestRunConditional:
    def test_keyword_selects_bug_fix(self, fake_agent, history):
        run_conditional("there is a bug causing a crash", history)
        names = [c[0] for c in fake_agent]
        assert names == WORKFLOWS["bug_fix"]

    def test_default_full_dev_when_no_keywords(self, fake_agent, history):
        run_conditional("say hello to the world", history)
        names = [c[0] for c in fake_agent]
        assert names == WORKFLOWS["full_dev"]

    def test_word_boundary_avoids_false_positive(self, fake_agent, history):
        # "build" contains "ui" but must NOT trigger the frontend workflow,
        # and contains no other keyword, so it falls back to full_dev.
        run_conditional("build me a widget", history)
        names = [c[0] for c in fake_agent]
        assert names == WORKFLOWS["full_dev"]

    def test_word_boundary_still_matches_real_keyword(self, fake_agent, history):
        run_conditional("improve the react ui component", history)
        names = [c[0] for c in fake_agent]
        assert names == WORKFLOWS["frontend"]

    def test_security_keywords_win(self, fake_agent, history):
        run_conditional("find the sql injection vulnerability", history)
        names = [c[0] for c in fake_agent]
        # security_audit and db_design both match; security has higher weight
        assert names == WORKFLOWS["security_audit"]


class TestFeedbackLoop:
    def test_invalid_agent(self, fake_agent, history):
        out = run_feedback_loop("coder", "task", "bogus", history)
        assert "❌ Unknown agent(s)" in out

    def test_approved_stops_early(self, monkeypatch, history):
        def _fake(name, task):
            if name == "reviewer":
                return "Looks great. APPROVED"
            return "some code"

        monkeypatch.setattr(runners, "run_agent", _fake)
        out = run_feedback_loop("coder", "task", "reviewer", history, max_rounds=3)
        assert out == "some code"

    def test_max_rounds_reached(self, monkeypatch, history):
        rounds = {"coder": 0}

        def _fake(name, task):
            if name == "reviewer":
                return "needs work, try again"
            rounds["coder"] += 1
            return f"attempt {rounds['coder']}"

        monkeypatch.setattr(runners, "run_agent", _fake)
        out = run_feedback_loop("coder", "task", "reviewer", history, max_rounds=2)
        assert rounds["coder"] == 2
        assert "attempt 2" in out

    def test_negated_approval_does_not_stop(self, monkeypatch, history):
        # "NOT APPROVED" must not be treated as approval.
        calls = {"coder": 0}

        def _fake(name, task):
            if name == "reviewer":
                return "This is NOT APPROVED yet, please fix."
            calls["coder"] += 1
            return "attempt"

        monkeypatch.setattr(runners, "run_agent", _fake)
        run_feedback_loop("coder", "task", "reviewer", history, max_rounds=2)
        assert calls["coder"] == 2  # ran all rounds, never early-stopped

    def test_verdict_approved_stops(self, monkeypatch, history):
        def _fake(name, task):
            if name == "reviewer":
                return "Looks solid.\nVERDICT: APPROVED"
            return "the code"

        monkeypatch.setattr(runners, "run_agent", _fake)
        out = run_feedback_loop("coder", "task", "reviewer", history, max_rounds=3)
        assert out == "the code"


class TestFileContext:
    def test_load_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("print('hi')")
        ctx = load_file_context(str(f))
        assert "code.py" in ctx
        assert "print('hi')" in ctx

    def test_load_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_file_context("/no/such/file.xyz")

    def test_load_file_non_utf8_does_not_crash(self, tmp_path):
        # Regression: binary / invalid-UTF8 content must not raise UnicodeDecodeError.
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\xff\xfe valid-ish \x80\x81 text")
        ctx = load_file_context(str(f))
        assert "blob.bin" in ctx

    def test_load_file_truncates(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("z" * 20000)
        ctx = load_file_context(str(f))
        assert "truncated" in ctx

    def test_multi_file_partial(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("aaa")
        ctx = load_multi_file_context([str(f1), str(tmp_path / "missing.py")])
        assert "a.py" in ctx
        assert "File not found" in ctx

    def test_multi_file_all_missing_returns_markers(self, tmp_path):
        # load_multi_file_context is intentionally lenient: missing files become
        # inline markers. It only raises when the input list itself is empty.
        ctx = load_multi_file_context([str(tmp_path / "x.py"), str(tmp_path / "y.py")])
        assert ctx.count("File not found") == 2

    def test_multi_file_empty_list_raises(self):
        with pytest.raises(FileNotFoundError):
            load_multi_file_context([])


class TestAgentMemory:
    def test_add_get_clear(self, tmp_path):
        mem = AgentMemory(path=tmp_path / "mem.json")
        mem.add("proj", "note one")
        mem.add("proj", "note two")
        assert mem.get("proj") == ["note one", "note two"]
        assert mem.get_all() == {"proj": ["note one", "note two"]}
        mem.clear("proj")
        assert mem.get("proj") == []

    def test_persistence_across_instances(self, tmp_path):
        p = tmp_path / "mem.json"
        AgentMemory(path=p).add("k", "v")
        assert AgentMemory(path=p).get("k") == ["v"]

    def test_corrupted_memory_resets(self, tmp_path):
        p = tmp_path / "mem.json"
        p.write_text("{broken")
        mem = AgentMemory(path=p)
        assert mem.get_all() == {}

    def test_clear_all(self, tmp_path):
        mem = AgentMemory(path=tmp_path / "mem.json")
        mem.add("a", "1")
        mem.add("b", "2")
        mem.clear()
        assert mem.get_all() == {}

    def test_nested_parent_dirs_created(self, tmp_path):
        # Regression: _save must create intermediate directories, not crash.
        mem = AgentMemory(path=tmp_path / "deep" / "nested" / "mem.json")
        mem.add("k", "v")
        assert (tmp_path / "deep" / "nested" / "mem.json").exists()
        assert AgentMemory(path=tmp_path / "deep" / "nested" / "mem.json").get("k") == ["v"]

    def test_get_all_returns_copy(self, tmp_path):
        # Regression: mutating the returned dict must not corrupt internal state.
        mem = AgentMemory(path=tmp_path / "mem.json")
        mem.add("k", "v")
        snapshot = mem.get_all()
        snapshot["k"].append("injected")
        snapshot["new"] = ["x"]
        assert mem.get("k") == ["v"]
        assert "new" not in mem.get_all()


class TestIsApproved:
    def test_verdict_approved(self):
        assert _is_approved("VERDICT: APPROVED") is True

    def test_verdict_revise(self):
        assert _is_approved("VERDICT: REVISE\nfix the bug") is False

    def test_not_approved(self):
        assert _is_approved("This is not approved.") is False

    def test_unapproved(self):
        assert _is_approved("UNAPPROVED") is False

    def test_plain_approved_fallback(self):
        assert _is_approved("Looks good, approved!") is True

    def test_no_approval_word(self):
        assert _is_approved("needs more work") is False


class TestGraphCache:
    def test_invalidate_resets_cache(self, monkeypatch):
        built = {"count": 0}

        def _fake_build(*args, **kwargs):
            built["count"] += 1
            return object()

        monkeypatch.setattr(runners, "build_agent_graph", _fake_build)
        invalidate_graph_cache()
        runners._get_graph()
        runners._get_graph()
        assert built["count"] == 1  # cached
        invalidate_graph_cache()
        runners._get_graph()
        assert built["count"] == 2  # rebuilt after invalidation
