"""Tests for the CLI command dispatcher in main.py."""

import config
import main
from session import ConversationHistory, TokenTracker


class _FakeMemory:
    def __init__(self):
        self.notes = {}

    def add(self, key, note):
        self.notes.setdefault(key, []).append(note)

    def get(self, key):
        return self.notes.get(key, [])

    def get_all(self):
        return self.notes

    def clear(self, key=None):
        if key:
            self.notes.pop(key, None)
        else:
            self.notes.clear()


def _handle(cmd):
    return main._handle_command(cmd, ConversationHistory(), TokenTracker(), _FakeMemory())


class TestCommandDispatch:
    def test_exact_command_handled(self):
        assert _handle("/help") is True
        assert _handle("/agents") is True

    def test_unknown_command_not_handled(self):
        assert _handle("/totallybogus") is False

    def test_prefix_typo_not_misrouted_to_model(self):
        # Regression: "/models" must NOT be silently routed to "/model" (which
        # previously set the model name to the leftover text "s").
        before = config.MODEL_NAME
        handled = _handle("/models")
        assert handled is False
        assert config.MODEL_NAME == before  # unchanged

    def test_file_vs_files_distinct(self):
        # Both are real commands and must dispatch to their own handlers.
        assert _handle("/file") is True   # prints usage, still "handled"
        assert _handle("/files") is True

    def test_model_switch_updates_config(self):
        original = config.MODEL_NAME
        try:
            assert _handle("/model some-model:latest") is True
            assert config.MODEL_NAME == "some-model:latest"
        finally:
            config.MODEL_NAME = original

    def test_remember_stores_note(self):
        history = ConversationHistory()
        tracker = TokenTracker()
        mem = _FakeMemory()
        main._handle_command("/remember proj using postgres", history, tracker, mem)
        assert mem.get("proj") == ["using postgres"]


class TestProcessInput:
    """H3: a failing LLM call in ANY command must not crash the REPL."""

    def _ctx(self):
        return ConversationHistory(), TokenTracker(), _FakeMemory()

    def test_empty_input_keeps_running(self):
        h, t, m = self._ctx()
        assert main._process_input("", h, t, m) is True

    def test_quit_stops(self):
        h, t, m = self._ctx()
        assert main._process_input("quit", h, t, m) is False
        assert main._process_input("exit", h, t, m) is False

    def test_command_exception_does_not_escape(self, monkeypatch, capsys):
        # Simulate Ollama down during a slash command (/ask -> run_direct -> run_agent).
        import runners
        monkeypatch.setattr(runners, "run_agent", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        h, t, m = self._ctx()
        # Must return True (keep running) and print a friendly error, not raise.
        assert main._process_input("/ask coder do it", h, t, m) is True
        assert "❌ Error" in capsys.readouterr().out

    def test_normal_flow_exception_does_not_escape(self, monkeypatch, capsys):
        monkeypatch.setattr(main, "run_multi_agent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        h, t, m = self._ctx()
        assert main._process_input("just a normal request", h, t, m) is True
        assert "❌ Error" in capsys.readouterr().out


class TestBatchTracking:
    """M4: batch token tracking must not mis-associate the error-sentinel result."""

    def test_error_sentinel_not_mistracked(self, monkeypatch):
        # Unknown agent -> run_batch returns a 1-element ❌ list while tasks has 3.
        import runners
        h = ConversationHistory()
        tracker = TokenTracker()
        main._cmd_batch("bogus_agent t1;;t2;;t3", h, tracker, _FakeMemory())
        # One tracked request (the error), input not mis-mapped to a real task.
        assert tracker.requests == 1

    def test_aligned_results_tracked_per_task(self, monkeypatch):
        import runners
        monkeypatch.setattr(runners, "run_agent", lambda name, task: f"done: {task}")
        h = ConversationHistory()
        tracker = TokenTracker()
        main._cmd_batch("coder a;;b;;c", h, tracker, _FakeMemory())
        assert tracker.requests == 3
