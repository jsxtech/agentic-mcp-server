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
