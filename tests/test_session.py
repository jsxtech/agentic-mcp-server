"""Tests for session management: history, token tracking, save/load/export."""

import pytest

import session
from session import (
    ConversationHistory,
    TokenTracker,
    save_session,
    load_session,
    list_sessions,
    export_session,
)


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """Redirect the sessions directory to a temp path for isolation."""
    d = tmp_path / "sessions"
    monkeypatch.setattr(session, "SESSIONS_DIR", d)
    return d


class TestConversationHistory:
    def test_add_turn_records_fields(self):
        h = ConversationHistory()
        h.add_turn("user", "hello")
        h.add_turn("assistant", "hi", agent="coder")
        assert len(h.turns) == 2
        assert h.turns[0]["role"] == "user"
        assert h.turns[1]["agent"] == "coder"
        assert "timestamp" in h.turns[0]

    def test_get_context_empty(self):
        assert ConversationHistory().get_context() == ""

    def test_get_context_truncates_content(self):
        h = ConversationHistory()
        h.add_turn("user", "x" * 500)
        ctx = h.get_context()
        assert "x" * 200 in ctx
        assert "x" * 201 not in ctx

    def test_get_context_respects_max_turns(self):
        h = ConversationHistory()
        for i in range(20):
            h.add_turn("user", f"msg{i}")
        ctx = h.get_context(max_turns=5)
        assert "msg19" in ctx
        assert "msg14" not in ctx

    def test_last_user_message(self):
        h = ConversationHistory()
        h.add_turn("user", "first")
        h.add_turn("assistant", "reply", agent="coder")
        h.add_turn("user", "second")
        assert h.last_user_message() == "second"

    def test_last_user_message_none(self):
        assert ConversationHistory().last_user_message() is None

    def test_roundtrip_dict(self):
        h = ConversationHistory()
        h.add_turn("user", "hi")
        restored = ConversationHistory.from_dict(h.to_dict())
        assert restored.turns == h.turns


class TestTokenTracker:
    def test_track_accumulates(self):
        t = TokenTracker()
        t.track("a" * 40, "b" * 80)
        assert t.total_input == 10   # 40 // 4
        assert t.total_output == 20  # 80 // 4
        assert t.requests == 1

    def test_summary_format(self):
        t = TokenTracker()
        t.track("a" * 40, "b" * 40)
        s = t.summary()
        assert "Requests: 1" in s
        assert "total" in s


class TestSessionPersistence:
    def test_save_and_load(self, sessions_dir):
        h = ConversationHistory()
        h.add_turn("user", "remember me")
        path = save_session(h, "mysession")
        assert path.endswith("mysession.json")
        loaded = load_session("mysession")
        assert loaded.turns[0]["content"] == "remember me"

    def test_load_missing_raises(self, sessions_dir):
        sessions_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_session("nope")

    def test_load_corrupted_raises(self, sessions_dir):
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "bad.json").write_text("{not valid json")
        with pytest.raises(FileNotFoundError):
            load_session("bad")

    def test_path_traversal_sanitized_on_save(self, sessions_dir):
        h = ConversationHistory()
        h.add_turn("user", "x")
        path = save_session(h, "../../etc/passwd")
        assert str(sessions_dir) in path
        assert "etc/passwd" not in path

    def test_list_sessions(self, sessions_dir):
        h = ConversationHistory()
        h.add_turn("user", "x")
        save_session(h, "s1")
        save_session(h, "s2")
        assert list_sessions() == ["s1", "s2"]

    def test_list_sessions_empty_no_dir(self, sessions_dir):
        assert list_sessions() == []

    def test_export_markdown(self, sessions_dir):
        h = ConversationHistory()
        h.add_turn("user", "question")
        h.add_turn("assistant", "answer", agent="coder")
        path = export_session(h)
        content = (sessions_dir / path.split("/")[-1]).read_text()
        assert "## 🧑 User" in content
        assert "question" in content
        assert "coder" in content
