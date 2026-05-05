"""Session management — conversation history, save/load, export."""

import json
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path("sessions")


class ConversationHistory:
    """Tracks conversation turns for context."""

    def __init__(self):
        self.turns: list[dict] = []

    def add_turn(self, role: str, content: str, agent: str = None):
        self.turns.append({"role": role, "content": content, "agent": agent, "timestamp": datetime.now().isoformat()})

    def get_context(self, max_turns: int = 10) -> str:
        recent = self.turns[-max_turns:]
        if not recent:
            return ""
        lines = ["Previous conversation:"]
        for t in recent:
            prefix = f"[{t['agent']}]" if t.get("agent") else "[user]"
            lines.append(f"  {prefix}: {t['content'][:200]}")
        return "\n".join(lines)

    def last_user_message(self) -> str | None:
        return next((t["content"] for t in reversed(self.turns) if t["role"] == "user"), None)

    def to_dict(self) -> list[dict]:
        return self.turns

    @classmethod
    def from_dict(cls, data: list[dict]) -> "ConversationHistory":
        h = cls()
        h.turns = data
        return h


class TokenTracker:
    """Tracks approximate token usage."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.requests = 0

    def track(self, input_text: str, output_text: str):
        self.total_input += len(input_text) // 4
        self.total_output += len(output_text) // 4
        self.requests += 1

    def summary(self) -> str:
        total = self.total_input + self.total_output
        return f"Tokens: ~{total:,} total ({self.total_input:,} in / {self.total_output:,} out) | Requests: {self.requests}"


def save_session(history: ConversationHistory, name: str = None) -> str:
    SESSIONS_DIR.mkdir(exist_ok=True)
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{name}.json"
    path.write_text(json.dumps(history.to_dict(), indent=2))
    return str(path)


def load_session(name: str) -> ConversationHistory:
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        path = SESSIONS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {name}")
    return ConversationHistory.from_dict(json.loads(path.read_text()))


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def export_session(history: ConversationHistory) -> str:
    SESSIONS_DIR.mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{name}.md"
    lines = [f"# Conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for t in history.turns:
        if t["role"] == "user":
            lines.append(f"## 🧑 User\n\n{t['content']}\n")
        else:
            lines.append(f"## 🤖 {t.get('agent') or 'assistant'}\n\n{t['content']}\n")
    path.write_text("\n".join(lines))
    return str(path)
