"""Main entry point for the multi-agent application."""

import argparse
import asyncio
import logging
import time

import requests

from agent_registry import AGENT_NAMES
from config import MCP_SERVER_TRANSPORT, MCP_SSE_HOST, MCP_SSE_PORT, EXTERNAL_MCP_SERVERS, OLLAMA_BASE_URL, MODEL_NAME
from session import ConversationHistory, TokenTracker, save_session, load_session, list_sessions, export_session
from runners import (
    run_multi_agent, run_direct, run_chain, run_parallel,
    run_compare, run_workflow, run_streaming, load_file_context, WORKFLOWS,
    run_batch, run_conditional, AgentMemory, run_feedback_loop, load_multi_file_context,
)
from tool_registry import ToolRegistry, ExternalServerConfig


def check_ollama_health() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(MODEL_NAME in m for m in models):
            print(f"⚠️  Model '{MODEL_NAME}' not found. Available: {models}")
            return False
        return True
    except requests.ConnectionError:
        return False


HELP = """
Commands:
  /ask <agent> msg     Run a specific agent directly
  /chain <a|b|c> msg  Chain agents in sequence
  /compare <a,b> msg  Compare multiple agents on same task
  /parallel <a,b> msg Run agents concurrently (faster)
  /workflow <name> msg Run a predefined pipeline
  /auto msg            Auto-select best workflow based on keywords
  /batch <agent> msg1;;msg2  Run agent on multiple tasks in parallel
  /feedback <agent> <reviewer> msg  Iterative refinement loop
  /files <p1,p2> msg   Load multiple files as context
  /workflows           List available workflows
  /file <path> msg     Load file as context for the request
  /stream msg          Stream a direct response
  /retry               Re-run the last request
  /remember <key> note Store a note in agent memory
  /recall [key]        Recall stored notes
  /forget [key]        Clear memory (all or by key)
  /tokens              Show token usage stats
  /model <name>        Switch Ollama model at runtime
  /agents              List all available agents
  /save [name]         Save session  |  /load <name>  Load session
  /export              Export as markdown
  /sessions            List saved sessions
  /history             Show history  |  /clear  Clear history
  /health              Check Ollama  |  /help   This help
  quit / exit          Exit
"""


def _handle_command(cmd: str, history: ConversationHistory, tracker: TokenTracker, memory: AgentMemory) -> bool:
    """Handle a slash command. Returns True if handled."""
    if cmd == "/help":
        print(HELP)
    elif cmd == "/agents":
        for name in AGENT_NAMES: print(f"  • {name}")
    elif cmd == "/health":
        print("✅ OK" if check_ollama_health() else "❌ Not reachable")
    elif cmd == "/history":
        for t in (history.turns[-20:] or [{"content": "(empty)", "agent": None}]):
            print(f"  [{t.get('agent','user')}] {t['content'][:100]}")
    elif cmd == "/clear":
        history.turns.clear(); print("  ✓ Cleared")
    elif cmd == "/sessions":
        for s in (list_sessions() or ["(none)"]): print(f"  • {s}")
    elif cmd == "/tokens":
        print(f"  {tracker.summary()}")
    elif cmd == "/workflows":
        for name, agents in WORKFLOWS.items(): print(f"  • {name}: {' → '.join(agents)}")
    elif cmd == "/export":
        print(f"  ✓ Exported to {export_session(history)}")
    elif cmd.startswith("/save"):
        print(f"  ✓ Saved to {save_session(history, cmd[5:].strip() or None)}")
    elif cmd.startswith("/load "):
        try:
            history.turns = load_session(cmd[6:].strip()).turns
            print(f"  ✓ Loaded ({len(history.turns)} turns)")
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
    elif cmd.startswith("/model "):
        import config; config.MODEL_NAME = cmd[7:].strip()
        print(f"  ✓ Switched to: {config.MODEL_NAME}")
    elif cmd == "/retry":
        last = history.last_user_message()
        if not last: print("  ❌ No previous request"); return True
        _run_and_print(last, history, tracker)
    elif cmd.startswith("/ask "):
        parts = cmd[5:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /ask <agent> msg"); return True
        name, msg = parts
        history.add_turn("user", msg)
        print(f"\n🤖 [{name}]...\n")
        start = time.time()
        result = run_direct(name, msg, history)
        print(f"📋 ({time.time()-start:.1f}s):\n\n{result}")
        history.add_turn("assistant", result, agent=name)
        tracker.track(msg, result)
    elif cmd.startswith("/chain "):
        parts = cmd[7:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /chain a|b|c msg"); return True
        history.add_turn("user", parts[1]); print()
        start = time.time()
        result = run_chain(parts[0], parts[1], history)
        print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
        tracker.track(parts[1], result)
    elif cmd.startswith("/compare "):
        parts = cmd[9:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /compare a,b msg"); return True
        history.add_turn("user", parts[1]); print()
        start = time.time()
        result = run_compare(parts[0], parts[1], history)
        print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
        tracker.track(parts[1], result)
    elif cmd.startswith("/parallel "):
        parts = cmd[10:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /parallel a,b msg"); return True
        agent_list = [a.strip() for a in parts[0].split(",")]
        invalid = [a for a in agent_list if a not in AGENT_NAMES]
        if invalid: print(f"  ❌ Unknown: {invalid}"); return True
        history.add_turn("user", parts[1]); print()
        start = time.time()
        results = run_parallel(agent_list, parts[1], history)
        for name, output in results.items():
            print(f"\n━━━ {name.upper()} ━━━\n{output}")
            history.add_turn("assistant", output, agent=name)
            tracker.track(parts[1], output)
        print(f"\n  ⏱ {time.time()-start:.1f}s total")
    elif cmd.startswith("/workflow "):
        parts = cmd[10:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /workflow <name> msg"); return True
        history.add_turn("user", parts[1]); print()
        start = time.time()
        result = run_workflow(parts[0], parts[1], history)
        print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
        tracker.track(parts[1], result)
    elif cmd.startswith("/file "):
        parts = cmd[6:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /file <path> msg"); return True
        try:
            ctx = load_file_context(parts[0])
            full_msg = f"{ctx}\n\n{parts[1]}"
            print(f"  📄 Loaded {parts[0]}")
            _run_and_print(full_msg, history, tracker)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
    elif cmd.startswith("/stream "):
        msg = cmd[8:].strip()
        history.add_turn("user", msg)
        print("\n🤖 ", end="")
        result = run_streaming(msg, history)
        history.add_turn("assistant", result)
        tracker.track(msg, result)
    elif cmd.startswith("/auto "):
        msg = cmd[6:].strip()
        history.add_turn("user", msg); print()
        start = time.time()
        result = run_conditional(msg, history)
        print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
        tracker.track(msg, result)
    elif cmd.startswith("/batch "):
        parts = cmd[7:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /batch <agent> task1;;task2;;task3"); return True
        agent_name, tasks_str = parts
        tasks = [t.strip() for t in tasks_str.split(";;") if t.strip()]
        print(f"  Running {agent_name} on {len(tasks)} tasks...\n")
        start = time.time()
        results = run_batch(agent_name, tasks, history)
        for i, r in enumerate(results):
            print(f"━━━ Task {i+1} ━━━\n{r}\n")
            tracker.track(tasks[i], r)
        print(f"  ⏱ {time.time()-start:.1f}s total")
    elif cmd.startswith("/remember "):
        parts = cmd[10:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /remember <key> note"); return True
        memory.add(parts[0], parts[1])
        print(f"  ✓ Stored under '{parts[0]}'")
    elif cmd.startswith("/recall"):
        key = cmd[7:].strip() if len(cmd) > 7 else None
        if key:
            notes = memory.get(key)
            if not notes: print(f"  (no notes for '{key}')"); return True
            for n in notes: print(f"  • {n}")
        else:
            all_mem = memory.get_all()
            if not all_mem: print("  (empty)"); return True
            for k, notes in all_mem.items():
                print(f"  [{k}] {len(notes)} notes")
                for n in notes[:3]: print(f"    • {n}")
    elif cmd.startswith("/forget"):
        key = cmd[7:].strip() if len(cmd) > 7 else None
        memory.clear(key)
        print(f"  ✓ {'Cleared ' + key if key else 'All memory cleared'}")
    elif cmd.startswith("/feedback "):
        parts = cmd[10:].strip().split(" ", 2)
        if len(parts) < 3: print("  Usage: /feedback <agent> <reviewer> message"); return True
        agent_name, reviewer_name, msg = parts
        history.add_turn("user", msg); print()
        start = time.time()
        result = run_feedback_loop(agent_name, msg, reviewer_name, history)
        print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
        tracker.track(msg, result)
    elif cmd.startswith("/files "):
        parts = cmd[7:].strip().split(" ", 1)
        if len(parts) < 2: print("  Usage: /files path1,path2 message"); return True
        file_paths = parts[0].split(",")
        msg = parts[1]
        ctx = load_multi_file_context(file_paths)
        full_msg = f"{ctx}\n\n{msg}"
        print(f"  📄 Loaded {len(file_paths)} files")
        history.add_turn("user", full_msg)
        print("\n🤖 Processing...\n")
        start = time.time()
        result = run_multi_agent(full_msg, history)
        print(f"📋 ({time.time()-start:.1f}s):\n\n{result}")
        history.add_turn("assistant", result, agent="supervisor")
        tracker.track(full_msg, result)
    else:
        return False
    return True


def _run_and_print(user_input: str, history: ConversationHistory, tracker: TokenTracker):
    """Run multi-agent and print result."""
    history.add_turn("user", user_input)
    print("\n🤖 Processing...\n")
    start = time.time()
    result = run_multi_agent(user_input, history)
    print(f"📋 Final Answer ({time.time()-start:.1f}s):\n\n{result}")
    history.add_turn("assistant", result, agent="supervisor")
    tracker.track(user_input, result)


def run_cli():
    print("=" * 60)
    print("  Multi-Agent AI System (Local LLM via Ollama)")
    print(f"  Model: {MODEL_NAME} | Agents: {len(AGENT_NAMES)}")
    print("=" * 60)
    print("  Type /help for commands\n")

    if check_ollama_health():
        print(f"✅ Ollama running, model '{MODEL_NAME}' available\n")
    else:
        print(f"⚠️  Ollama not reachable. Start with: ollama serve\n")

    history = ConversationHistory()
    tracker = TokenTracker()
    memory = AgentMemory()

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!"); break
        if user_input.startswith("/"):
            if _handle_command(user_input, history, tracker, memory):
                continue

        # Normal multi-agent flow
        try:
            _run_and_print(user_input, history, tracker)
        except Exception as e:
            print(f"❌ Error: {e}\n   Make sure Ollama is running: ollama serve")


def run_mcp_server(transport: str, host: str, port: int):
    from mcp_server import start_server
    if EXTERNAL_MCP_SERVERS:
        registry = ToolRegistry()
        asyncio.run(registry.initialize([ExternalServerConfig(**c) for c in EXTERNAL_MCP_SERVERS]))
    start_server(transport=transport, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent AI System")
    parser.add_argument("--mcp-server", action="store_true", help="Start as MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default=MCP_SERVER_TRANSPORT)
    parser.add_argument("--host", default=MCP_SSE_HOST)
    parser.add_argument("--port", type=int, default=MCP_SSE_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.mcp_server:
        run_mcp_server(args.transport, args.host, args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
