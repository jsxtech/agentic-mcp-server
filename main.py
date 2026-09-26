"""Main entry point for the multi-agent application."""

import argparse
import logging
import time

import requests

from agent_registry import AGENT_NAMES
import config as _config
from config import MCP_SERVER_TRANSPORT, MCP_SSE_HOST, MCP_SSE_PORT, EXTERNAL_MCP_SERVERS
from session import ConversationHistory, TokenTracker, save_session, load_session, list_sessions, export_session
from runners import (
    run_multi_agent, run_direct, run_chain, run_parallel,
    run_compare, run_workflow, run_streaming, load_file_context, WORKFLOWS,
    run_batch, run_conditional, AgentMemory, run_feedback_loop, load_multi_file_context,
    invalidate_graph_cache,
)
from tool_registry import ExternalServerConfig


def check_ollama_health() -> bool:
    try:
        resp = requests.get(f"{_config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(_config.MODEL_NAME in m for m in models):
            print(f"⚠️  Model '{_config.MODEL_NAME}' not found. Available: {models}")
            return False
        return True
    except (requests.ConnectionError, requests.Timeout, ValueError):
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


# --- Command Handlers ---
# Each handler receives (args, history, tracker, memory) and returns True if handled.

def _cmd_help(args, history, tracker, memory):
    print(HELP)


def _cmd_agents(args, history, tracker, memory):
    for name in AGENT_NAMES:
        print(f"  • {name}")


def _cmd_health(args, history, tracker, memory):
    print("✅ OK" if check_ollama_health() else "❌ Not reachable")


def _cmd_history(args, history, tracker, memory):
    for t in (history.turns[-20:] or [{"content": "(empty)", "agent": None}]):
        print(f"  [{t.get('agent', 'user')}] {t['content'][:100]}")


def _cmd_clear(args, history, tracker, memory):
    history.turns.clear()
    print("  ✓ Cleared")


def _cmd_sessions(args, history, tracker, memory):
    for s in (list_sessions() or ["(none)"]):
        print(f"  • {s}")


def _cmd_tokens(args, history, tracker, memory):
    print(f"  {tracker.summary()}")


def _cmd_workflows(args, history, tracker, memory):
    for name, agents in WORKFLOWS.items():
        print(f"  • {name}: {' → '.join(agents)}")


def _cmd_export(args, history, tracker, memory):
    print(f"  ✓ Exported to {export_session(history)}")


def _cmd_save(args, history, tracker, memory):
    print(f"  ✓ Saved to {save_session(history, args or None)}")


def _cmd_load(args, history, tracker, memory):
    if not args:
        print("  Usage: /load <name>")
        return
    try:
        history.turns = load_session(args).turns
        print(f"  ✓ Loaded ({len(history.turns)} turns)")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")


def _cmd_model(args, history, tracker, memory):
    if not args:
        print("  Usage: /model <name>")
        return
    import config
    config.MODEL_NAME = args
    invalidate_graph_cache()
    print(f"  ✓ Switched to: {config.MODEL_NAME}")


def _cmd_retry(args, history, tracker, memory):
    last = history.last_user_message()
    if not last:
        print("  ❌ No previous request")
        return
    # Don't re-add to history — it's already there. Just re-run and record the response.
    print("\n🤖 Retrying...\n")
    start = time.time()
    result = run_multi_agent(last, history)
    print(f"📋 Final Answer ({time.time()-start:.1f}s):\n\n{result}")
    history.add_turn("assistant", result, agent="supervisor")
    tracker.track(last, result)


def _cmd_ask(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /ask <agent> msg")
        return
    name, msg = parts
    history.add_turn("user", msg)
    print(f"\n🤖 [{name}]...\n")
    start = time.time()
    result = run_direct(name, msg, history)
    print(f"📋 ({time.time()-start:.1f}s):\n\n{result}")
    history.add_turn("assistant", result, agent=name)
    tracker.track(msg, result)


def _cmd_chain(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /chain a|b|c msg")
        return
    history.add_turn("user", parts[1])
    print()
    start = time.time()
    result = run_chain(parts[0], parts[1], history)
    print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
    tracker.track(parts[1], result)


def _cmd_compare(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /compare a,b msg")
        return
    history.add_turn("user", parts[1])
    print()
    start = time.time()
    result = run_compare(parts[0], parts[1], history)
    print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
    tracker.track(parts[1], result)


def _cmd_parallel(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /parallel a,b msg")
        return
    agent_list = [a.strip() for a in parts[0].split(",")]
    invalid = [a for a in agent_list if a not in AGENT_NAMES]
    if invalid:
        print(f"  ❌ Unknown: {invalid}")
        return
    history.add_turn("user", parts[1])
    print()
    start = time.time()
    results = run_parallel(agent_list, parts[1], history)
    for name, output in results.items():
        print(f"\n━━━ {name.upper()} ━━━\n{output}")
        history.add_turn("assistant", output, agent=name)
    # Track the shared input once (not once per agent) plus the combined output,
    # so the input token count isn't inflated N times for N parallel agents.
    tracker.track(parts[1], "\n".join(results.values()))
    print(f"\n  ⏱ {time.time()-start:.1f}s total")


def _cmd_workflow(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /workflow <name> msg")
        return
    history.add_turn("user", parts[1])
    print()
    start = time.time()
    result = run_workflow(parts[0], parts[1], history)
    print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
    tracker.track(parts[1], result)


def _cmd_file(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /file <path> msg")
        return
    try:
        ctx = load_file_context(parts[0])
        full_msg = f"{ctx}\n\n{parts[1]}"
        print(f"  📄 Loaded {parts[0]}")
        _run_and_print(full_msg, history, tracker)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")


def _cmd_stream(args, history, tracker, memory):
    if not args:
        print("  Usage: /stream msg")
        return
    history.add_turn("user", args)
    print("\n🤖 ", end="")
    result = run_streaming(args, history)
    history.add_turn("assistant", result)
    tracker.track(args, result)


def _cmd_auto(args, history, tracker, memory):
    if not args:
        print("  Usage: /auto msg")
        return
    history.add_turn("user", args)
    print()
    start = time.time()
    result = run_conditional(args, history)
    print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
    tracker.track(args, result)


def _cmd_batch(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /batch <agent> task1;;task2;;task3")
        return
    agent_name, tasks_str = parts
    tasks = [t.strip() for t in tasks_str.split(";;") if t.strip()]
    if not tasks:
        print("  ❌ No tasks provided. Separate tasks with ;;")
        return
    print(f"  Running {agent_name} on {len(tasks)} tasks...\n")
    start = time.time()
    results = run_batch(agent_name, tasks, history)
    # run_batch returns either one result per task, or a single-element error
    # list (e.g. unknown agent). Only map results to tasks 1:1 when the lengths
    # match; otherwise track the error result without mis-associating it.
    aligned = len(results) == len(tasks)
    for i, r in enumerate(results):
        print(f"━━━ Task {i+1} ━━━\n{r}\n")
        tracker.track(tasks[i] if aligned else "", r)
    print(f"  ⏱ {time.time()-start:.1f}s total")


def _cmd_remember(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /remember <key> note")
        return
    memory.add(parts[0], parts[1])
    print(f"  ✓ Stored under '{parts[0]}'")


def _cmd_recall(args, history, tracker, memory):
    key = args if args else None
    if key:
        notes = memory.get(key)
        if not notes:
            print(f"  (no notes for '{key}')")
            return
        for n in notes:
            print(f"  • {n}")
    else:
        all_mem = memory.get_all()
        if not all_mem:
            print("  (empty)")
            return
        for k, notes in all_mem.items():
            print(f"  [{k}] {len(notes)} notes")
            for n in notes[:3]:
                print(f"    • {n}")


def _cmd_forget(args, history, tracker, memory):
    key = args if args else None
    memory.clear(key)
    print(f"  ✓ {'Cleared ' + key if key else 'All memory cleared'}")


def _cmd_feedback(args, history, tracker, memory):
    parts = args.split(" ", 2) if args else []
    if len(parts) < 3:
        print("  Usage: /feedback <agent> <reviewer> message")
        return
    agent_name, reviewer_name, msg = parts
    history.add_turn("user", msg)
    print()
    start = time.time()
    result = run_feedback_loop(agent_name, msg, reviewer_name, history)
    print(f"\n📋 ({time.time()-start:.1f}s):\n\n{result}")
    tracker.track(msg, result)


def _cmd_files(args, history, tracker, memory):
    parts = args.split(" ", 1) if args else []
    if len(parts) < 2:
        print("  Usage: /files path1,path2 message")
        return
    file_paths = parts[0].split(",")
    msg = parts[1]
    try:
        ctx = load_multi_file_context(file_paths)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return
    full_msg = f"{ctx}\n\n{msg}"
    print(f"  📄 Loaded {len(file_paths)} files")
    history.add_turn("user", full_msg)
    print("\n🤖 Processing...\n")
    start = time.time()
    result = run_multi_agent(full_msg, history)
    print(f"📋 ({time.time()-start:.1f}s):\n\n{result}")
    history.add_turn("assistant", result, agent="supervisor")
    tracker.track(full_msg, result)


# --- Command Registry ---
# Maps command name to handler. Every command is a single whitespace-delimited
# token, so an exact match on the first token is sufficient and unambiguous
# (e.g. "/file" vs "/files" are distinct tokens and never collide).

COMMANDS = {
    "/help": _cmd_help,
    "/agents": _cmd_agents,
    "/health": _cmd_health,
    "/history": _cmd_history,
    "/clear": _cmd_clear,
    "/sessions": _cmd_sessions,
    "/tokens": _cmd_tokens,
    "/workflows": _cmd_workflows,
    "/export": _cmd_export,
    "/save": _cmd_save,
    "/load": _cmd_load,
    "/model": _cmd_model,
    "/retry": _cmd_retry,
    "/ask": _cmd_ask,
    "/chain": _cmd_chain,
    "/compare": _cmd_compare,
    "/parallel": _cmd_parallel,
    "/workflow": _cmd_workflow,
    "/file": _cmd_file,
    "/stream": _cmd_stream,
    "/auto": _cmd_auto,
    "/batch": _cmd_batch,
    "/remember": _cmd_remember,
    "/recall": _cmd_recall,
    "/forget": _cmd_forget,
    "/feedback": _cmd_feedback,
    "/files": _cmd_files,
}


def _handle_command(cmd: str, history: ConversationHistory, tracker: TokenTracker, memory: AgentMemory) -> bool:
    """Handle a slash command using the command registry. Returns True if handled.

    Matching is by exact command token only. An unrecognized token (e.g. a typo
    like "/models") is NOT handled here and falls through to the normal flow,
    rather than being silently mis-routed to a similarly-named command.
    """
    parts = cmd.split(" ", 1)
    cmd_name = parts[0]
    args = parts[1].strip() if len(parts) > 1 else ""

    handler = COMMANDS.get(cmd_name)
    if handler is None:
        return False
    handler(args, history, tracker, memory)
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


def _process_input(user_input: str, history: ConversationHistory, tracker: TokenTracker, memory: AgentMemory) -> bool:
    """Process a single line of user input. Returns False when the REPL should exit.

    All command dispatch and the normal multi-agent flow run inside one try/except
    so that an LLM/network error (e.g. Ollama down) during ANY command — not just
    the normal flow — is reported gracefully instead of crashing the REPL.
    """
    if not user_input:
        return True
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        return False

    try:
        if user_input.startswith("/"):
            if _handle_command(user_input, history, tracker, memory):
                return True
        # Normal multi-agent flow (also reached for a "/" input that isn't a command)
        _run_and_print(user_input, history, tracker)
    except Exception as e:
        print(f"❌ Error: {e}\n   Make sure Ollama is running: ollama serve")
    return True


def run_cli():
    print("=" * 60)
    print("  Multi-Agent AI System (Local LLM via Ollama)")
    print(f"  Model: {_config.MODEL_NAME} | Agents: {len(AGENT_NAMES)}")
    print("=" * 60)
    print("  Type /help for commands\n")

    if check_ollama_health():
        print(f"✅ Ollama running, model '{_config.MODEL_NAME}' available\n")
    else:
        print(f"⚠️  Ollama not reachable. Start with: ollama serve\n")

    history = ConversationHistory()
    tracker = TokenTracker()
    memory = AgentMemory()

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not _process_input(user_input, history, tracker, memory):
            break


def run_mcp_server(transport: str, host: str, port: int):
    from mcp_server import start_server, set_tool_registry
    if EXTERNAL_MCP_SERVERS:
        # Pass configs to the MCP server module for lifecycle-managed initialization.
        # The registry must be initialized in the same event loop as the server.
        set_tool_registry([ExternalServerConfig(**c) for c in EXTERNAL_MCP_SERVERS])
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
