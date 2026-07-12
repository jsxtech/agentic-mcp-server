"""Agent execution modes — direct, chain, parallel, compare, stream, workflow."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

from agent_registry import AGENT_NAMES, run_agent
from graph import build_agent_graph
from session import ConversationHistory

# Cache compiled graph — stateless and reusable across requests
_cached_graph = None


def _get_graph():
    """Return the cached compiled graph, building it once on first use."""
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_agent_graph()
    return _cached_graph


def invalidate_graph_cache():
    """Clear cached graph (call when config changes that affect graph structure)."""
    global _cached_graph
    _cached_graph = None

WORKFLOWS = {
    "code_review": ["coder", "reviewer", "tester"],
    "full_dev": ["planner", "coder", "reviewer", "tester"],
    "docs": ["researcher", "writer", "reviewer"],
    "security_audit": ["coder", "security", "compliance"],
    "design": ["planner", "architect", "diagrammer"],
    "optimize": ["coder", "optimizer", "reviewer"],
    "api_build": ["api_designer", "coder", "tester", "writer"],
    "db_design": ["planner", "database", "reviewer"],
    "bug_fix": ["debugger", "coder", "tester"],
    "learn": ["researcher", "mentor", "summarizer"],
    "migrate": ["planner", "migrator", "tester", "reviewer"],
    "frontend": ["ux_designer", "coder", "accessibility", "reviewer"],
    "deploy": ["devops", "security", "validator"],
    "estimate": ["planner", "estimator", "reviewer"],
    "refactor": ["explainer", "refactorer", "reviewer", "tester"],
    "translate": ["translator", "reviewer", "writer"],
    "prompt_craft": ["prompt_engineer", "tester", "optimizer"],
    "full_stack": ["planner", "architect", "api_designer", "database", "coder", "tester"],
    "onboarding": ["explainer", "mentor", "diagrammer"],
    "incident": ["debugger", "devops", "summarizer"],
    "mvp": ["product_manager", "planner", "coder", "tester"],
    "code_explain": ["explainer", "diagrammer", "summarizer"],
    "perf_audit": ["performance_tester", "optimizer", "reviewer"],
    "compliance_check": ["security", "compliance", "validator"],
    "interview_prep": ["researcher", "interviewer", "mentor"],
    "git_workflow": ["git_expert", "devops", "automator"],
    "ux_audit": ["ux_designer", "accessibility", "reviewer"],
    "data_pipeline": ["data_analyst", "coder", "tester", "devops"],
    "tech_spec": ["product_manager", "architect", "api_designer", "estimator"],
    "legacy_modernize": ["explainer", "architect", "migrator", "coder", "tester"],
    "error_resilience": ["error_handler", "coder", "tester", "reviewer"],
    "ml_project": ["researcher", "ml_engineer", "coder", "tester", "documentation"],
    "shell_automation": ["shell_expert", "automator", "tester"],
    "regex_build": ["regex_expert", "tester", "explainer"],
    "concurrent_system": ["architect", "concurrency", "coder", "tester", "reviewer"],
    "scaffold": ["planner", "code_generator", "coder", "tester"],
    "config_setup": ["config_manager", "devops", "validator"],
    "production_ready": ["coder", "error_handler", "security", "performance_tester", "devops"],
    "api_full": ["api_designer", "code_generator", "coder", "tester", "documentation", "security"],
    "team_onboard": ["documentation", "diagrammer", "mentor", "explainer"],
    "release": ["tester", "security", "compliance", "documentation", "devops"],
    "seo_optimize": ["seo_expert", "coder", "performance_tester"],
    "observability": ["monitoring", "devops", "shell_expert"],
    "microservice": ["architect", "api_designer", "coder", "contract_tester", "devops"],
    "tech_decision": ["researcher", "tech_lead", "architect", "estimator"],
    "network_setup": ["networking", "security", "devops", "validator"],
    "full_review": ["explainer", "reviewer", "security", "optimizer", "accessibility"],
    "startup_mvp": ["product_manager", "planner", "architect", "code_generator", "coder", "tester", "devops"],
    "incident_response": ["debugger", "monitoring", "devops", "summarizer", "documentation"],
    "api_contract": ["api_designer", "contract_tester", "tester", "documentation"],
}


def run_multi_agent(user_input: str, history: ConversationHistory = None) -> str:
    """Run the full supervisor orchestration."""
    graph = _get_graph()
    context_input = user_input
    if history and history.turns:
        context_input = f"{history.get_context()}\n\nCurrent request: {user_input}"
    initial_state = {
        "user_input": context_input, "agent_outputs": {}, "next_agent": "",
        "final_answer": "", "iteration": 0, "tool_calls": [], "available_tools": [],
    }
    return graph.invoke(initial_state).get("final_answer", "No response generated.")


def run_direct(agent_name: str, task: str, history: ConversationHistory) -> str:
    """Run a specific agent directly.

    Returns error string prefixed with ❌ for unknown agents.
    """
    if agent_name not in AGENT_NAMES:
        return f"❌ Unknown agent: '{agent_name}'. Available: {', '.join(AGENT_NAMES[:5])}..."
    context = history.get_context()
    return run_agent(agent_name, f"{context}\n\n{task}" if context else task)


def run_chain(chain: str, user_input: str, history: ConversationHistory) -> str:
    """Run agents in sequence: 'coder|reviewer|tester'.

    Returns error string prefixed with ❌ for unknown agents.
    """
    agents = [a.strip() for a in chain.split("|")]
    invalid = [name for name in agents if name not in AGENT_NAMES]
    if invalid:
        return f"❌ Unknown agent(s): {invalid}. Use /agents to see available agents."

    current_input = user_input
    context = history.get_context()

    # Context window protection: limit how much of previous agent output
    # is passed forward. For long chains (5+ agents), intermediate outputs
    # can exceed the model's context window.
    max_input_chars = 8000  # ~2000 tokens — leaves room for system prompt + response

    for i, name in enumerate(agents):
        print(f"  ⛓️  [{i+1}/{len(agents)}] Running {name}...")
        task = f"{context}\n\n{current_input}" if context else current_input
        # Truncate if task exceeds context budget
        if len(task) > max_input_chars:
            task = task[:max_input_chars] + f"\n\n... [truncated — {len(task)} chars total, showing first {max_input_chars}]"
        current_input = run_agent(name, task)
        history.add_turn("assistant", current_input, agent=name)

    return current_input


def run_parallel(agent_names: list[str], task: str, history: ConversationHistory) -> dict[str, str]:
    """Run multiple agents concurrently."""
    if not agent_names:
        return {}
    context = history.get_context()
    full_task = f"{context}\n\n{task}" if context else task
    results = {}

    with ThreadPoolExecutor(max_workers=min(len(agent_names), 4)) as pool:
        futures = {pool.submit(run_agent, name, full_task): name for name in agent_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"Error: {e}"
            print(f"  ✓ {name} done")

    return results


def run_compare(agents_str: str, task: str, history: ConversationHistory) -> str:
    """Run multiple agents on same task, format side-by-side.

    Returns error string prefixed with ❌ for unknown agents.
    """
    agents = [a.strip() for a in agents_str.split(",")]
    invalid = [name for name in agents if name not in AGENT_NAMES]
    if invalid:
        return f"❌ Unknown agent(s): {invalid}. Use /agents to see available agents."

    context = history.get_context()
    full_task = f"{context}\n\n{task}" if context else task
    lines = []
    for name in agents:
        print(f"  🔄 Running {name}...")
        try:
            output = run_agent(name, full_task)
        except Exception as e:
            output = f"[Error from {name}: {e}]"
        history.add_turn("assistant", output, agent=name)
        lines.append(f"━━━ {name.upper()} ━━━\n{output}\n")
    return "\n".join(lines)


def run_workflow(workflow_name: str, task: str, history: ConversationHistory) -> str:
    """Run a predefined workflow."""
    if workflow_name not in WORKFLOWS:
        return f"❌ Unknown workflow: '{workflow_name}'. Available: {', '.join(WORKFLOWS.keys())}"
    return run_chain("|".join(WORKFLOWS[workflow_name]), task, history)


def run_streaming(user_input: str, history: ConversationHistory) -> str:
    """Stream a direct LLM response."""
    from config import MODEL_NAME, OLLAMA_BASE_URL, MAX_TOKENS, MCP_REQUEST_TIMEOUT
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_predict=MAX_TOKENS, timeout=MCP_REQUEST_TIMEOUT)
    context = history.get_context()
    prompt = f"{context}\n\nUser request: {user_input}" if context else user_input
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Answer the user's question directly."),
        HumanMessage(content=prompt),
    ]
    result = ""
    for chunk in llm.stream(messages):
        text = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(text, end="", flush=True)
        result += text
    print()
    return result


def load_file_context(file_path: str) -> str:
    """Load a file's content to use as context."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    content = path.read_text()
    if len(content) > 10000:
        content = content[:10000] + f"\n\n... (truncated, {len(content)} chars total)"
    return f"File: {path.name}\n```\n{content}\n```"


# --- Batch Processing ---

def run_batch(agent_name: str, tasks: list[str], history: ConversationHistory) -> list[str]:
    """Run the same agent on multiple tasks.

    Returns a single-element list with error string for unknown agents or empty tasks.
    """
    if agent_name not in AGENT_NAMES:
        return [f"❌ Unknown agent: '{agent_name}'. Use /agents to see available agents."]
    if not tasks:
        return [f"❌ No tasks provided."]

    results = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        futures = [pool.submit(run_agent, agent_name, task) for task in tasks]
        for i, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as e:
                result = f"Error: {e}"
            results.append(result)
            print(f"  ✓ Task {i+1}/{len(tasks)} done")
            history.add_turn("assistant", result, agent=agent_name)
    return results


# --- Conditional Workflow ---

def run_conditional(task: str, history: ConversationHistory) -> str:
    """Auto-select workflow based on task keywords using weighted scoring.

    Uses multi-keyword matching with specificity weighting. Longer/more-specific
    keywords score higher. Multiple keyword hits for the same workflow accumulate.
    """
    task_lower = task.lower()

    # Each workflow maps to a list of (keyword, weight) pairs.
    # Multi-word keywords get higher weight (more specific).
    # A keyword only matches as a whole word boundary to avoid false positives.
    workflow_keywords: dict[str, list[tuple[str, int]]] = {
        "bug_fix": [("bug", 3), ("fix", 2), ("error", 2), ("crash", 3), ("broken", 2), ("traceback", 3)],
        "code_review": [("test", 1), ("review", 2), ("code review", 4)],
        "deploy": [("deploy", 3), ("ci/cd", 4), ("pipeline", 2), ("kubernetes", 3), ("docker", 2)],
        "api_build": [("api", 2), ("endpoint", 3), ("rest api", 4), ("graphql", 3)],
        "db_design": [("database", 3), ("schema", 3), ("sql", 2), ("migration", 2), ("table", 1)],
        "security_audit": [("security", 3), ("vulnerability", 4), ("exploit", 4), ("xss", 4), ("injection", 4)],
        "migrate": [("migrate", 3), ("upgrade", 2), ("migration", 3)],
        "docs": [("document", 2), ("readme", 3), ("documentation", 3), ("write docs", 4)],
        "design": [("design", 2), ("architect", 3), ("system design", 4), ("architecture", 3)],
        "perf_audit": [("performance", 3), ("slow", 2), ("latency", 3), ("bottleneck", 3)],
        "optimize": [("optimize", 3), ("optimization", 3), ("faster", 2)],
        "frontend": [("frontend", 3), ("ui", 2), ("react", 2), ("component", 1)],
        "ux_audit": [("ux", 3), ("usability", 3), ("accessibility", 3), ("wcag", 4)],
        "refactor": [("refactor", 3), ("clean up", 3), ("restructure", 3), ("technical debt", 4)],
        "learn": [("learn", 2), ("explain", 2), ("how does", 3), ("tutorial", 3), ("teach", 2)],
        "code_explain": [("explain this code", 5), ("what does this", 4), ("walk through", 4)],
        "estimate": [("estimate", 3), ("timeline", 3), ("how long", 3), ("effort", 2)],
        "scaffold": [("scaffold", 4), ("boilerplate", 4), ("generate", 2), ("starter", 2)],
        "ml_project": [("machine learning", 5), ("ml pipeline", 5), ("train model", 4), ("neural", 3)],
    }

    # Score each workflow by summing weights of all matching keywords
    scores: dict[str, int] = {}
    for workflow, keywords in workflow_keywords.items():
        score = 0
        for keyword, weight in keywords:
            if keyword in task_lower:
                score += weight
        if score > 0:
            scores[workflow] = score

    if scores:
        selected = max(scores, key=scores.get)
    else:
        selected = "full_dev"

    print(f"  🧠 Auto-selected workflow: {selected}")
    return run_workflow(selected, task, history)


# --- Agent Memory (persistent per-agent notes) ---

class AgentMemory:
    """Persistent memory that agents can reference across sessions."""

    _PROJECT_ROOT = Path(__file__).parent

    def __init__(self, path: Path = None):
        self._path = path or (self._PROJECT_ROOT / "sessions" / "memory.json")
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                if isinstance(data, dict):
                    self._data = data
                else:
                    logger.warning(f"Memory file has unexpected format, resetting: {self._path}")
                    self._data = {}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Corrupted memory file, resetting: {self._path} ({e})")
                self._data = {}

    def _save(self):
        self._path.parent.mkdir(exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def add(self, key: str, note: str):
        self._data.setdefault(key, []).append(note)
        self._save()

    def get(self, key: str) -> list[str]:
        return self._data.get(key, [])

    def get_all(self) -> dict[str, list[str]]:
        return self._data

    def clear(self, key: str = None):
        if key:
            self._data.pop(key, None)
        else:
            self._data.clear()
        self._save()


# --- Feedback Loop (iterative refinement) ---

def run_feedback_loop(agent_name: str, task: str, reviewer_name: str, history: ConversationHistory, max_rounds: int = 3) -> str:
    """Run agent, get review, iterate until approved or max rounds.

    Returns error string prefixed with ❌ for unknown agents.
    """
    invalid = [n for n in (agent_name, reviewer_name) if n not in AGENT_NAMES]
    if invalid:
        return f"❌ Unknown agent(s): {invalid}. Use /agents to see available agents."

    context = history.get_context()
    current_task = f"{context}\n\n{task}" if context else task
    result = ""
    max_input_chars = 8000  # Context window protection

    for i in range(max_rounds):
        print(f"  🔄 Round {i+1}: {agent_name}...")
        # Truncate if accumulated context exceeds budget
        agent_input = current_task
        if len(agent_input) > max_input_chars:
            agent_input = agent_input[:max_input_chars] + f"\n\n... [truncated for context window]"
        result = run_agent(agent_name, agent_input)
        history.add_turn("assistant", result, agent=agent_name)

        print(f"  🔍 Round {i+1}: {reviewer_name} reviewing...")
        review = run_agent(reviewer_name, f"Review this output and say APPROVED if it's good, or provide specific feedback for improvement:\n\n{result}")
        history.add_turn("assistant", review, agent=reviewer_name)

        if "APPROVED" in review.upper():
            print(f"  ✅ Approved after {i+1} round(s)")
            return result

        # Feed review back for next iteration
        current_task = f"{task}\n\nPrevious attempt:\n{result}\n\nFeedback:\n{review}\n\nPlease improve based on the feedback."

    print(f"  ⚠️ Max rounds ({max_rounds}) reached")
    return result


# --- Multi-File Context ---

def load_multi_file_context(file_paths: list[str]) -> str:
    """Load multiple files as context.

    Skips files that don't exist with a warning marker inline.
    This is intentionally lenient (vs raising) since partial context is still useful
    when loading multiple files.
    """
    parts = []
    for fp in file_paths:
        path = Path(fp.strip())
        if not path.exists():
            parts.append(f"[⚠️ File not found: {fp}]")
            continue
        content = path.read_text()
        if len(content) > 5000:
            content = content[:5000] + f"\n... (truncated, {len(content)} chars)"
        parts.append(f"### {path.name}\n```\n{content}\n```")
    if not parts:
        raise FileNotFoundError(f"None of the specified files exist: {file_paths}")
    return "\n\n".join(parts)
