"""Agent execution modes — direct, chain, parallel, compare, stream, workflow."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agent_registry import AGENT_NAMES, run_agent
from graph import build_agent_graph, AgentState
from session import ConversationHistory

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
    graph = build_agent_graph()
    context_input = user_input
    if history and history.turns:
        context_input = f"{history.get_context()}\n\nCurrent request: {user_input}"
    initial_state: AgentState = {
        "user_input": context_input, "agent_outputs": {}, "next_agent": "",
        "final_answer": "", "iteration": 0, "tool_calls": [], "available_tools": [],
    }
    return graph.invoke(initial_state).get("final_answer", "No response generated.")


def run_direct(agent_name: str, task: str, history: ConversationHistory) -> str:
    """Run a specific agent directly."""
    if agent_name not in AGENT_NAMES:
        return f"❌ Unknown agent: '{agent_name}'"
    context = history.get_context()
    return run_agent(agent_name, f"{context}\n\n{task}" if context else task)


def run_chain(chain: str, user_input: str, history: ConversationHistory) -> str:
    """Run agents in sequence: 'coder|reviewer|tester'"""
    agents = [a.strip() for a in chain.split("|")]
    for name in agents:
        if name not in AGENT_NAMES:
            return f"❌ Unknown agent: '{name}'"

    current_input = user_input
    context = history.get_context()

    for i, name in enumerate(agents):
        print(f"  ⛓️  [{i+1}/{len(agents)}] Running {name}...")
        task = f"{context}\n\n{current_input}" if context else current_input
        current_input = run_agent(name, task)
        history.add_turn("assistant", current_input, agent=name)

    return current_input


def run_parallel(agent_names: list[str], task: str, history: ConversationHistory) -> dict[str, str]:
    """Run multiple agents concurrently."""
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
    """Run multiple agents on same task, format side-by-side."""
    agents = [a.strip() for a in agents_str.split(",")]
    for name in agents:
        if name not in AGENT_NAMES:
            return f"❌ Unknown agent: '{name}'"

    context = history.get_context()
    full_task = f"{context}\n\n{task}" if context else task
    lines = []
    for name in agents:
        print(f"  🔄 Running {name}...")
        output = run_agent(name, full_task)
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
    from config import MODEL_NAME, OLLAMA_BASE_URL, MAX_TOKENS
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_predict=MAX_TOKENS)
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
    """Run the same agent on multiple tasks."""
    if agent_name not in AGENT_NAMES:
        return [f"❌ Unknown agent: '{agent_name}'"]

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
    """Auto-select workflow based on task keywords."""
    task_lower = task.lower()
    keyword_map = {
        "bug": "bug_fix", "fix": "bug_fix", "error": "bug_fix",
        "test": "code_review", "review": "code_review",
        "deploy": "deploy", "ci/cd": "deploy", "pipeline": "deploy",
        "api": "api_build", "endpoint": "api_build", "rest": "api_build",
        "database": "db_design", "schema": "db_design", "sql": "db_design",
        "security": "security_audit", "vulnerability": "security_audit",
        "migrate": "migrate", "upgrade": "migrate",
        "document": "docs", "readme": "docs",
        "design": "design", "architect": "design",
        "performance": "perf_audit", "slow": "perf_audit", "optimize": "optimize",
        "frontend": "frontend", "ui": "frontend", "ux": "ux_audit",
        "refactor": "refactor", "clean": "refactor",
        "learn": "learn", "explain": "code_explain", "how": "learn",
        "estimate": "estimate", "timeline": "estimate",
        "scaffold": "scaffold", "boilerplate": "scaffold", "generate": "scaffold",
        "ml": "ml_project", "machine learning": "ml_project", "model": "ml_project",
    }

    selected = None
    for keyword, workflow in keyword_map.items():
        if keyword in task_lower:
            selected = workflow
            break

    if not selected:
        selected = "full_dev"

    print(f"  🧠 Auto-selected workflow: {selected}")
    return run_workflow(selected, task, history)


# --- Agent Memory (persistent per-agent notes) ---

class AgentMemory:
    """Persistent memory that agents can reference across sessions."""

    def __init__(self, path: Path = Path("sessions/memory.json")):
        self._path = path
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

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
    """Run agent, get review, iterate until approved or max rounds."""
    if agent_name not in AGENT_NAMES or reviewer_name not in AGENT_NAMES:
        return "❌ Unknown agent name"

    context = history.get_context()
    current_task = f"{context}\n\n{task}" if context else task
    result = ""

    for i in range(max_rounds):
        print(f"  🔄 Round {i+1}: {agent_name}...")
        result = run_agent(agent_name, current_task)
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
    """Load multiple files as context."""
    parts = []
    for fp in file_paths:
        path = Path(fp.strip())
        if not path.exists():
            parts.append(f"[File not found: {fp}]")
            continue
        content = path.read_text()
        if len(content) > 5000:
            content = content[:5000] + f"\n... (truncated, {len(content)} chars)"
        parts.append(f"### {path.name}\n```\n{content}\n```")
    return "\n\n".join(parts)
