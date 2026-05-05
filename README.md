# Multi-Agent AI Application (100% Offline)

A fully local, open-source agentic AI system with **45 specialized agents**, **50 predefined workflows**, and an **MCP server** — powered by Ollama. No API keys, no cloud services, no internet required after setup.

---

## Table of Contents

- [Complete Installation Guide](#complete-installation-guide)
- [Local LLM Setup (Ollama)](#local-llm-setup-ollama)
- [Local MCP Server Setup](#local-mcp-server-setup)
- [External Local MCP Servers](#external-local-mcp-servers)
- [Quick Start](#quick-start)
- [CLI Commands Reference](#cli-commands-reference)
- [Agents (45)](#agents-45)
- [Workflows (50)](#workflows-50)
- [Use Cases with Examples](#use-cases-with-examples)
- [MCP Client Integration](#mcp-client-integration)
- [Advanced Features](#advanced-features)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)

---

## Complete Installation Guide

### Prerequisites

- Python 3.12+
- 8GB+ RAM (16GB recommended for 13B models)
- ~5GB disk space (for model + dependencies)
- No GPU required (but speeds up inference)

### Step 1: Install Ollama (Local LLM Runtime)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows — download installer from https://ollama.com/download

# Verify installation
ollama --version
```

### Step 2: Start Ollama Service

```bash
# Start the Ollama daemon
ollama serve

# It runs on http://localhost:11434 by default
# Keep this terminal open, or run as a system service:
# sudo systemctl enable ollama && sudo systemctl start ollama  (Linux)
```

### Step 3: Pull a Local LLM Model

```bash
# Recommended: good balance of speed and quality
ollama pull llama3.1:8b

# Alternatives:
ollama pull mistral              # Fast, 7B params
ollama pull codellama:13b        # Best for code tasks
ollama pull qwen2.5:7b           # Good multilingual
ollama pull llama3.1:70b         # Best quality (needs 40GB+ RAM)
ollama pull deepseek-coder:6.7b  # Specialized for code
ollama pull phi3:mini             # Smallest, fastest

# Verify model is available
ollama list
```

### Step 4: Set Up the Application

```bash
cd multi-agent-app

# Create Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
# .venv\Scripts\activate.bat     # Windows CMD

# Install all dependencies
pip install -r requirements.txt
```

### Step 5: Configure (Optional)

Edit `config.py` if you changed defaults:

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama address
MODEL_NAME = "llama3.1:8b"                  # Model you pulled
MAX_TOKENS = 2048                           # Max response length
```

### Step 6: Run

```bash
# Interactive CLI mode
python main.py

# Or as MCP server
python main.py --mcp-server
```

---

## Local LLM Setup (Ollama)

### Managing Models

```bash
# List installed models
ollama list

# Pull a new model
ollama pull <model-name>

# Remove a model
ollama rm <model-name>

# Show model details
ollama show llama3.1:8b

# Test a model directly
ollama run llama3.1:8b "Hello, how are you?"
```

### Recommended Models by Use Case

| Use Case | Model | RAM Needed | Speed |
|----------|-------|-----------|-------|
| General (default) | `llama3.1:8b` | 8GB | Fast |
| Code-heavy work | `codellama:13b` | 16GB | Medium |
| Fast responses | `mistral` or `phi3:mini` | 4-8GB | Very fast |
| Complex reasoning | `llama3.1:70b` | 40GB+ | Slow |
| Multilingual | `qwen2.5:7b` | 8GB | Fast |
| Code + explanation | `deepseek-coder:6.7b` | 8GB | Fast |

### Switch Model at Runtime

No restart needed — switch in the CLI:
```
/model codellama:13b
/model mistral
```

### Ollama Configuration

```bash
# Change Ollama host/port (if needed)
export OLLAMA_HOST=0.0.0.0:11434

# Set GPU layers (for partial GPU offload)
export OLLAMA_NUM_GPU=999

# Set number of threads
export OLLAMA_NUM_THREAD=8
```

---

## Local MCP Server Setup

### This App as an MCP Server

Your multi-agent system IS an MCP server. Start it:

```bash
# stdio transport (for Claude Desktop, Cursor, etc.)
python main.py --mcp-server

# SSE/HTTP transport (for web clients or remote access on LAN)
python main.py --mcp-server --transport sse --host 0.0.0.0 --port 8080
```

### What Gets Exposed via MCP

| Type | Count | Description |
|------|-------|-------------|
| Tools | 47 | `run_multi_agent` + 45 individual agent tools + `list_agents` |
| Resources | 2 | `agents://list`, `config://system` |

### MCP Server CLI Arguments

```bash
python main.py --mcp-server [OPTIONS]

Options:
  --transport {stdio,sse}   Transport protocol (default: stdio)
  --host HOST               Bind address for SSE (default: 0.0.0.0)
  --port PORT               Port for SSE (default: 8080)
```

---

## External Local MCP Servers

Your agents can consume tools from OTHER local MCP servers running on your machine.

### Install Local MCP Servers

```bash
# Install Node.js MCP servers (one-time, cached locally)
npx -y @modelcontextprotocol/server-filesystem /tmp
npx -y @modelcontextprotocol/server-sqlite mydb.sqlite
npx -y @modelcontextprotocol/server-memory

# Or install Python-based MCP servers
pip install mcp-server-fetch
pip install mcp-server-git
```

### Configure External Servers

Edit `config.py`:

```python
EXTERNAL_MCP_SERVERS = [
    # Filesystem access — agents can read/write local files
    {
        "name": "filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/jaspal/projects"]
    },

    # SQLite — agents can query local databases
    {
        "name": "sqlite",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "/home/jaspal/data/app.db"]
    },

    # Memory/Knowledge base — persistent agent memory
    {
        "name": "memory",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"]
    },

    # Git — agents can interact with local repos
    {
        "name": "git",
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_server_git", "--repo", "/home/jaspal/projects/myapp"]
    },

    # Custom local MCP server (running on localhost)
    {
        "name": "custom-tools",
        "transport": "sse",
        "url": "http://localhost:9090/sse"
    },
]
```

### How It All Connects (100% Local)

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR MACHINE                          │
│                                                         │
│  ┌─────────────┐     ┌──────────────────────────────┐  │
│  │   Ollama    │     │   Multi-Agent App            │  │
│  │ (Local LLM) │◄───►│   45 agents + supervisor     │  │
│  │ :11434      │     │   MCP server (stdio/SSE)     │  │
│  └─────────────┘     └──────────┬───────────────────┘  │
│                                  │                      │
│                    ┌─────────────┼─────────────┐        │
│                    ▼             ▼             ▼        │
│  ┌──────────────┐ ┌───────────┐ ┌───────────────┐     │
│  │ MCP Server:  │ │MCP Server:│ │ MCP Server:   │     │
│  │ filesystem   │ │ sqlite    │ │ memory        │     │
│  │ (local files)│ │ (local db)│ │ (local store) │     │
│  └──────────────┘ └───────────┘ └───────────────┘     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ MCP Clients: Claude Desktop / Cursor / VS Code   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Network: ZERO external traffic                         │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run the app
cd multi-agent-app
source .venv/bin/activate
python main.py
```

Then type:
```
🧑 You: write a Python REST API with Flask for a todo app
```

The supervisor automatically routes to the best agent(s) and returns the result.

---

## CLI Commands Reference

### Agent Execution

| Command | Description | Example |
|---------|-------------|---------|
| *(just type)* | Auto-route via supervisor | `write a REST API for users` |
| `/ask <agent> msg` | Run specific agent | `/ask coder write binary search` |
| `/chain <a\|b\|c> msg` | Chain agents sequentially | `/chain coder\|reviewer\|tester build calculator` |
| `/parallel <a,b> msg` | Run concurrently | `/parallel coder,security write login` |
| `/compare <a,b> msg` | Compare outputs | `/compare coder,refactorer implement sort` |
| `/workflow <name> msg` | Predefined pipeline | `/workflow full_dev build todo app` |
| `/auto msg` | Auto-select workflow | `/auto fix the login bug` |
| `/feedback <a> <r> msg` | Iterative refinement | `/feedback coder reviewer write parser` |
| `/batch <agent> t1;;t2` | Batch process | `/batch coder sort;;search;;hash` |
| `/stream msg` | Stream response | `/stream explain monads` |

### File Context

| Command | Description | Example |
|---------|-------------|---------|
| `/file <path> msg` | Single file context | `/file src/app.py review this` |
| `/files <p1,p2> msg` | Multiple files | `/files api.py,db.py find bugs` |

### Session Management

| Command | Description |
|---------|-------------|
| `/save [name]` | Save session |
| `/load <name>` | Load session |
| `/export` | Export as markdown |
| `/sessions` | List sessions |
| `/history` | Show history |
| `/clear` | Clear history |

### Memory

| Command | Description |
|---------|-------------|
| `/remember <key> note` | Store persistent note |
| `/recall [key]` | Recall notes |
| `/forget [key]` | Clear memory |

### System

| Command | Description |
|---------|-------------|
| `/agents` | List all 45 agents |
| `/workflows` | List all 50 workflows |
| `/tokens` | Token usage stats |
| `/model <name>` | Switch model |
| `/health` | Check Ollama |
| `/retry` | Re-run last request |
| `/help` | Show help |
| `quit` / `exit` | Exit |

---

## Agents (45)

| Agent | Description | Temp |
|-------|-------------|------|
| researcher | Gathers information and provides summaries | 0.7 |
| coder | Writes clean, production-quality code | 0.3 |
| reviewer | Reviews code/content for quality and correctness | 0.4 |
| planner | Breaks down complex tasks into actionable steps | 0.5 |
| debugger | Diagnoses errors and suggests targeted fixes | 0.2 |
| writer | Writes documentation, emails, and reports | 0.6 |
| tester | Writes test cases and testing strategies | 0.3 |
| optimizer | Performance optimization and bottleneck analysis | 0.3 |
| security | Security analysis and vulnerability detection | 0.2 |
| data_analyst | Data analysis, SQL queries, and data modeling | 0.4 |
| devops | CI/CD, Docker, Kubernetes, and infrastructure | 0.3 |
| translator | Translation and localization between languages | 0.5 |
| architect | System architecture design and trade-off analysis | 0.4 |
| mentor | Explains concepts and guides learning | 0.6 |
| summarizer | Condenses content into key points and summaries | 0.3 |
| api_designer | API design, OpenAPI specs, and contracts | 0.3 |
| database | Schema design, SQL optimization, and DB architecture | 0.3 |
| ux_designer | UI/UX design, wireframes, and accessibility | 0.5 |
| refactorer | Code restructuring and maintainability improvements | 0.2 |
| explainer | Code walkthroughs and detailed explanations | 0.5 |
| validator | Verifies implementations match requirements | 0.2 |
| automator | Automation scripts, CLI tools, and workflows | 0.3 |
| migrator | Code/database/infrastructure migrations | 0.3 |
| prompt_engineer | Crafts and optimizes LLM prompts | 0.4 |
| diagrammer | Creates Mermaid/PlantUML system diagrams | 0.3 |
| estimator | Effort and time estimation for tasks | 0.4 |
| compliance | Regulatory compliance and standards audits | 0.2 |
| product_manager | Requirements, user stories, and prioritization | 0.5 |
| interviewer | Interview questions and answer evaluation | 0.5 |
| git_expert | Git workflows, branching, and conflict resolution | 0.3 |
| accessibility | WCAG compliance and inclusive design | 0.3 |
| performance_tester | Load testing and scalability analysis | 0.3 |
| error_handler | Error handling patterns and resilience | 0.3 |
| documentation | API docs, changelogs, and guides | 0.5 |
| regex_expert | Crafts and explains regular expressions | 0.2 |
| shell_expert | Shell scripting and Unix tools | 0.3 |
| ml_engineer | ML pipelines, training, and evaluation | 0.4 |
| concurrency | Async, threading, and parallel processing | 0.3 |
| config_manager | Configuration, env vars, and feature flags | 0.3 |
| code_generator | Boilerplate, scaffolding, and templates | 0.3 |
| tech_lead | Technical decisions and team guidance | 0.4 |
| seo_expert | SEO optimization and web performance | 0.4 |
| monitoring | Observability, alerting, and SRE practices | 0.3 |
| networking | DNS, load balancing, and network architecture | 0.3 |
| contract_tester | API contract testing and compatibility | 0.2 |

---

## Workflows (50)

### Development
| Workflow | Pipeline |
|----------|----------|
| `full_dev` | planner → coder → reviewer → tester |
| `code_review` | coder → reviewer → tester |
| `bug_fix` | debugger → coder → tester |
| `refactor` | explainer → refactorer → reviewer → tester |
| `scaffold` | planner → code_generator → coder → tester |
| `optimize` | coder → optimizer → reviewer |
| `error_resilience` | error_handler → coder → tester → reviewer |
| `concurrent_system` | architect → concurrency → coder → tester → reviewer |

### API & Backend
| Workflow | Pipeline |
|----------|----------|
| `api_build` | api_designer → coder → tester → writer |
| `api_full` | api_designer → code_generator → coder → tester → documentation → security |
| `api_contract` | api_designer → contract_tester → tester → documentation |
| `db_design` | planner → database → reviewer |
| `microservice` | architect → api_designer → coder → contract_tester → devops |
| `full_stack` | planner → architect → api_designer → database → coder → tester |
| `data_pipeline` | data_analyst → coder → tester → devops |

### DevOps & Infrastructure
| Workflow | Pipeline |
|----------|----------|
| `deploy` | devops → security → validator |
| `production_ready` | coder → error_handler → security → performance_tester → devops |
| `release` | tester → security → compliance → documentation → devops |
| `observability` | monitoring → devops → shell_expert |
| `config_setup` | config_manager → devops → validator |
| `network_setup` | networking → security → devops → validator |
| `git_workflow` | git_expert → devops → automator |
| `shell_automation` | shell_expert → automator → tester |

### Security & Quality
| Workflow | Pipeline |
|----------|----------|
| `security_audit` | coder → security → compliance |
| `compliance_check` | security → compliance → validator |
| `full_review` | explainer → reviewer → security → optimizer → accessibility |
| `perf_audit` | performance_tester → optimizer → reviewer |

### Frontend & UX
| Workflow | Pipeline |
|----------|----------|
| `frontend` | ux_designer → coder → accessibility → reviewer |
| `ux_audit` | ux_designer → accessibility → reviewer |
| `seo_optimize` | seo_expert → coder → performance_tester |

### Documentation & Learning
| Workflow | Pipeline |
|----------|----------|
| `docs` | researcher → writer → reviewer |
| `learn` | researcher → mentor → summarizer |
| `code_explain` | explainer → diagrammer → summarizer |
| `team_onboard` | documentation → diagrammer → mentor → explainer |
| `translate` | translator → reviewer → writer |

### Planning & Management
| Workflow | Pipeline |
|----------|----------|
| `design` | planner → architect → diagrammer |
| `estimate` | planner → estimator → reviewer |
| `tech_spec` | product_manager → architect → api_designer → estimator |
| `tech_decision` | researcher → tech_lead → architect → estimator |
| `mvp` | product_manager → planner → coder → tester |
| `startup_mvp` | product_manager → planner → architect → code_generator → coder → tester → devops |

### Migration & Modernization
| Workflow | Pipeline |
|----------|----------|
| `migrate` | planner → migrator → tester → reviewer |
| `legacy_modernize` | explainer → architect → migrator → coder → tester |

### Incident & Operations
| Workflow | Pipeline |
|----------|----------|
| `incident` | debugger → devops → summarizer |
| `incident_response` | debugger → monitoring → devops → summarizer → documentation |

### Specialized
| Workflow | Pipeline |
|----------|----------|
| `ml_project` | researcher → ml_engineer → coder → tester → documentation |
| `regex_build` | regex_expert → tester → explainer |
| `prompt_craft` | prompt_engineer → tester → optimizer |
| `interview_prep` | researcher → interviewer → mentor |
| `onboarding` | explainer → mentor → diagrammer |

---

## Use Cases with Examples

### 🚀 Build a New Feature
```
/workflow full_dev implement user authentication with JWT and refresh tokens
```

### 🐛 Fix a Bug
```
/workflow bug_fix TypeError: Cannot read property 'map' of undefined in UserList.tsx
```

### 🏗️ Design a System
```
/workflow design design a real-time notification system for 100k users
```

### 📝 Write Documentation
```
/workflow docs document the payment processing module with API reference
```

### 🔒 Security Review
```
/files src/auth.py,src/middleware.py security audit these files
```

### ⚡ Optimize Performance
```
/workflow perf_audit our API response time is 2s, analyze and optimize
```

### 🎯 Direct Agent Call
```
/ask coder write a Python decorator for caching with TTL
/ask database design a schema for multi-tenant SaaS
/ask devops write a GitHub Actions CI/CD pipeline for a Node.js app
/ask shell_expert write a bash script to backup PostgreSQL daily
```

### 🔄 Iterative Refinement
```
/feedback coder reviewer write a thread-safe LRU cache in Python
```
*(Coder writes → reviewer critiques → coder improves → until approved)*

### 📊 Compare Approaches
```
/compare architect,optimizer design a caching strategy for product catalog
```

### ⚡ Parallel Execution
```
/parallel security,optimizer,accessibility audit this React component
```

### 📋 Batch Processing
```
/batch coder implement stack;;implement queue;;implement linked list;;implement BST
```

### 🤖 Auto-Routing
```
/auto our login endpoint is returning 500 errors in production
```
*(Automatically selects `bug_fix` workflow)*

### 📁 Multi-File Analysis
```
/files src/api.py,src/models.py,src/tests.py review for consistency issues
```

### 🎓 Learning
```
/workflow learn explain event-driven architecture with examples
/ask mentor explain the CAP theorem like I'm a junior developer
```

### 🚢 Production Release
```
/workflow release prepare v2.0 release for the payment service
```

### 🏢 Full Startup MVP
```
/workflow startup_mvp build a SaaS invoicing app with Stripe integration
```

---

## MCP Client Integration

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (Linux) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "multi-agent": {
      "command": "/home/jaspal/jscode/js-ai-apps-api/multi-agent-app/.venv/bin/python",
      "args": ["/home/jaspal/jscode/js-ai-apps-api/multi-agent-app/main.py", "--mcp-server"]
    }
  }
}
```

### Cursor

Add to Cursor MCP settings:
```json
{
  "multi-agent": {
    "command": "/home/jaspal/jscode/js-ai-apps-api/multi-agent-app/.venv/bin/python",
    "args": ["main.py", "--mcp-server"],
    "cwd": "/home/jaspal/jscode/js-ai-apps-api/multi-agent-app"
  }
}
```

### VS Code (Copilot MCP)

```json
{
  "mcp": {
    "servers": {
      "multi-agent": {
        "command": "python",
        "args": ["main.py", "--mcp-server"],
        "cwd": "/home/jaspal/jscode/js-ai-apps-api/multi-agent-app"
      }
    }
  }
}
```

### Any MCP Client (SSE/HTTP)

```bash
# Start HTTP server
python main.py --mcp-server --transport sse --port 8080

# Connect from any MCP client to: http://localhost:8080/sse
```

---

## Advanced Features

### 🔄 Feedback Loop
Iteratively refines output until approved:
```
/feedback coder reviewer write a production-ready connection pool
```
Agent writes → reviewer evaluates → agent improves → repeat (max 3 rounds).

### 🧠 Persistent Memory
Notes that survive across sessions:
```
/remember project Using PostgreSQL 15 with pgvector extension
/remember style snake_case, type hints, 4-space indent
/recall project
/forget project
```

### ⚡ Parallel Execution
Multiple agents simultaneously:
```
/parallel security,optimizer,reviewer analyze this module
```

### 📦 Batch Processing
Same agent, multiple tasks:
```
/batch tester write tests for login;;signup;;logout;;password-reset
```

### 🎯 Auto-Routing
Keyword-based workflow selection:
```
/auto deploy our app to kubernetes with monitoring
```

### 📁 Multi-File Context
Cross-file analysis:
```
/files src/api.py,src/models.py,tests/test_api.py find inconsistencies
```

### 🔀 Custom Chains
Build pipelines on the fly:
```
/chain planner|architect|coder|tester|documentation build a rate limiter
```

### 💾 Session Persistence
```
/save my-project
/load my-project
/export
```

### 📊 Token Tracking
```
/tokens
# Output: Tokens: ~12,450 total (4,200 in / 8,250 out) | Requests: 7
```

---

## Configuration Reference

### config.py

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `llama3.1:8b` | Default model |
| `TEMPERATURE` | `0.7` | Default temperature |
| `MAX_TOKENS` | `2048` | Max response tokens |
| `MCP_SERVER_NAME` | `MultiAgentSystem` | MCP server name |
| `MCP_SERVER_TRANSPORT` | `stdio` | Default transport |
| `MCP_SSE_HOST` | `0.0.0.0` | SSE bind address |
| `MCP_SSE_PORT` | `8080` | SSE port |
| `MCP_REQUEST_TIMEOUT` | `300` | Request timeout (seconds) |
| `EXTERNAL_MCP_SERVERS` | `[]` | External local MCP servers |

### Environment Variables (Ollama)

```bash
export OLLAMA_HOST=0.0.0.0:11434   # Bind address
export OLLAMA_NUM_GPU=999           # GPU layers
export OLLAMA_NUM_THREAD=8          # CPU threads
export OLLAMA_KEEP_ALIVE=5m         # Model keep-alive time
```

---

## Project Structure

```
multi-agent-app/
├── agent_registry.py   # 45 agent definitions (single source of truth)
├── config.py           # All settings + supervisor prompt
├── graph.py            # LangGraph supervisor orchestration
├── runners.py          # Execution modes + 50 workflows + advanced features
├── session.py          # History, tokens, save/load/export
├── main.py             # CLI dispatcher + entry point
├── mcp_server.py       # FastMCP server (dynamic tool registration)
├── tool_registry.py    # External MCP server consumption
├── requirements.txt    # Python dependencies
└── sessions/           # Saved sessions + agent memory
    ├── *.json          # Session files
    ├── *.md            # Exported conversations
    └── memory.json     # Persistent agent memory
```

---

## Extending the System

### Add a New Agent

Add one entry to `agent_registry.py`:

```python
"my_agent": {
    "description": "What it does",
    "temperature": 0.3,
    "prompt": "You are a ... agent. Your job is to: 1) ... 2) ... 3) ...",
},
```

Automatically available in: supervisor routing, `/ask`, `/chain`, MCP tools.

### Add a New Workflow

Add to `WORKFLOWS` in `runners.py`:

```python
"my_workflow": ["planner", "my_agent", "reviewer", "tester"],
```

### Add External MCP Server

Add to `EXTERNAL_MCP_SERVERS` in `config.py`:

```python
{"name": "my-server", "transport": "stdio", "command": "python", "args": ["my_server.py"]}
```

---

## Troubleshooting

### Ollama not reachable
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start it
ollama serve
```

### Model not found
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.1:8b
```

### Slow responses
- Use a smaller model: `/model mistral` or `/model phi3:mini`
- Reduce `MAX_TOKENS` in config.py
- Use GPU: install CUDA/ROCm drivers

### Out of memory
- Use smaller model: `llama3.1:8b` instead of `13b`/`70b`
- Close other applications
- Set `OLLAMA_NUM_GPU=0` to use CPU only (slower but less RAM)

### Import errors
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Requirements

- Python 3.12+
- Ollama (any version)
- 8GB+ RAM (16GB recommended)
- No GPU required
- **No internet after initial setup**

### Python Dependencies

```
langchain>=0.3.0
langchain-ollama>=0.2.0
langgraph>=0.2.0
pydantic>=2.0.0
mcp>=1.0.0
requests>=2.28.0
```

---

## License

MIT
