"""Configuration for the multi-agent application."""

from agent_registry import get_supervisor_agent_list

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# MCP Server settings
MCP_SERVER_NAME = "MultiAgentSystem"
MCP_SERVER_TRANSPORT = "stdio"
MCP_SSE_HOST = "0.0.0.0"
MCP_SSE_PORT = 8080
MCP_REQUEST_TIMEOUT = 300

# External MCP server configurations.
#
# SECURITY: entries here launch local processes (stdio transport runs
# `command` + `args` as a subprocess) or open network connections (sse `url`).
# This list is trusted, operator-controlled configuration. Do NOT populate it
# from untrusted input (user data, network, uploaded files) — doing so is
# equivalent to arbitrary local command execution. If it must ever be sourced
# dynamically, validate `command` against an allowlist first.
EXTERNAL_MCP_SERVERS: list[dict] = []

# Supervisor prompt (dynamically includes all registered agents)
SUPERVISOR_SYSTEM_PROMPT = f"""You are a supervisor agent that coordinates a team of specialized agents.
Your job is to analyze the user's request and decide which agent(s) should handle it.

Available agents:
{get_supervisor_agent_list()}

Respond with ONLY a JSON object in this format:
{{"next": "agent_name", "reason": "brief explanation"}}

If the task is complete and you have a final answer, respond with:
{{"next": "FINISH", "final_answer": "your synthesized response"}}
"""
