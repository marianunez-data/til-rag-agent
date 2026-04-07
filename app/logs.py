"""
logs.py
-------
Structured interaction logging for the TIL agent.
Persists every agent run to a timestamped JSON file for downstream
evaluation and analysis.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

LOG_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _default_serializer(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Log construction
# ---------------------------------------------------------------------------

def build_log_entry(
    agent: Agent,
    messages: list,
    source: str = "user",
) -> dict:
    """
    Construct a structured log entry from an agent run.

    Args:
        agent:    The Pydantic AI Agent instance that produced the run.
        messages: Message list from result.new_messages().
        source:   Origin of the query ('user' or 'ai-generated').

    Returns:
        Dict containing agent metadata and the full message history.
    """
    tools: list[str] = []
    for toolset in agent.toolsets:
        tools.extend(toolset.tools.keys())

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "model": "llama3.2",
        "tools": tools,
        "source": source,
        "messages": ModelMessagesTypeAdapter.dump_python(messages),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_interaction(
    agent: Agent,
    messages: list,
    source: str = "user",
    log_dir: Path = LOG_DIR,
) -> Path:
    """
    Persist an agent interaction to a JSON file.

    The filename encodes the agent name, a UTC timestamp, and a short
    random hex suffix to avoid collisions.

    Args:
        agent:    The Pydantic AI Agent instance.
        messages: Message list from result.new_messages().
        source:   Origin of the query.
        log_dir:  Directory where log files are written.

    Returns:
        Path to the written log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = build_log_entry(agent, messages, source)

    try:
        raw_ts = entry["messages"][-1].get("timestamp")
        if isinstance(raw_ts, str):
            ts_obj = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        else:
            ts_obj = datetime.utcnow()
    except Exception:
        ts_obj = datetime.utcnow()

    filename = f"{agent.name}_{ts_obj.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}.json"
    filepath = log_dir / filename

    with filepath.open("w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, default=_default_serializer)

    return filepath


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_log(path: Path | str) -> dict:
    """
    Load a previously saved interaction log.

    Args:
        path: Path to the JSON log file.

    Returns:
        Log dict with an additional 'log_file' key set to the Path object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["log_file"] = path
    return data


def load_all_logs(
    log_dir: Path = LOG_DIR,
    agent_name: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """
    Load and optionally filter all logs from a directory.

    Args:
        log_dir:    Directory to scan.
        agent_name: If set, only return logs for this agent.
        source:     If set, only return logs with this source value.

    Returns:
        List of log dicts.
    """
    logs = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            entry = load_log(path)
        except Exception:
            continue
        if agent_name and agent_name not in path.name:
            continue
        if source and entry.get("source") != source:
            continue
        logs.append(entry)
    return logs
