"""Mechanical-tool detection + label inference.

A "mechanical-only" assistant turn is one whose entire output is a tool call
to a tool that doesn't require reasoning to drive (Bash, Read, Grep, skill
lookup, memory read/write, etc.). These are the turns that *could* have been
served by a cheaper model without hurting quality — the foundation of the
counterfactual savings story.

The whitelist deliberately spans both Claude-Code-native tool names and the
generic tool names emitted by non-Anthropic agent frameworks (Hermes, OpenAI
function-calling, OpenRouter, LangGraph, LlamaIndex, Autogen). If your agent's
tool isn't in the set but is clearly mechanical, add it here; the structural
gate fails closed, so unknown tools are treated as reasoning by default.
"""

from __future__ import annotations

import json

import pandas as pd


MECHANICAL_TOOLS: set[str] = {
    # Claude Code / Cursor native
    "Bash", "Read", "Glob", "Grep", "Edit", "Write", "MultiEdit",
    "ls", "LS", "TodoWrite", "TaskOutput", "TaskUpdate", "TaskStop",
    "TaskCreate", "ExitPlanMode", "EnterPlanMode", "NotebookEdit", "NotebookRead",

    # Generic agent-framework names (Hermes, LangGraph, OpenAI function-calling,
    # Autogen, LlamaIndex, OpenRouter, etc.). Kept in the whitelist because the
    # semantic category is mechanical even though the name varies.
    "run_command", "shell", "execute", "exec", "bash", "sh",
    "read_file", "readFile", "view_file", "file_read",
    "write_file", "writeFile", "file_write", "create_file",
    "edit_file", "editFile", "modify_file",
    "delete_file", "remove_file", "move_file", "rename_file", "copy_file",
    "list_files", "listFiles", "list_dir", "listDir",
    "search_files", "searchFiles", "search_code", "code_search", "grep",
    "glob", "find_files",

    # Skill / memory / state-management primitives seen in Hermes-style agents
    "skill_manage", "skill_view", "skill_create", "skill_list", "skill_search",
    "memory", "memory_add", "memory_read", "memory_write", "memory_list",
    "memory_search", "memory_delete", "note", "todo",

    # HTTP / scrape primitives (mechanical iff used for fetching, not reasoning)
    "http_get", "fetch", "curl", "wget", "web_fetch",
}


def is_mechanical_assistant_turn(
    message_uuid: str,
    messages_df: pd.DataFrame,
    tool_calls_df: pd.DataFrame,
) -> bool:
    """True iff the assistant turn emitted only tool_use blocks targeting
    MECHANICAL_TOOLS and zero text or thinking blocks."""
    sub = messages_df[messages_df["uuid"] == message_uuid]
    if sub.empty:
        return False
    row = sub.iloc[0]
    if (row.get("n_text_blocks") or 0) > 0:
        return False
    if (row.get("n_thinking_blocks") or 0) > 0:
        return False
    if (row.get("n_tool_use_blocks") or 0) == 0:
        return False
    tcs = tool_calls_df[tool_calls_df["parent_assistant_uuid"] == message_uuid]
    if tcs.empty:
        return False
    return all(tn in MECHANICAL_TOOLS for tn in tcs["tool_name"].dropna())


def mechanical_turn_text(tool_name: str | None, input_preview: str | None) -> str:
    """Normalise a tool call's payload into a short string suitable for embedding.

    Near-identical mechanical operations (50× ``Bash: ls`` on different paths,
    30× ``memory.add`` with variant content) should embed close together so
    density clustering groups them.
    """
    name = tool_name or "?"
    raw = input_preview or "{}"
    try:
        d = json.loads(raw)
    except Exception:
        return f"{name}: {str(raw)[:200]}"
    if not isinstance(d, dict):
        return f"{name}: {str(d)[:200]}"
    # Claude Code-native shapes with specific fields worth pinning
    if name in ("Bash", "run_command", "shell", "execute", "exec", "bash", "sh"):
        return f"{name}: {str(d.get('command', d.get('cmd', '')))[:300]}"
    if name in ("Read", "read_file", "view_file", "file_read", "readFile"):
        return f"{name}: {d.get('file_path', d.get('path', ''))}"
    if name == "NotebookEdit":
        return f"NotebookEdit: {d.get('notebook_path', '')}"
    if name in ("Glob", "glob", "find_files"):
        return f"{name}: pattern={d.get('pattern', '')} path={d.get('path', '')}"
    if name in ("Grep", "search_files", "search_code", "code_search", "grep", "searchFiles"):
        return (
            f"{name}: pattern={str(d.get('pattern', d.get('query', '')))[:120]} "
            f"path={d.get('path', '')}"
        )
    # skill / memory: action + first-tier arg (name/content) is usually the
    # clustering signal
    if name.startswith("skill_") or name.startswith("memory"):
        action = d.get("action", "")
        payload = d.get("name") or d.get("content") or d.get("query") or ""
        return f"{name}: {action} {str(payload)[:180]}"
    # fallback: dump the dict
    return f"{name}: {str(d)[:200]}"


def label_from_medoid(text: str, tool: str) -> str:
    """Heuristic short label so the dashboard reads naturally.

    Pattern matches are English- and unix-command-biased, which is intentional:
    most mechanical tool-calls across agent frameworks bottom out in shell-ish
    invocations. Add heuristics here as you see new patterns in the wild.
    """
    t = (text or "").lower()
    if tool == "TaskOutput" or "taskoutput" in t:
        return "Subagent output polling"
    if "ps aux" in t and "grep" in t:
        return "Process monitoring"
    if "tail -" in t and "log" in t:
        return "Log tailing / experiment monitoring"
    if "ls -la" in t or "bash: ls " in t:
        return "Directory inspection"
    if "pip install" in t and "pytest" in t:
        return "pip install + pytest"
    if "pip install" in t:
        return "pip install"
    if "pytest" in t:
        return "pytest invocations"
    if "git " in t and ("status" in t or "diff" in t or "log" in t):
        return "git status / diff / log"
    if "git push" in t or "git commit" in t:
        return "git push / commit"
    if "dig " in t or "curl -" in t:
        return "DNS / curl checks"
    if "python -c" in t or 'python\\\\ -c' in t:
        return "one-liner Python invocations"
    if "json.loads" in t or "json.load(" in t:
        return "JSONL parsing one-liners"
    if "mkdir" in t or " cp " in t or " mv " in t or " rm " in t:
        return "filesystem housekeeping"
    if t.startswith("read:") or t.startswith("read_file:"):
        return "File reads"
    if t.startswith("grep:") or t.startswith("search"):
        return "Code search"
    if t.startswith("glob:"):
        return "File globbing"
    if t.startswith("skill_manage"):
        return "Skill management"
    if t.startswith("skill_view"):
        return "Skill lookup"
    if t.startswith("memory"):
        return "Memory read/write"
    if tool == "TodoWrite":
        return "TodoWrite housekeeping"
    return f"{tool} pattern"
