"""Source extractors — one per agentic IDE we support.

Each source returns the same ``Tables`` dataclass (``sessions``,
``messages``, ``tool_calls``, ``prompts``, ``errors``) so the rest of
the client treats them identically.
"""

from __future__ import annotations

from pathlib import Path

from . import claude_code, cursor


SOURCES: dict[str, object] = {
    "claude_code": claude_code,
    "cursor":      cursor,
}


def autodetect() -> str | None:
    """Return the source whose data dir exists. Prefers Claude Code if both
    are present (it's where this tool started). Returns None if neither is
    found."""
    if claude_code.PROJECTS_DIR.exists() and any(
        claude_code.PROJECTS_DIR.iterdir() if claude_code.PROJECTS_DIR.is_dir() else []
    ):
        return "claude_code"
    if cursor.PROJECTS_DIR.exists() and any(
        cursor.PROJECTS_DIR.iterdir() if cursor.PROJECTS_DIR.is_dir() else []
    ):
        return "cursor"
    return None


def load(source: str):
    """Load tables from the named source. Raises KeyError on unknown source."""
    return SOURCES[source].load_all()
