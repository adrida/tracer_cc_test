"""Source extractors — one per agentic system we support.

Each source returns the same ``Tables`` dataclass (``sessions``,
``messages``, ``tool_calls``, ``prompts``, ``errors``) so the rest of
the client treats them identically.

Built-in sources:
    - ``claude_code``: Anthropic's Claude Code (~/.claude/projects/)
    - ``cursor``:      Cursor (~/.cursor/projects + ai-tracking SQLite)
    - ``custom``:      generic JSONL events under ``$TRACERCC_CUSTOM_DIR``
                       (or ``./traces/custom``). Use this for any other
                       agentic system — see ``custom.py`` for the schema.
"""

from __future__ import annotations

from pathlib import Path

from . import claude_code, cursor, custom


SOURCES: dict[str, object] = {
    "claude_code": claude_code,
    "cursor":      cursor,
    "custom":      custom,
}


def _has_files(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    return any(d.iterdir())


def _has_jsonl(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    return any(d.rglob("*.jsonl"))


def autodetect() -> str | None:
    """Return the source whose data dir exists. Order of preference:

        1. ``custom`` if ``./traces/custom`` (or ``$TRACERCC_CUSTOM_DIR``)
           contains any ``*.jsonl`` — explicit user intent wins.
        2. ``claude_code`` if ``~/.claude/projects/`` exists and is non-empty.
        3. ``cursor``      if ``~/.cursor/projects/`` exists and is non-empty.

    Returns None if none of the above match.
    """
    if _has_jsonl(custom.PROJECTS_DIR):
        return "custom"
    if _has_files(claude_code.PROJECTS_DIR):
        return "claude_code"
    if _has_files(cursor.PROJECTS_DIR):
        return "cursor"
    return None


def load(source: str):
    """Load tables from the named source. Raises KeyError on unknown source."""
    return SOURCES[source].load_all()
