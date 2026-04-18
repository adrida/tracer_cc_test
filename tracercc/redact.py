"""Redaction layer.

Takes the rich ``Tables`` produced by source extractors (which include raw
prompt + response text in ``messages.content_text`` etc.) and returns
serialisable dicts with only the fields the backend actually consumes.

Every text-bearing column on messages and prompts is dropped here. Tool
call inputs are kept (they're what gets embedded for clustering) but
truncated to 500 chars.

This module is the contract surface that gets audited if a user asks
"what data leaves my machine?".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


# Fields kept on each message — explicitly NOT including any text content.
_KEEP_MESSAGE = (
    "uuid", "session_id", "type", "model", "timestamp",
    "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens",
    "n_text_blocks", "n_thinking_blocks", "n_tool_use_blocks",
    "project_dir",
)

_INT_MESSAGE_FIELDS = {
    "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens",
    "n_text_blocks", "n_thinking_blocks", "n_tool_use_blocks",
}

_KEEP_SESSION = (
    "session_id", "project_dir", "decoded_cwd",
    "first_event_at", "last_event_at", "n_compactions",
)

_KEEP_TOOL_CALL = (
    "tool_use_id", "session_id", "parent_assistant_uuid",
    "tool_name", "input_preview", "is_error", "started_at",
)

_KEEP_PROMPT = ("uuid", "session_id", "timestamp")
_KEEP_ERROR = ("session_id", "timestamp", "error_type")


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return v


def _row_to_dict(row: pd.Series, keep: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for k in keep:
        if k not in row.index:
            continue
        out[k] = _clean(row.get(k))
    return out


def redact_messages(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    cols = [c for c in _KEEP_MESSAGE if c in df.columns]
    for _, r in df[cols].iterrows():
        d: dict[str, Any] = {}
        for k in cols:
            v = _clean(r.get(k))
            if k in _INT_MESSAGE_FIELDS and v is not None:
                try:
                    v = int(v)
                except Exception:
                    v = None
            d[k] = v
        d["uuid"] = str(d.get("uuid"))
        d["session_id"] = str(d.get("session_id"))
        d["type"] = str(d.get("type") or "")
        if d.get("model") is not None:
            d["model"] = str(d["model"])
        out.append(d)
    return out


def redact_sessions(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    cols = [c for c in _KEEP_SESSION if c in df.columns]
    for _, r in df[cols].iterrows():
        d = _row_to_dict(r, tuple(cols))
        if "n_compactions" in d:
            try:
                d["n_compactions"] = int(d["n_compactions"] or 0)
            except Exception:
                d["n_compactions"] = 0
        d["session_id"] = str(d.get("session_id") or "")
        out.append(d)
    return out


def redact_tool_calls(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    cols = [c for c in _KEEP_TOOL_CALL if c in df.columns]
    for _, r in df[cols].iterrows():
        d = _row_to_dict(r, tuple(cols))
        d["session_id"] = str(d.get("session_id") or "")
        d["parent_assistant_uuid"] = str(d.get("parent_assistant_uuid") or "")
        if d.get("tool_use_id") is not None:
            d["tool_use_id"] = str(d["tool_use_id"])
        if d.get("input_preview") is not None:
            # Truncate aggressively. Tool inputs aren't user prompts but they
            # can carry file paths / snippets we want to keep small.
            d["input_preview"] = str(d["input_preview"])[:500]
        d["is_error"] = bool(d.get("is_error") or False)
        out.append(d)
    return out


def redact_prompts(df: pd.DataFrame) -> list[dict]:
    """Drop every prompt's text content; keep only timestamp + char_count."""
    if df.empty:
        return []
    out = []
    has_preview = "preview" in df.columns
    for _, r in df.iterrows():
        d = {
            "uuid": str(r.get("uuid") or ""),
            "session_id": str(r.get("session_id") or ""),
            "timestamp": _clean(r.get("timestamp")),
            "char_count": int(len(str(r.get("preview") or ""))) if has_preview else 0,
        }
        out.append(d)
    return out


def redact_errors(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    has_type = "error_type" in df.columns
    for _, r in df.iterrows():
        d = {
            "session_id": str(r.get("session_id") or ""),
            "timestamp": _clean(r.get("timestamp")),
            "error_type": (str(r.get("error_type")) if has_type and r.get("error_type") is not None else None),
        }
        out.append(d)
    return out


@dataclass
class RedactedPayload:
    sessions: list[dict]
    messages: list[dict]
    tool_calls: list[dict]
    prompts: list[dict]
    errors: list[dict]

    def to_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "prompts": self.prompts,
            "errors": self.errors,
        }


def redact_tables(tables) -> RedactedPayload:
    """Convert a ``Tables`` object (with pandas DataFrames) into a
    privacy-redacted, JSON-serialisable payload for the analysis backend."""
    return RedactedPayload(
        sessions=redact_sessions(tables.sessions),
        messages=redact_messages(tables.messages),
        tool_calls=redact_tool_calls(tables.tool_calls),
        prompts=redact_prompts(tables.prompts),
        errors=redact_errors(tables.errors),
    )
