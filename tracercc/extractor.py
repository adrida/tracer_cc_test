"""
Tracer PLG — Claude Code session extractor.

Walks every Claude Code project at ~/.claude/projects/<encoded-cwd>/<session>.jsonl,
parses the line-delimited event stream and returns five normalized
pandas DataFrames suitable for analyzing model-selection waste:

    sessions   — one row per session
    messages   — one row per assistant/user message
    tool_calls — one row per tool_use, joined to its tool_result
    prompts    — one row per user-initiated prompt (free-text, not tool_result)
    errors     — one row per API/tool error

The parser is defensive: Claude Code's JSONL schema is mostly stable but
keeps adding event types (progress, system, ai-title, custom-title,
file-history-snapshot, queue-operation, last-prompt, permission-mode,
attachment, agent-name, ...). Anything unrecognized is ignored rather
than blowing up so the notebook stays usable as the schema evolves.

Usage:
    from extractor import load_all
    df = load_all()
    df.sessions, df.messages, df.tool_calls, df.prompts, df.errors
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

CLAUDE_HOME = Path(os.environ.get("CLAUDE_HOME", Path.home() / ".claude"))
PROJECTS_DIR = CLAUDE_HOME / "projects"


# --------------------------------------------------------------------------- #
# Low-level utilities
# --------------------------------------------------------------------------- #

def _decode_project_path(encoded: str) -> str:
    """Claude encodes the cwd into a folder name by replacing '/' with '-'.

    The mapping is lossy (a literal '-' in the path is indistinguishable
    from a separator), but for the common case of standard absolute paths
    this gives back something close to the original cwd.
    """
    if not encoded.startswith("-"):
        return encoded
    return "/" + encoded[1:].replace("-", "/")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        return None


def _content_blocks(message: dict | None) -> list[dict]:
    if not message:
        return []
    content = message.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _truncate(text: str | None, n: int = 500) -> str | None:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[:n] + "…"


def _stringify(obj: Any, n: int = 500) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate(obj, n)
    try:
        return _truncate(json.dumps(obj, ensure_ascii=False, default=str), n)
    except Exception:
        return _truncate(repr(obj), n)


# --------------------------------------------------------------------------- #
# Iteration over the on-disk projects tree
# --------------------------------------------------------------------------- #

def iter_session_files(projects_dir: Path = PROJECTS_DIR) -> Iterator[tuple[str, Path]]:
    """Yield (project_dir_name, jsonl_path) for every session log on disk."""
    if not projects_dir.exists():
        return
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for jsonl in sorted(project_dir.glob("*.jsonl")):
            yield project_dir.name, jsonl


def iter_events(jsonl_path: Path) -> Iterator[dict]:
    """Stream JSON events from a single session log, skipping bad lines."""
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# Per-session parsing
# --------------------------------------------------------------------------- #

@dataclass
class _ToolUse:
    """In-flight tool call recorded from an assistant message."""
    tool_use_id: str
    name: str
    input_preview: str | None
    input_size: int
    parent_assistant_uuid: str
    session_id: str
    project_dir: str
    model: str | None
    is_sidechain: bool
    permission_mode: str | None
    cwd: str | None
    git_branch: str | None
    request_id: str | None
    started_at: pd.Timestamp | None


def _parse_session(project_dir: str, jsonl_path: Path) -> dict[str, list[dict]]:
    """Parse one .jsonl into row lists for each output table."""
    messages: list[dict] = []
    tool_calls: list[dict] = []
    prompts: list[dict] = []
    errors: list[dict] = []

    pending_tools: dict[str, _ToolUse] = {}

    session_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    version: str | None = None
    entrypoint: str | None = None
    ai_title: str | None = None
    custom_title: str | None = None
    permission_modes: set[str] = set()

    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    queue_events = 0
    progress_events = 0
    system_events = 0
    snapshot_events = 0
    attachment_events = 0
    sidechain_msgs = 0
    n_compactions = 0
    n_local_commands = 0
    slash_commands: list[str] = []
    # uuid -> turn_duration_ms (parented by the assistant message uuid)
    turn_duration_by_parent: dict[str, int] = {}

    for ev in iter_events(jsonl_path):
        etype = ev.get("type")
        ts = _parse_ts(ev.get("timestamp"))
        if ts is not None:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

        if not session_id and ev.get("sessionId"):
            session_id = ev.get("sessionId")
        if not cwd and ev.get("cwd"):
            cwd = ev.get("cwd")
        if not git_branch and ev.get("gitBranch"):
            git_branch = ev.get("gitBranch")
        if not version and ev.get("version"):
            version = ev.get("version")
        if not entrypoint and ev.get("entrypoint"):
            entrypoint = ev.get("entrypoint")
        if ev.get("permissionMode"):
            permission_modes.add(ev["permissionMode"])

        if etype == "queue-operation":
            queue_events += 1
            continue
        if etype == "progress":
            progress_events += 1
            continue
        if etype == "system":
            system_events += 1
            subtype = ev.get("subtype")
            if subtype == "compact_boundary":
                n_compactions += 1
            elif subtype == "turn_duration":
                # parentUuid is the assistant message uuid this duration measures
                pu = ev.get("parentUuid")
                dur = ev.get("durationMs")
                if pu and isinstance(dur, (int, float)):
                    turn_duration_by_parent[pu] = int(dur)
            elif subtype == "local_command":
                content = ev.get("content") or ""
                # extract /command-name from "<command-name>/foo</command-name>..."
                m = re.search(r"<command-name>([^<]+)</command-name>", str(content))
                if m:
                    n_local_commands += 1
                    slash_commands.append(m.group(1).strip())
            elif subtype == "api_error":
                errors.append({
                    "session_id": ev.get("sessionId"),
                    "project_dir": project_dir,
                    "uuid": ev.get("uuid"),
                    "timestamp": ts,
                    "model": None,
                    "kind": "system_api_error",
                    "error": _truncate(ev.get("content"), 400),
                    "message_preview": None,
                })
            continue
        if etype == "file-history-snapshot":
            snapshot_events += 1
            continue
        if etype == "attachment":
            attachment_events += 1
            continue
        if etype == "ai-title":
            ai_title = ev.get("aiTitle") or ai_title
            continue
        if etype == "custom-title":
            custom_title = ev.get("customTitle") or custom_title
            continue
        if etype in {"last-prompt", "permission-mode", "agent-name"}:
            continue

        if etype not in {"user", "assistant"}:
            continue

        msg = ev.get("message", {}) or {}
        role = msg.get("role")
        model = msg.get("model")
        usage = msg.get("usage", {}) or {}
        blocks = _content_blocks(msg)

        is_sidechain = bool(ev.get("isSidechain"))
        if is_sidechain:
            sidechain_msgs += 1

        n_text = sum(1 for b in blocks if b.get("type") == "text")
        n_thinking = sum(1 for b in blocks if b.get("type") == "thinking")
        n_tool_use = sum(1 for b in blocks if b.get("type") == "tool_use")
        n_tool_result = sum(1 for b in blocks if b.get("type") == "tool_result")

        text_chars = sum(len(b.get("text") or "") for b in blocks if b.get("type") == "text")
        thinking_chars = sum(len(b.get("thinking") or "") for b in blocks if b.get("type") == "thinking")

        first_text = next(
            (b.get("text") for b in blocks if b.get("type") == "text" and b.get("text")),
            None,
        )

        messages.append({
            "session_id": ev.get("sessionId"),
            "project_dir": project_dir,
            "uuid": ev.get("uuid"),
            "parent_uuid": ev.get("parentUuid"),
            "role": role,
            "type": etype,
            "model": model,
            "timestamp": ts,
            "request_id": ev.get("requestId"),
            "stop_reason": msg.get("stop_reason"),
            "is_sidechain": is_sidechain,
            "is_api_error": bool(ev.get("isApiErrorMessage")),
            "error": ev.get("error"),
            "permission_mode": ev.get("permissionMode"),
            "prompt_id": ev.get("promptId"),
            "entrypoint": ev.get("entrypoint"),
            "cwd": ev.get("cwd"),
            "git_branch": ev.get("gitBranch"),
            "version": ev.get("version"),
            "user_type": ev.get("userType"),
            "n_text_blocks": n_text,
            "n_thinking_blocks": n_thinking,
            "n_tool_use_blocks": n_tool_use,
            "n_tool_result_blocks": n_tool_result,
            "text_chars": text_chars,
            "thinking_chars": thinking_chars,
            "first_text_preview": _truncate(first_text, 280),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "service_tier": usage.get("service_tier"),
            "turn_duration_ms": None,  # filled in post-loop from system.turn_duration events
        })

        if ev.get("error") or ev.get("isApiErrorMessage"):
            errors.append({
                "session_id": ev.get("sessionId"),
                "project_dir": project_dir,
                "uuid": ev.get("uuid"),
                "timestamp": ts,
                "model": model,
                "kind": "api_error",
                "error": ev.get("error"),
                "message_preview": _truncate(first_text, 400),
            })

        if etype == "user":
            prompt_id = ev.get("promptId")
            if prompt_id and role == "user":
                text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                joined = "\n".join(p for p in text_parts if p).strip()
                if joined:
                    prompts.append({
                        "session_id": ev.get("sessionId"),
                        "project_dir": project_dir,
                        "uuid": ev.get("uuid"),
                        "prompt_id": prompt_id,
                        "timestamp": ts,
                        "char_len": len(joined),
                        "preview": _truncate(joined, 600),
                        "permission_mode": ev.get("permissionMode"),
                        "git_branch": ev.get("gitBranch"),
                        "cwd": ev.get("cwd"),
                    })

            for b in blocks:
                if b.get("type") != "tool_result":
                    continue
                tu_id = b.get("tool_use_id")
                pending = pending_tools.pop(tu_id, None) if tu_id else None
                tur = ev.get("toolUseResult")
                stdout = stderr = None
                interrupted = is_image = no_output_expected = None
                duration_ms = total_tokens = total_tool_use_count = None
                agent_id = agent_type = None
                if isinstance(tur, dict):
                    stdout = tur.get("stdout")
                    stderr = tur.get("stderr")
                    interrupted = tur.get("interrupted")
                    is_image = tur.get("isImage")
                    no_output_expected = tur.get("noOutputExpected")
                    duration_ms = tur.get("totalDurationMs") or tur.get("durationMs")
                    total_tokens = tur.get("totalTokens")
                    total_tool_use_count = tur.get("totalToolUseCount")
                    agent_id = tur.get("agentId")
                    agent_type = tur.get("agentType")

                latency_ms = None
                if pending and ts is not None and pending.started_at is not None:
                    latency_ms = (ts - pending.started_at).total_seconds() * 1000.0

                content = b.get("content")
                if isinstance(content, list):
                    result_text = "\n".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    result_text = content if isinstance(content, str) else _stringify(content, 600)

                tool_calls.append({
                    "session_id": ev.get("sessionId"),
                    "project_dir": project_dir,
                    "tool_use_id": tu_id,
                    "tool_name": pending.name if pending else None,
                    "model": pending.model if pending else None,
                    "is_sidechain": pending.is_sidechain if pending else is_sidechain,
                    "parent_assistant_uuid": pending.parent_assistant_uuid if pending else ev.get("sourceToolAssistantUUID"),
                    "request_id": pending.request_id if pending else None,
                    "started_at": pending.started_at if pending else None,
                    "ended_at": ts,
                    "wallclock_latency_ms": latency_ms,
                    "reported_duration_ms": duration_ms,
                    "input_preview": pending.input_preview if pending else None,
                    "input_size": pending.input_size if pending else None,
                    "is_error": bool(b.get("is_error")),
                    "interrupted": interrupted,
                    "is_image": is_image,
                    "no_output_expected": no_output_expected,
                    "result_preview": _truncate(result_text, 600),
                    "stdout_preview": _truncate(stdout, 400),
                    "stderr_preview": _truncate(stderr, 400),
                    "subagent_total_tokens": total_tokens,
                    "subagent_tool_use_count": total_tool_use_count,
                    "subagent_id": agent_id,
                    "subagent_type": agent_type,
                    "permission_mode": pending.permission_mode if pending else ev.get("permissionMode"),
                    "cwd": pending.cwd if pending else ev.get("cwd"),
                    "git_branch": pending.git_branch if pending else ev.get("gitBranch"),
                })

                if b.get("is_error"):
                    errors.append({
                        "session_id": ev.get("sessionId"),
                        "project_dir": project_dir,
                        "uuid": ev.get("uuid"),
                        "timestamp": ts,
                        "model": pending.model if pending else None,
                        "kind": "tool_error",
                        "error": _truncate(result_text, 400),
                        "message_preview": pending.name if pending else None,
                    })

        elif etype == "assistant":
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                tu_id = b.get("id")
                if not tu_id:
                    continue
                tool_input = b.get("input")
                pending_tools[tu_id] = _ToolUse(
                    tool_use_id=tu_id,
                    name=b.get("name") or "?",
                    input_preview=_stringify(tool_input, 600),
                    input_size=len(json.dumps(tool_input, default=str)) if tool_input is not None else 0,
                    parent_assistant_uuid=ev.get("uuid"),
                    session_id=ev.get("sessionId"),
                    project_dir=project_dir,
                    model=model,
                    is_sidechain=is_sidechain,
                    permission_mode=ev.get("permissionMode"),
                    cwd=ev.get("cwd"),
                    git_branch=ev.get("gitBranch"),
                    request_id=ev.get("requestId"),
                    started_at=ts,
                )

    # Backfill turn_duration_ms from system.turn_duration events keyed by parentUuid
    if turn_duration_by_parent:
        for m in messages:
            d = turn_duration_by_parent.get(m.get("uuid"))
            if d is not None:
                m["turn_duration_ms"] = d

    # Tool calls that never received a result (interrupted, timed out, etc.)
    for tu in pending_tools.values():
        tool_calls.append({
            "session_id": tu.session_id,
            "project_dir": tu.project_dir,
            "tool_use_id": tu.tool_use_id,
            "tool_name": tu.name,
            "model": tu.model,
            "is_sidechain": tu.is_sidechain,
            "parent_assistant_uuid": tu.parent_assistant_uuid,
            "request_id": tu.request_id,
            "started_at": tu.started_at,
            "ended_at": None,
            "wallclock_latency_ms": None,
            "reported_duration_ms": None,
            "input_preview": tu.input_preview,
            "input_size": tu.input_size,
            "is_error": False,
            "interrupted": True,
            "is_image": None,
            "no_output_expected": None,
            "result_preview": None,
            "stdout_preview": None,
            "stderr_preview": None,
            "subagent_total_tokens": None,
            "subagent_tool_use_count": None,
            "subagent_id": None,
            "subagent_type": None,
            "permission_mode": tu.permission_mode,
            "cwd": tu.cwd,
            "git_branch": tu.git_branch,
        })

    session_row = {
        "session_id": session_id or jsonl_path.stem,
        "project_dir": project_dir,
        "decoded_cwd": cwd or _decode_project_path(project_dir),
        "git_branch": git_branch,
        "version": version,
        "entrypoint": entrypoint,
        "ai_title": ai_title,
        "custom_title": custom_title,
        "first_event_at": first_ts,
        "last_event_at": last_ts,
        "duration_seconds": (
            (last_ts - first_ts).total_seconds() if first_ts is not None and last_ts is not None else None
        ),
        "n_messages": len(messages),
        "n_assistant": sum(1 for m in messages if m["type"] == "assistant"),
        "n_user": sum(1 for m in messages if m["type"] == "user"),
        "n_sidechain_messages": sidechain_msgs,
        "n_tool_calls": len(tool_calls),
        "n_tool_errors": sum(1 for t in tool_calls if t["is_error"]),
        "n_api_errors": sum(1 for e in errors if e["kind"] == "api_error"),
        "n_prompts": len(prompts),
        "n_progress_events": progress_events,
        "n_queue_events": queue_events,
        "n_system_events": system_events,
        "n_attachment_events": attachment_events,
        "n_snapshot_events": snapshot_events,
        "n_compactions": n_compactions,
        "n_local_commands": n_local_commands,
        "slash_commands": ",".join(slash_commands) or None,
        "permission_modes": ",".join(sorted(permission_modes)) or None,
        "file_path": str(jsonl_path),
        "file_size_bytes": jsonl_path.stat().st_size,
    }

    return {
        "session": [session_row],
        "messages": messages,
        "tool_calls": tool_calls,
        "prompts": prompts,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Top-level loader
# --------------------------------------------------------------------------- #

@dataclass
class Tables:
    sessions: pd.DataFrame
    messages: pd.DataFrame
    tool_calls: pd.DataFrame
    prompts: pd.DataFrame
    errors: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"Tables(sessions={len(self.sessions)}, messages={len(self.messages)}, "
            f"tool_calls={len(self.tool_calls)}, prompts={len(self.prompts)}, "
            f"errors={len(self.errors)})"
        )


def load_all(projects_dir: Path = PROJECTS_DIR) -> Tables:
    """Parse every session JSONL under projects_dir and return the 5 tables."""
    sessions: list[dict] = []
    messages: list[dict] = []
    tool_calls: list[dict] = []
    prompts: list[dict] = []
    errors: list[dict] = []

    for project_dir, jsonl in iter_session_files(projects_dir):
        parsed = _parse_session(project_dir, jsonl)
        sessions.extend(parsed["session"])
        messages.extend(parsed["messages"])
        tool_calls.extend(parsed["tool_calls"])
        prompts.extend(parsed["prompts"])
        errors.extend(parsed["errors"])

    sessions_df = pd.DataFrame(sessions)
    messages_df = pd.DataFrame(messages)
    tool_calls_df = pd.DataFrame(tool_calls)
    prompts_df = pd.DataFrame(prompts)
    errors_df = pd.DataFrame(errors)

    for df, col in [
        (sessions_df, "first_event_at"),
        (sessions_df, "last_event_at"),
        (messages_df, "timestamp"),
        (tool_calls_df, "started_at"),
        (tool_calls_df, "ended_at"),
        (prompts_df, "timestamp"),
        (errors_df, "timestamp"),
    ]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if not messages_df.empty:
        messages_df = messages_df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    if not tool_calls_df.empty:
        tool_calls_df = tool_calls_df.sort_values(["session_id", "started_at"]).reset_index(drop=True)
    if not sessions_df.empty:
        sessions_df = sessions_df.sort_values("first_event_at", na_position="last").reset_index(drop=True)

    return Tables(sessions_df, messages_df, tool_calls_df, prompts_df, errors_df)
