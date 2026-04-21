"""Custom JSONL source — feed tracerCC traces from any agentic system.

Use this source when your agent isn't Claude Code or Cursor. Emit a small
set of JSONL events and the full Tracer pipeline (BGE-M3 embeddings →
density clustering → per-cluster counterfactual re-pricing) runs unchanged.

Layout on disk
--------------
Root defaults to ``$TRACERCC_CUSTOM_DIR``, else ``./traces/custom``.
The loader walks it recursively for ``*.jsonl`` files:

    <root>/<project_name>/<session>.jsonl   → project_dir = <project_name>
    <root>/<session>.jsonl                  → project_dir = "custom"

A single .jsonl may contain many sessions (e.g. OTel batch export); they
are split by ``session_id`` before parsing.

JSONL event schema (one JSON object per line)
---------------------------------------------
Only ``event`` + ``session_id`` are required on every row. Everything else
is optional — the more you provide the better the dashboard.

    session_start    (optional, recommended)
    {"event":"session_start", "session_id":"s1",
     "ts":"2025-01-15T10:00:00Z",
     "model":"claude-opus-4-7", "cwd":"/repo", "entrypoint":"my-agent"}

    user_prompt
    {"event":"user_prompt", "session_id":"s1",
     "ts":"2025-01-15T10:00:01Z", "text":"please refactor X", "prompt_id":"p1"}

    assistant_message  (text-bearing turn — reasoning output)
    {"event":"assistant_message", "session_id":"s1",
     "ts":"2025-01-15T10:00:02Z", "model":"claude-opus-4-7",
     "text":"sure, I'll start by reading the file",
     "input_tokens":1234, "output_tokens":89}

    assistant_message  (tool-only turn — purely mechanical)
    {"event":"assistant_message", "session_id":"s1",
     "ts":"2025-01-15T10:00:03Z", "model":"claude-opus-4-7",
     "input_tokens":3100, "output_tokens":0}

    tool_call  (one event per tool invocation, after the assistant_message)
    {"event":"tool_call", "session_id":"s1",
     "ts":"2025-01-15T10:00:03Z", "model":"claude-opus-4-7",
     "tool_name":"Bash", "tool_input":{"command":"ls -la /repo"},
     "tool_use_id":"t1", "duration_ms":120,
     "is_error":false, "result_preview":"file1\\nfile2"}

    error
    {"event":"error", "session_id":"s1",
     "ts":"2025-01-15T10:00:04Z", "kind":"api_error", "message":"rate limited"}

    session_end  (optional)
    {"event":"session_end", "session_id":"s1", "ts":"2025-01-15T10:01:00Z"}

Key rules for mechanical detection (the Tracer thesis)
-------------------------------------------------------
A turn counts as "mechanical" — eligible for re-pricing to a cheaper model —
iff its assistant_message row has:
    - n_text_blocks  == 0   (no reasoning output)
    - n_thinking_blocks == 0
    - n_tool_use_blocks >= 1

This means: emit a *text-free* assistant_message immediately before each
batch of tool_call events. Text-bearing assistant messages are automatically
excluded from the savings calculation.

Accepted aliases (for OTel / custom log formats)
-------------------------------------------------
    ts / timestamp / time / started_at   → timestamp
    role: "user"                         → user_prompt
    role: "assistant"                    → assistant_message
    content / message / preview          → text
    name                                 → tool_name
    input                                → tool_input
    id                                   → tool_use_id
"""

from __future__ import annotations

import json
import os
import uuid as _uuid
from pathlib import Path
from typing import Iterator

import pandas as pd

from .. import extractor as cc

CUSTOM_DIR = Path(os.environ.get(
    "TRACERCC_CUSTOM_DIR",
    str(Path.cwd() / "traces" / "custom"),
))
PROJECTS_DIR = CUSTOM_DIR  # uniform API for autodetect

# Model + token fallbacks for agent exports that don't carry them inline
# (e.g. Hermes OpenAI-shaped sessions v2 — ``role: user / assistant / tool``
# with timestamps only). ``TRACERCC_DEFAULT_MODEL`` tags every message whose
# event has no ``model`` field so the family-aware mechanical gate can still
# fire. ``TRACERCC_APPROX_TOKENS`` (default on) estimates input/output tokens
# from content + reasoning + tool-result + tools-schema char counts when
# ``input_tokens`` / ``output_tokens`` are absent. The dashboard surfaces this
# as "(approximate)". Default chars-per-token ratio is 3.5 (mixed English +
# code/JSON, closer to OpenAI's real tokenisation than the English-only 4.0).
DEFAULT_MODEL = os.environ.get("TRACERCC_DEFAULT_MODEL", "gpt-5-2")
APPROX_TOKENS = os.environ.get("TRACERCC_APPROX_TOKENS", "1").lower() not in ("0", "false", "no")
CHARS_PER_TOKEN = float(os.environ.get("TRACERCC_CHARS_PER_TOKEN", "3.5"))


# --------------------------------------------------------------------------- #
# Event normalisation helpers
# --------------------------------------------------------------------------- #

_EVENT_ALIASES: dict[str, str] = {
    "user":           "user_prompt",
    "user_message":   "user_prompt",
    "prompt":         "user_prompt",
    "assistant":      "assistant_message",
    "assistant_text": "assistant_message",
    "tool_use":       "tool_call",
    "tool":           "tool_call",
    "tool_result":    "tool_result",
    "session":        "session_start",
    "session_open":   "session_start",
    "session_meta":   "session_start",
    "session_close":  "session_end",
}


def _norm_event(ev: dict) -> str | None:
    raw = ev.get("event") or ev.get("type")
    if isinstance(raw, str):
        return _EVENT_ALIASES.get(raw.strip().lower(), raw.strip().lower())
    role = ev.get("role")
    if isinstance(role, str):
        r = role.strip().lower()
        if r == "user":
            return "user_prompt"
        if r == "assistant":
            return "assistant_message"
        if r == "session_meta":
            return "session_start"
        if r == "tool":
            # A ``role: tool`` entry is an OpenAI-shaped tool result message;
            # not a tool *call*. We don't need to emit anything for it — tool
            # calls are already captured from the assistant's tool_calls array.
            return "tool_result"
    return None


def _ts(ev: dict) -> pd.Timestamp | None:
    for key in ("ts", "timestamp", "time", "started_at"):
        v = ev.get(key)
        if v:
            try:
                return pd.Timestamp(v, tz="UTC")
            except Exception:
                continue
    return None


def _text(ev: dict) -> str | None:
    for key in ("text", "content", "message", "preview"):
        v = ev.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _trunc(text, n: int = 600) -> str | None:
    if text is None:
        return None
    s = str(text)
    return s if len(s) <= n else s[:n] + "…"


def _str(obj, n: int = 600) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _trunc(obj, n)
    try:
        return _trunc(json.dumps(obj, ensure_ascii=False, default=str), n)
    except Exception:
        return _trunc(repr(obj), n)


# --------------------------------------------------------------------------- #
# File walking + session bucketing
# --------------------------------------------------------------------------- #

def iter_session_files(root: Path = CUSTOM_DIR) -> Iterator[tuple[str, Path]]:
    """Yield (project_dir_name, path) for every per-session file under root.

    Supports two shapes:
      1. ``*.jsonl`` — one JSON object per line (streaming format).
      2. ``session_*.json`` — a single dict with ``session_id``, ``model``,
         ``tools``, and ``messages``: list[...]. This is the shape Hermes-style
         agents emit when they dump a completed session in one shot. Takes
         precedence over a ``.jsonl`` with the same session id (the single-dump
         is authoritative — it carries ``model`` and the full tools schema).

    Skips ``request_dump_*.json`` (error-only per-request logs) and
    ``sessions.json`` (the top-level session registry index).
    """
    if not root.exists():
        return
    import re
    _id_re = re.compile(r"([0-9]{8}_[0-9]{6}_[0-9a-f]+)")

    # First pass — collect session_*.json files and the session-ids they cover.
    json_files: list[Path] = []
    covered_ids: set[str] = set()
    for p in sorted(root.rglob("session_*.json")):
        name = p.name
        if name.startswith("request_dump_") or name == "sessions.json":
            continue
        json_files.append(p)
        m = _id_re.search(name)
        if m:
            covered_ids.add(m.group(1))

    # JSONL files that aren't shadowed by a .json dump get emitted.
    for p in sorted(root.rglob("*.jsonl")):
        m = _id_re.search(p.name)
        if m and m.group(1) in covered_ids:
            continue  # prefer the .json dump (authoritative model + tools)
        project_dir = "custom" if p.parent == root else p.parent.name
        yield project_dir, p

    for p in json_files:
        project_dir = "custom" if p.parent == root else p.parent.name
        yield project_dir, p


def _iter_events(path: Path) -> Iterator[dict]:
    """Normalise a session file into a stream of events.

    JSONL files (one object per line) pass through unchanged. Single-dict
    JSON dumps (``{session_id, model, tools, messages: [...]}``) are converted
    into a synthesised ``session_meta`` event followed by the message array
    so the rest of the parser can treat them uniformly.
    """
    # Single-dict JSON dumps ("session_YYYYMMDD_*.json" shape from Hermes-style
    # agents): convert into a synthesised event stream.
    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                d = json.load(fh)
        except Exception:
            return
        if not isinstance(d, dict) or "messages" not in d:
            return
        sid = d.get("session_id") or path.stem
        session_start = d.get("session_start") or d.get("created_at")
        # Synthesise a session_meta line.
        yield {
            "role": "session_meta",
            "session_id": sid,
            "model": d.get("model"),
            "platform": d.get("platform"),
            "tools": d.get("tools") or [],
            "timestamp": session_start,
        }
        # Walk the messages array. Hermes dumps don't put timestamps on every
        # individual message — synthesise monotonic ones from session_start
        # so the downstream ordering stays stable.
        import datetime
        base_ts = None
        if session_start:
            try:
                base_ts = datetime.datetime.fromisoformat(session_start.replace("Z", "+00:00"))
            except Exception:
                base_ts = None
        for j, m in enumerate(d.get("messages") or []):
            if not isinstance(m, dict):
                continue
            ts = m.get("timestamp")
            if not ts and base_ts is not None:
                ts = (base_ts + datetime.timedelta(milliseconds=j)).isoformat()
            yield {
                **m,
                "session_id": sid,
                "timestamp": ts or m.get("timestamp"),
            }
        return

    # Classic JSONL stream: one event per line.
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _split_by_session(jsonl_path: Path) -> dict[str, list[dict]]:
    """Bucket events from one .jsonl by session_id.

    Supports single-session files (one key) and multi-session batch exports
    (many keys). Events with no session_id all fall into the same fallback
    bucket keyed on the filename stem so the dashboard shows a human-readable
    session id (e.g. ``20260401_154556_395f70d8``) rather than an opaque UUID.
    """
    fallback = jsonl_path.stem or str(_uuid.uuid5(_uuid.NAMESPACE_OID, str(jsonl_path)))
    buckets: dict[str, list[dict]] = {}
    for ev in _iter_events(jsonl_path):
        sid = (ev.get("session_id") or ev.get("sessionId")
               or ev.get("conversation_id") or fallback)
        buckets.setdefault(str(sid), []).append(ev)
    return buckets


# --------------------------------------------------------------------------- #
# Per-session parser
# --------------------------------------------------------------------------- #

def _parse_session(
    project_dir: str,
    jsonl_path: Path,
    session_id: str,
    raw_events: list[dict],
) -> dict[str, list[dict]]:
    """Parse a pre-bucketed list of events for one session_id.

    Returns a dict of row lists with the same shape as the Claude Code and
    Cursor extractors so the downstream pipeline needs zero changes.
    """
    messages: list[dict] = []
    tool_calls: list[dict] = []
    prompts: list[dict] = []
    errors: list[dict] = []

    cwd: str | None = None
    entrypoint: str | None = None
    primary_model: str | None = None
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    last_assistant_uuid: str | None = None

    # Pull session-level metadata from session_start events first, including
    # the tools schema length (OpenAI rebills the full tools JSON on every turn
    # so it's part of input_tokens on every assistant message).
    tools_schema_chars = 0
    for ev in raw_events:
        if _norm_event(ev) == "session_start":
            cwd = ev.get("cwd") or cwd
            entrypoint = ev.get("entrypoint") or entrypoint
            primary_model = ev.get("model") or primary_model
            tools = ev.get("tools")
            if tools is not None:
                try:
                    tools_schema_chars = len(json.dumps(tools, default=str))
                except Exception:
                    tools_schema_chars = 0
    # Fall back to the global default model when the export doesn't tag one.
    # Without a model the downstream family-aware gate returns an empty set of
    # cheaper siblings and the whole session is dropped from clustering.
    if not primary_model:
        primary_model = DEFAULT_MODEL

    # Running tally for the chars/N token estimator. The cumulative char count
    # approximates "prior context the next assistant turn bills against". We
    # count every billable input source: user prompts, previous assistant
    # text + reasoning, and tool-result content (which OpenAI rebills verbatim
    # as part of the conversation history). The tools schema length is added
    # per-turn on top (not cumulative — it rebills on every call either way).
    cumulative_input_chars = 0

    for i, ev in enumerate(raw_events):
        ev_name = _norm_event(ev)
        if ev_name in (None, "session_start", "session_end"):
            continue

        ts = _ts(ev)
        if ts is not None:
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts  = ts if last_ts  is None or ts > last_ts  else last_ts

        model   = ev.get("model") or primary_model
        ev_uuid = str(ev.get("uuid") or _uuid.uuid5(
            _uuid.NAMESPACE_OID, f"{jsonl_path}:{session_id}:{i}",
        ))

        if ev_name == "user_prompt":
            text = _text(ev)
            if text:
                cumulative_input_chars += len(text)
            prompt_id = ev.get("prompt_id") or ev.get("promptId") or str(
                _uuid.uuid5(_uuid.NAMESPACE_OID, f"{jsonl_path}:{session_id}:p:{i}"),
            )
            messages.append({
                "session_id": session_id, "project_dir": project_dir,
                "uuid": ev_uuid, "parent_uuid": None,
                "role": "user", "type": "user",
                "model": model, "timestamp": ts,
                "request_id": None, "stop_reason": None,
                "is_sidechain": False, "is_api_error": False,
                "error": None, "permission_mode": None,
                "prompt_id": prompt_id,
                "entrypoint": entrypoint or "custom", "cwd": cwd,
                "git_branch": None, "version": None, "user_type": None,
                "n_text_blocks": 1 if text else 0,
                "n_thinking_blocks": 0, "n_tool_use_blocks": 0, "n_tool_result_blocks": 0,
                "text_chars": len(text or ""), "thinking_chars": 0,
                "first_text_preview": _trunc(text, 280),
                "input_tokens": ev.get("input_tokens"), "output_tokens": None,
                "cache_creation_input_tokens": ev.get("cache_creation_tokens"),
                "cache_read_input_tokens": ev.get("cache_read_tokens"),
                "service_tier": None, "turn_duration_ms": None,
            })
            if text:
                prompts.append({
                    "session_id": session_id, "project_dir": project_dir,
                    "uuid": ev_uuid, "prompt_id": prompt_id, "timestamp": ts,
                    "char_len": len(text), "preview": _trunc(text, 600),
                    "permission_mode": None, "git_branch": None, "cwd": cwd,
                })

        elif ev_name == "assistant_message":
            text = _text(ev)
            # ``reasoning`` on OpenAI-shape assistant messages is the thinking
            # block — count its chars so the mechanical-turn gate recognises
            # these turns as carrying reasoning.
            reasoning = ev.get("reasoning")
            thinking_chars = int(ev.get("thinking_chars") or 0) or (
                len(reasoning) if isinstance(reasoning, str) else 0
            )
            last_assistant_uuid = ev_uuid
            # Token estimation fallback when the export doesn't record usage.
            # Per-turn input = tools schema (rebilled every call) + cumulative
            # conversation context so far. Per-turn output = this turn's content
            # + reasoning.
            in_tok = ev.get("input_tokens")
            out_tok = ev.get("output_tokens")
            if APPROX_TOKENS and in_tok is None and out_tok is None:
                in_tok = int((cumulative_input_chars + tools_schema_chars) / CHARS_PER_TOKEN)
                out_tok = int((len(text or "") + thinking_chars) / CHARS_PER_TOKEN)
            # Advance cumulative input-char tally: this turn's output becomes
            # context that later turns bill against.
            cumulative_input_chars += (len(text or "") + thinking_chars)
            messages.append({
                "session_id": session_id, "project_dir": project_dir,
                "uuid": ev_uuid, "parent_uuid": None,
                "role": "assistant", "type": "assistant",
                "model": model, "timestamp": ts,
                "request_id": ev.get("request_id"),
                # OpenAI-shape messages use ``finish_reason``; Anthropic uses
                # ``stop_reason``. Carry whichever is present.
                "stop_reason": ev.get("stop_reason") or ev.get("finish_reason"),
                "is_sidechain": bool(ev.get("is_sidechain")),
                "is_api_error": False, "error": None, "permission_mode": None,
                "prompt_id": None,
                "entrypoint": entrypoint or "custom", "cwd": cwd,
                "git_branch": None, "version": None, "user_type": None,
                # n_tool_use_blocks is backfilled below after parsing tool_calls
                "n_text_blocks": 1 if text else 0,
                "n_thinking_blocks": 1 if thinking_chars else 0,
                "n_tool_use_blocks": 0, "n_tool_result_blocks": 0,
                "text_chars": len(text or ""), "thinking_chars": thinking_chars,
                "first_text_preview": _trunc(text, 280),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cache_creation_input_tokens": ev.get("cache_creation_tokens") or 0,
                "cache_read_input_tokens": ev.get("cache_read_tokens") or 0,
                "service_tier": ev.get("service_tier"),
                "turn_duration_ms": ev.get("duration_ms") or ev.get("turn_duration_ms"),
            })

            # OpenAI-shape inline tool calls: the assistant message carries an
            # array like ``[{"id","function":{"name","arguments"}}]``. Expand
            # each one into a standalone tool_call row so the mechanical gate
            # and BGE clustering operate on them the same way they do on
            # Claude Code tool_use blocks.
            inline = ev.get("tool_calls")
            if isinstance(inline, list):
                for j, tc_inline in enumerate(inline):
                    if not isinstance(tc_inline, dict):
                        continue
                    fn = tc_inline.get("function") or {}
                    tool_name = (
                        fn.get("name")
                        or tc_inline.get("name")
                        or tc_inline.get("tool_name")
                        or "?"
                    )
                    args_raw = fn.get("arguments") if fn else tc_inline.get("arguments")
                    if isinstance(args_raw, str):
                        try:
                            tool_input = json.loads(args_raw)
                        except Exception:
                            tool_input = {"_raw_arguments": args_raw[:400]}
                    else:
                        tool_input = args_raw or tc_inline.get("input") or tc_inline.get("tool_input")
                    tu_id = (
                        tc_inline.get("id")
                        or tc_inline.get("tool_use_id")
                        or tc_inline.get("call_id")
                        or str(_uuid.uuid5(
                            _uuid.NAMESPACE_OID,
                            f"{jsonl_path}:{session_id}:{i}:inline-tc:{j}",
                        ))
                    )
                    tool_calls.append({
                        "session_id": session_id, "project_dir": project_dir,
                        "tool_use_id": tu_id, "tool_name": tool_name, "model": model,
                        "is_sidechain": bool(ev.get("is_sidechain")),
                        "parent_assistant_uuid": ev_uuid,
                        "request_id": ev.get("request_id"),
                        "started_at": ts, "ended_at": ts,
                        "wallclock_latency_ms": None,
                        "reported_duration_ms": None,
                        "input_preview": _str(tool_input, 600),
                        "input_size": len(json.dumps(tool_input, default=str))
                                       if tool_input is not None else 0,
                        "is_error": False,
                        "interrupted": None,
                        "is_image": None, "no_output_expected": None,
                        "result_preview": None,
                        "stdout_preview": None, "stderr_preview": None,
                        "subagent_total_tokens": None, "subagent_tool_use_count": None,
                        "subagent_id": None, "subagent_type": None,
                        "permission_mode": None, "cwd": cwd, "git_branch": None,
                    })

        elif ev_name == "tool_call":
            tool_name  = ev.get("tool_name") or ev.get("name") or "?"
            tool_input = ev.get("tool_input") or ev.get("input")
            tu_id = ev.get("tool_use_id") or ev.get("id") or str(
                _uuid.uuid5(_uuid.NAMESPACE_OID, f"{jsonl_path}:{session_id}:t:{i}"),
            )
            duration_ms = ev.get("duration_ms")
            ended = ts
            if ts is not None and duration_ms:
                try:
                    ended = ts + pd.Timedelta(milliseconds=int(duration_ms))
                except Exception:
                    pass
            tool_calls.append({
                "session_id": session_id, "project_dir": project_dir,
                "tool_use_id": tu_id, "tool_name": tool_name, "model": model,
                "is_sidechain": bool(ev.get("is_sidechain")),
                "parent_assistant_uuid": ev.get("parent_assistant_uuid") or last_assistant_uuid,
                "request_id": ev.get("request_id"),
                "started_at": ts, "ended_at": ended,
                "wallclock_latency_ms": float(duration_ms) if duration_ms else None,
                "reported_duration_ms": int(duration_ms) if duration_ms else None,
                "input_preview": _str(tool_input, 600),
                "input_size": len(json.dumps(tool_input, default=str)) if tool_input is not None else 0,
                "is_error": bool(ev.get("is_error")),
                "interrupted": bool(ev.get("interrupted")) if ev.get("interrupted") is not None else None,
                "is_image": None, "no_output_expected": None,
                "result_preview": _trunc(ev.get("result_preview") or ev.get("result"), 600),
                "stdout_preview": _trunc(ev.get("stdout"), 400),
                "stderr_preview": _trunc(ev.get("stderr"), 400),
                "subagent_total_tokens": None, "subagent_tool_use_count": None,
                "subagent_id": None, "subagent_type": None,
                "permission_mode": None, "cwd": cwd, "git_branch": None,
            })
            if ev.get("is_error"):
                errors.append({
                    "session_id": session_id, "project_dir": project_dir,
                    "uuid": ev_uuid, "timestamp": ts, "model": model,
                    "kind": "tool_error",
                    "error": _trunc(ev.get("result_preview") or ev.get("result") or ev.get("message"), 400),
                    "message_preview": tool_name,
                })

        elif ev_name == "error":
            errors.append({
                "session_id": session_id, "project_dir": project_dir,
                "uuid": ev_uuid, "timestamp": ts, "model": model,
                "kind": ev.get("kind") or "api_error",
                "error": _trunc(ev.get("message") or ev.get("error"), 400),
                "message_preview": None,
            })

        elif ev_name == "tool_result":
            # A role=tool entry carrying the result that the NEXT assistant
            # turn will bill against. Not a separate table row, but its content
            # must feed cumulative_input_chars so the approximation reflects
            # reality (tool results are often multi-KB of JSON).
            tr_text = ev.get("content")
            if isinstance(tr_text, list):
                try:
                    tr_text = json.dumps(tr_text, ensure_ascii=False, default=str)
                except Exception:
                    tr_text = str(tr_text)
            if tr_text:
                cumulative_input_chars += len(str(tr_text))
        # Everything else is silently ignored (forward-compatible).

    # Backfill n_tool_use_blocks on the assistant_message rows that own each
    # tool_call. The mechanical gate (is_mechanical_assistant_turn) checks this
    # field; without it, every custom turn would be wrongly excluded.
    uuid_to_msg = {m["uuid"]: m for m in messages if m["role"] == "assistant"}
    for tc in tool_calls:
        parent = tc.get("parent_assistant_uuid")
        if parent and parent in uuid_to_msg:
            uuid_to_msg[parent]["n_tool_use_blocks"] += 1

    session_row = {
        "session_id": session_id, "project_dir": project_dir,
        "decoded_cwd": cwd or project_dir,
        "git_branch": None, "version": None,
        "entrypoint": entrypoint or "custom",
        "ai_title": None, "custom_title": None,
        "first_event_at": first_ts, "last_event_at": last_ts,
        "duration_seconds": (last_ts - first_ts).total_seconds()
                             if first_ts and last_ts else None,
        "n_messages": len(messages),
        "n_assistant": sum(1 for m in messages if m["role"] == "assistant"),
        "n_user":      sum(1 for m in messages if m["role"] == "user"),
        "n_sidechain_messages": 0,
        "n_tool_calls": len(tool_calls),
        "n_tool_errors": sum(1 for t in tool_calls if t["is_error"]),
        "n_api_errors":  sum(1 for e in errors if e["kind"] == "api_error"),
        "n_prompts": len(prompts),
        "n_progress_events": 0, "n_queue_events": 0, "n_system_events": 0,
        "n_attachment_events": 0, "n_snapshot_events": 0,
        "n_compactions": 0, "n_local_commands": 0,
        "slash_commands": None, "permission_modes": None,
        "file_path": str(jsonl_path), "file_size_bytes": jsonl_path.stat().st_size,
    }

    return {
        "session":    [session_row],
        "messages":   messages,
        "tool_calls": tool_calls,
        "prompts":    prompts,
        "errors":     errors,
    }


# --------------------------------------------------------------------------- #
# Top-level loader (called by cli.py via sources/__init__.py)
# --------------------------------------------------------------------------- #

def load_all(custom_dir: Path = CUSTOM_DIR) -> "cc.Tables":
    """Parse every *.jsonl under custom_dir into the canonical Tables shape.

    Each file is split by session_id first, so a single file can carry
    multiple sessions (OTel batch exports, replay logs, etc.).
    """
    sessions:   list[dict] = []
    messages:   list[dict] = []
    tool_calls: list[dict] = []
    prompts:    list[dict] = []
    errors:     list[dict] = []

    for project_dir, jsonl in iter_session_files(custom_dir):
        for sid, evs in _split_by_session(jsonl).items():
            parsed = _parse_session(project_dir, jsonl, sid, evs)
            sessions.extend(parsed["session"])
            messages.extend(parsed["messages"])
            tool_calls.extend(parsed["tool_calls"])
            prompts.extend(parsed["prompts"])
            errors.extend(parsed["errors"])

    sessions_df    = pd.DataFrame(sessions)
    messages_df    = pd.DataFrame(messages)
    tool_calls_df  = pd.DataFrame(tool_calls)
    prompts_df     = pd.DataFrame(prompts)
    errors_df      = pd.DataFrame(errors)

    for df, col in [
        (sessions_df,   "first_event_at"),
        (sessions_df,   "last_event_at"),
        (messages_df,   "timestamp"),
        (tool_calls_df, "started_at"),
        (tool_calls_df, "ended_at"),
        (prompts_df,    "timestamp"),
        (errors_df,     "timestamp"),
    ]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if not messages_df.empty:
        messages_df   = messages_df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    if not tool_calls_df.empty:
        tool_calls_df = tool_calls_df.sort_values(["session_id", "started_at"]).reset_index(drop=True)
    if not sessions_df.empty:
        sessions_df   = sessions_df.sort_values("first_event_at", na_position="last").reset_index(drop=True)

    return cc.Tables(sessions_df, messages_df, tool_calls_df, prompts_df, errors_df)


# --------------------------------------------------------------------------- #
# Smoke test (python -m tracercc.sources.custom)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    t = load_all()
    print(t)
    if not t.sessions.empty:
        print("\nSessions:")
        cols = [c for c in ["session_id", "project_dir", "n_messages", "n_tool_calls",
                             "first_event_at", "duration_seconds"] if c in t.sessions.columns]
        print(t.sessions[cols].to_string(index=False))
    if not t.messages.empty:
        print("\nModels seen:")
        print(t.messages["model"].fillna("(none)").value_counts().to_string())
        print("\nn_tool_use_blocks on assistant turns:")
        asst = t.messages[t.messages["role"] == "assistant"]
        print(asst["n_tool_use_blocks"].value_counts().sort_index().to_string())
    if not t.tool_calls.empty:
        print("\nTop tools:")
        print(t.tool_calls["tool_name"].value_counts().head(10).to_string())
