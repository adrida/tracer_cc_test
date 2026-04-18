"""
tracerCC — Cursor agent transcripts extractor.

Walks every Cursor conversation transcript at
``~/.cursor/projects/<workspace>/agent-transcripts/<conversation_id>/<conversation_id>.jsonl``
and joins it against the per-request model attribution in
``~/.cursor/ai-tracking/ai-code-tracking.db`` (SQLite).

Returns the SAME ``Tables`` shape as the Claude Code extractor so the
downstream pipeline (episodes → BGE → cluster → counterfactual) can
consume it unchanged. The only non-trivial difference: Cursor JSONL
does not carry per-turn timestamps, model attribution, token counts
or tool_result blocks, so those fields are filled best-effort:

    - model: per-conversation primary model (mode of SQLite rows for
      that conversationId), or assigned per-turn via a chronological
      walk over (requestId, model, started_at) tuples when there are
      multiple models in one conversation.
    - timestamps: the conversation's first/last SQLite timestamp,
      with assistant turns linearly interpolated by ordinal.
    - tokens: None. Cursor billing is request-based, not token-based,
      so the downstream cost layer needs Cursor-specific pricing.
    - tool_calls: synthesised from ``tool_use`` blocks in the JSONL
      (which carry ``name`` + ``input``, exactly what
      ``mechanical_turn_text`` needs). No tool_result data is recorded.

Honest limits, surfaced in ``Tables.sessions``:
    - n_tool_errors / n_api_errors are always 0 (Cursor doesn't log them
      in transcripts).
    - turn_duration_ms is always None.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

# Reuse the Tables dataclass and helpers from the Claude Code extractor
# so both sources produce structurally identical output.
from .. import extractor as cc

CURSOR_HOME = Path(os.environ.get("CURSOR_HOME", Path.home() / ".cursor"))
PROJECTS_DIR = CURSOR_HOME / "projects"
TRACKING_DB = CURSOR_HOME / "ai-tracking" / "ai-code-tracking.db"


# --------------------------------------------------------------------------- #
# SQLite — model attribution per conversation
# --------------------------------------------------------------------------- #

@dataclass
class _ConvAttribution:
    """Per-conversation model summary pulled from the ai-tracking SQLite."""
    conversation_id: str
    primary_model: str | None         # mode-of-rows model, the "main" one
    model_share: dict[str, float]     # model -> fraction of rows
    n_requests: int                   # distinct requestIds seen
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None
    # Ordered (start_ts, model) pairs — one tuple per requestId, used for
    # per-turn interpolation when the conversation spans multiple models.
    request_order: list[tuple[pd.Timestamp | None, str | None]]


def _load_conv_attributions(db_path: Path = TRACKING_DB) -> dict[str, _ConvAttribution]:
    """Build a {conversationId: _ConvAttribution} map from the ai-tracking db.

    Returns {} if the database is missing — Cursor users who don't have
    AI-code tracking enabled (or who installed Cursor before the feature
    landed) will simply have no model attribution.
    """
    if not db_path.exists():
        return {}

    out: dict[str, _ConvAttribution] = {}
    con = sqlite3.connect(str(db_path))
    try:
        # Aggregate per conversation
        rows = list(con.execute("""
            SELECT conversationId,
                   model,
                   COUNT(*)  AS n,
                   MIN(timestamp) AS ts_min,
                   MAX(timestamp) AS ts_max
            FROM ai_code_hashes
            WHERE conversationId IS NOT NULL
            GROUP BY conversationId, model
        """))
        per_conv: dict[str, list[tuple[str | None, int, str | None, str | None]]] = {}
        for cid, model, n, ts_min, ts_max in rows:
            per_conv.setdefault(cid, []).append((model, n, ts_min, ts_max))

        # Per-request order (used for multi-model conversations)
        per_conv_requests: dict[str, list[tuple[pd.Timestamp | None, str | None]]] = {}
        req_rows = list(con.execute("""
            SELECT conversationId,
                   requestId,
                   model,
                   MIN(timestamp) AS ts_min
            FROM ai_code_hashes
            WHERE conversationId IS NOT NULL AND requestId IS NOT NULL
            GROUP BY conversationId, requestId
            ORDER BY conversationId, ts_min
        """))
        for cid, _rid, model, ts_min in req_rows:
            per_conv_requests.setdefault(cid, []).append(
                (_safe_ts(ts_min), model)
            )

        for cid, parts in per_conv.items():
            total = sum(p[1] for p in parts) or 1
            primary = max(parts, key=lambda p: p[1])[0]
            share = {(p[0] or "unknown"): p[1] / total for p in parts}
            ts_mins = [_safe_ts(p[2]) for p in parts if p[2] is not None]
            ts_maxs = [_safe_ts(p[3]) for p in parts if p[3] is not None]
            ts_mins = [t for t in ts_mins if t is not None]
            ts_maxs = [t for t in ts_maxs if t is not None]
            n_req = len(per_conv_requests.get(cid, [])) or len(parts)
            out[cid] = _ConvAttribution(
                conversation_id=cid,
                primary_model=primary,
                model_share=share,
                n_requests=n_req,
                first_ts=min(ts_mins) if ts_mins else None,
                last_ts=max(ts_maxs) if ts_maxs else None,
                request_order=per_conv_requests.get(cid, []),
            )
    finally:
        con.close()
    return out


def _safe_ts(value) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        # SQLite stores ms-since-epoch in this DB
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="ms", tz="UTC")
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        return None


def _model_for_turn(att: _ConvAttribution | None, turn_ordinal: int, n_assistant_turns: int) -> str | None:
    """Pick the model for assistant turn #turn_ordinal in a conversation.

    If there's only one request (or no SQLite data), return primary.
    Otherwise distribute turns across requests proportionally so multi-
    model conversations get reasonable per-turn attribution.
    """
    if att is None:
        return None
    if not att.request_order or len(att.request_order) == 1:
        return att.primary_model
    n_req = len(att.request_order)
    if n_assistant_turns <= 0:
        return att.primary_model
    bucket = min(n_req - 1, int(turn_ordinal * n_req / n_assistant_turns))
    return att.request_order[bucket][1] or att.primary_model


def _normalize_model_id(raw: str | None) -> str | None:
    """Map Cursor's model-id strings to the canonical IDs in ``MODEL_PRICING_USD_PER_MTOK``.

    Cursor exposes models in several inconsistent formats:
      - ``claude-4.6-sonnet-medium-thinking``  (dot-separated, with mode/thinking suffix)
      - ``claude-opus-4-7-thinking-high``      (dash-separated, already close to canonical)
      - ``gpt-5.4-medium``                     (dot-separated, with reasoning suffix)
      - ``composer-2``                         (canonical already)
      - ``claude-opus-4-6-fast``               (Cursor's Opus Fast research preview)

    Returns the canonical ID used in ``extractor.MODEL_PRICING_USD_PER_MTOK``
    so cost / counterfactual computations Just Work.
    """
    if not isinstance(raw, str):
        return raw
    s = raw.lower().strip()
    # Preserve a "fast" marker before stripping mode suffixes — we want to
    # keep "fast" because Opus Fast is priced at 6x the normal rate.
    is_fast = "-fast" in s
    if is_fast:
        s = s.replace("-fast", "")
    # Strip trailing thinking/reasoning/mode suffixes (Cursor adds these
    # to indicate reasoning effort; pricing is the same regardless).
    for suffix in (
        "-thinking-high", "-thinking-medium", "-thinking-low",
        "-medium-thinking", "-high-thinking", "-low-thinking",
        "-high-reasoning", "-medium-reasoning", "-low-reasoning",
        "-high", "-medium", "-low", "-thinking", "-reasoning",
    ):
        while s.endswith(suffix):
            s = s[: -len(suffix)]
    # Dots → dashes (gpt-5.4 → gpt-5-4, claude-4.6-sonnet → claude-4-6-sonnet)
    s = s.replace(".", "-")
    # Reorder claude-<num>-<num>-<family> into claude-<family>-<num>-<num>
    if s.startswith("claude-"):
        rest = s[len("claude-"):]
        parts = rest.split("-")
        if (
            len(parts) >= 3
            and parts[0].isdigit() and parts[1].isdigit()
            and parts[2] in {"sonnet", "opus", "haiku"}
        ):
            tail = "-".join(parts[3:])
            base = f"claude-{parts[2]}-{parts[0]}-{parts[1]}"
            s = f"{base}-{tail}".rstrip("-") if tail else base
    # Re-attach the fast marker so the pricing table resolves to the
    # 6x-priced "claude-opus-4-6-fast" variant.
    if is_fast and "fast" not in s:
        s = f"{s}-fast"
    return s


# --------------------------------------------------------------------------- #
# JSONL parsing
# --------------------------------------------------------------------------- #

def iter_transcript_files(projects_dir: Path = PROJECTS_DIR) -> Iterator[tuple[str, str, Path]]:
    """Yield (workspace_dir, conversation_id, jsonl_path) for every Cursor transcript."""
    if not projects_dir.exists():
        return
    for ws in sorted(projects_dir.iterdir()):
        if not ws.is_dir():
            continue
        at = ws / "agent-transcripts"
        if not at.is_dir():
            continue
        for conv_dir in sorted(at.iterdir()):
            if not conv_dir.is_dir():
                continue
            for jsonl in sorted(conv_dir.glob("*.jsonl")):
                yield ws.name, conv_dir.name, jsonl


def _iter_events(jsonl_path: Path) -> Iterator[dict]:
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _content_blocks(message) -> list[dict]:
    if not isinstance(message, dict):
        return []
    c = message.get("content")
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    if isinstance(c, str) and c:
        return [{"type": "text", "text": c}]
    return []


def _truncate(text, n=500):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[:n] + "…"


def _stringify(obj, n=500):
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate(obj, n)
    try:
        return _truncate(json.dumps(obj, ensure_ascii=False, default=str), n)
    except Exception:
        return _truncate(repr(obj), n)


# --------------------------------------------------------------------------- #
# Per-conversation parsing
# --------------------------------------------------------------------------- #

def _parse_conversation(
    workspace: str,
    conv_id: str,
    jsonl_path: Path,
    att: _ConvAttribution | None,
) -> dict[str, list[dict]]:
    """Parse one Cursor conversation JSONL into the same row lists as Claude Code."""
    messages: list[dict] = []
    tool_calls: list[dict] = []
    prompts: list[dict] = []
    errors: list[dict] = []  # Cursor doesn't surface these in transcripts

    events = list(_iter_events(jsonl_path))
    n_assistant = sum(1 for e in events if e.get("role") == "assistant")

    # Synthesise timestamps: spread linearly across the conversation window
    # if we have one, otherwise leave NaT.
    t0, t1 = (att.first_ts if att else None), (att.last_ts if att else None)

    def _interp_ts(i: int, n: int) -> pd.Timestamp | None:
        if t0 is None:
            return None
        if t1 is None or n <= 1:
            return t0
        frac = i / max(n - 1, 1)
        return t0 + (t1 - t0) * frac

    pending_tools: dict[str, dict] = {}
    asst_ord = 0  # ordinal among assistant turns (used for model interpolation)

    # First pass: cumulative-chars walker so we can synthesise input_tokens
    # for each assistant turn (≈ everything that came before, /4). This lets
    # the existing token-based pipeline price Cursor messages without any
    # downstream change — at the cost of being a rough approximation. The
    # dashboard surfaces this caveat.
    cumulative_chars_before = [0] * len(events)
    running = 0
    for i, ev in enumerate(events):
        cumulative_chars_before[i] = running
        for b in _content_blocks((ev.get("message") or {})):
            t = b.get("type")
            if t == "text":
                running += len(b.get("text") or "")
            elif t == "tool_use":
                running += len(json.dumps(b.get("input"), default=str)) if b.get("input") is not None else 0

    for i, ev in enumerate(events):
        role = ev.get("role")
        msg = ev.get("message") or {}
        blocks = _content_blocks(msg)

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

        # Synthetic uuid that is stable across runs (deterministic from conv+ordinal)
        msg_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_OID, f"cursor:{conv_id}:{i}"))
        ts = _interp_ts(i, len(events))

        # Per-turn model attribution: assistant turns get model from SQLite,
        # user turns inherit (for cost gating they don't matter — tokens=None).
        if role == "assistant":
            raw_model = _model_for_turn(att, asst_ord, max(1, n_assistant))
            asst_ord += 1
        else:
            raw_model = att.primary_model if att else None
        model = _normalize_model_id(raw_model)

        messages.append({
            "session_id": conv_id,
            "project_dir": workspace,
            "uuid": msg_uuid,
            "parent_uuid": None,
            "role": role,
            "type": role,                         # "user" / "assistant"
            "model": model,
            "timestamp": ts,
            "request_id": None,
            "stop_reason": None,
            "is_sidechain": False,
            "is_api_error": False,
            "error": None,
            "permission_mode": None,
            "prompt_id": str(_uuid.uuid5(_uuid.NAMESPACE_OID, f"cursor:{conv_id}:prompt:{i}")) if role == "user" else None,
            "entrypoint": "cursor",
            "cwd": workspace,
            "git_branch": None,
            "version": None,
            "user_type": None,
            "n_text_blocks": n_text,
            "n_thinking_blocks": n_thinking,
            "n_tool_use_blocks": n_tool_use,
            "n_tool_result_blocks": n_tool_result,
            "text_chars": text_chars,
            "thinking_chars": thinking_chars,
            "first_text_preview": _truncate(first_text, 280),
            # Token estimates synthesised from text lengths (≈ chars/4).
            # Cursor's transcripts don't carry usage stats, so the dashboard
            # treats Cursor cost figures as approximations and labels them.
            "input_tokens": (cumulative_chars_before[i] // 4) if role == "assistant" else None,
            "output_tokens": ((text_chars + thinking_chars) // 4) if role == "assistant" else None,
            "cache_creation_input_tokens": 0 if role == "assistant" else None,
            "cache_read_input_tokens": 0 if role == "assistant" else None,
            "service_tier": None,
            "turn_duration_ms": None,
        })

        # Prompts: Cursor user turns are pure text → eligible.
        if role == "user" and first_text:
            joined = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if joined:
                prompts.append({
                    "session_id": conv_id,
                    "project_dir": workspace,
                    "uuid": msg_uuid,
                    "prompt_id": messages[-1]["prompt_id"],
                    "timestamp": ts,
                    "char_len": len(joined),
                    "preview": _truncate(joined, 600),
                    "permission_mode": None,
                    "git_branch": None,
                    "cwd": workspace,
                })

        # Tool calls: every assistant tool_use → a tool_calls row.
        # Cursor doesn't store tool_results, so the row is "started but never
        # joined" — input fields are present, result/duration fields are None.
        if role == "assistant":
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                tool_input = b.get("input")
                tu_id = str(_uuid.uuid5(_uuid.NAMESPACE_OID, f"cursor:{conv_id}:{i}:{b.get('name','?')}"))
                tool_calls.append({
                    "session_id": conv_id,
                    "project_dir": workspace,
                    "tool_use_id": tu_id,
                    "tool_name": b.get("name") or "?",
                    "model": model,
                    "is_sidechain": False,
                    "parent_assistant_uuid": msg_uuid,
                    "request_id": None,
                    "started_at": ts,
                    "ended_at": None,
                    "wallclock_latency_ms": None,
                    "reported_duration_ms": None,
                    "input_preview": _stringify(tool_input, 600),
                    "input_size": len(json.dumps(tool_input, default=str)) if tool_input is not None else 0,
                    "is_error": False,
                    "interrupted": None,
                    "is_image": None,
                    "no_output_expected": None,
                    "result_preview": None,
                    "stdout_preview": None,
                    "stderr_preview": None,
                    "subagent_total_tokens": None,
                    "subagent_tool_use_count": None,
                    "subagent_id": None,
                    "subagent_type": None,
                    "permission_mode": None,
                    "cwd": workspace,
                    "git_branch": None,
                })

    session_row = {
        "session_id": conv_id,
        "project_dir": workspace,
        "decoded_cwd": _decode_workspace(workspace),
        "git_branch": None,
        "version": None,
        "entrypoint": "cursor",
        "ai_title": None,
        "custom_title": None,
        "first_event_at": t0,
        "last_event_at": t1,
        "duration_seconds": (t1 - t0).total_seconds() if (t0 is not None and t1 is not None) else None,
        "n_messages": len(messages),
        "n_assistant": sum(1 for m in messages if m["role"] == "assistant"),
        "n_user": sum(1 for m in messages if m["role"] == "user"),
        "n_sidechain_messages": 0,
        "n_tool_calls": len(tool_calls),
        "n_tool_errors": 0,
        "n_api_errors": 0,
        "n_prompts": len(prompts),
        "n_progress_events": 0,
        "n_queue_events": 0,
        "n_system_events": 0,
        "n_attachment_events": 0,
        "n_snapshot_events": 0,
        "n_compactions": 0,
        "n_local_commands": 0,
        "slash_commands": None,
        "permission_modes": None,
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


def _decode_workspace(name: str) -> str:
    """Cursor encodes workspace paths the same way Claude Code does (slash → dash)."""
    if not name.startswith("Users-") and not name.startswith("-"):
        return name
    if name.startswith("Users-"):
        return "/" + name.replace("-", "/")
    return "/" + name[1:].replace("-", "/")


# --------------------------------------------------------------------------- #
# Top-level loader
# --------------------------------------------------------------------------- #

def load_all(projects_dir: Path = PROJECTS_DIR, db_path: Path = TRACKING_DB) -> "cc.Tables":
    """Parse every Cursor transcript and return the same Tables shape as Claude Code."""
    attributions = _load_conv_attributions(db_path)

    sessions: list[dict] = []
    messages: list[dict] = []
    tool_calls: list[dict] = []
    prompts: list[dict] = []
    errors: list[dict] = []

    for workspace, conv_id, jsonl in iter_transcript_files(projects_dir):
        att = attributions.get(conv_id)
        parsed = _parse_conversation(workspace, conv_id, jsonl, att)
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
    ]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if not messages_df.empty:
        messages_df = messages_df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    if not tool_calls_df.empty:
        tool_calls_df = tool_calls_df.sort_values(["session_id", "started_at"]).reset_index(drop=True)
    if not sessions_df.empty:
        sessions_df = sessions_df.sort_values("first_event_at", na_position="last").reset_index(drop=True)

    return cc.Tables(sessions_df, messages_df, tool_calls_df, prompts_df, errors_df)


# --------------------------------------------------------------------------- #
# CLI / smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    t = load_all()
    print(t)
    if not t.sessions.empty:
        print("\nConversations:")
        cols = ["session_id", "project_dir", "n_messages", "n_tool_calls",
                "first_event_at", "duration_seconds"]
        print(t.sessions[cols].to_string(index=False))
    if not t.messages.empty:
        print("\nModels seen:")
        print(t.messages["model"].fillna("(none)").value_counts().to_string())
    if not t.tool_calls.empty:
        print("\nTop tools:")
        print(t.tool_calls["tool_name"].value_counts().head(15).to_string())
