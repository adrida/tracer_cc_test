"""Wire format between the public client and the analysis backend.

Narrow by design: only the fields the analysis layer needs. Anything that would
carry raw user-prompt or assistant-response text is absent — the client strips
that before sending, and this schema encodes that contract at the type level.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Request — normalised, redacted Tables
# --------------------------------------------------------------------------- #

class SessionRow(BaseModel):
    session_id: str
    project_dir: Optional[str] = None
    decoded_cwd: Optional[str] = None
    first_event_at: Optional[str] = None
    last_event_at: Optional[str] = None
    n_compactions: int = 0


class MessageRow(BaseModel):
    """A single user/assistant message — token counts and metadata only.

    Notably absent: ``content_text``. The client strips raw prompt + response
    text before sending. The only free-form text accepted is
    ``tool_calls.input_preview`` (used for clustering) and even there the
    client truncates aggressively.
    """
    uuid: str
    session_id: str
    type: str  # "user" | "assistant"
    model: Optional[str] = None
    timestamp: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    n_text_blocks: Optional[int] = 0
    n_thinking_blocks: Optional[int] = 0
    n_tool_use_blocks: Optional[int] = 0
    project_dir: Optional[str] = None


class ToolCallRow(BaseModel):
    """A tool_use block. ``input_preview`` is the only semantic text field, and
    it's already a structured tool argument (not a user prompt). The client
    truncates it to 500 chars before sending."""
    tool_use_id: Optional[str] = None
    session_id: str
    parent_assistant_uuid: str
    tool_name: Optional[str] = None
    input_preview: Optional[str] = None
    is_error: bool = False
    started_at: Optional[str] = None


class PromptRow(BaseModel):
    """User prompts: timestamps and lengths only — no text content."""
    uuid: str
    session_id: str
    timestamp: Optional[str] = None
    char_count: int = 0


class ErrorRow(BaseModel):
    session_id: str
    timestamp: Optional[str] = None
    error_type: Optional[str] = None


class TablesPayload(BaseModel):
    sessions: list[SessionRow] = Field(default_factory=list)
    messages: list[MessageRow] = Field(default_factory=list)
    tool_calls: list[ToolCallRow] = Field(default_factory=list)
    prompts: list[PromptRow] = Field(default_factory=list)
    errors: list[ErrorRow] = Field(default_factory=list)


class AnalyzeOptions(BaseModel):
    targets: list[str] = Field(
        default_factory=lambda: [
            "claude-sonnet-4-6", "claude-haiku-4-5", "composer-2",
            "gpt-5-4-mini", "gpt-5-4-nano", "gemini-2-5-flash",
        ]
    )
    min_cluster_size: Optional[int] = None  # auto if None
    # Gate-loosening knob: allow assistant turns with up to N chars of
    # reasoning/thinking text to still be classed as "mechanical" iff the
    # tool calls otherwise pass the filter. 0 (default) is the strict
    # tracer-paper discipline — no reasoning allowed. Higher values unlock
    # agents like GPT-5.2 that auto-emit short chain-of-thought on every
    # call regardless of difficulty. The clustering layer still has to agree
    # the turn is part of a dense repeated-pattern group before it re-routes.
    reasoning_threshold_chars: int = 0


class AnalyzeRequest(BaseModel):
    schema_version: str = "1.0"
    source: str
    client_version: str = "unknown"
    redacted_prompts: bool = True
    data: TablesPayload
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


# --------------------------------------------------------------------------- #
# Response — WrappedReport
# --------------------------------------------------------------------------- #

class ClusterCard(BaseModel):
    cluster_id: int
    n_turns: int
    dominant_tool: str
    actual_usd: float
    save_haiku_usd: float
    save_sonnet_usd: float
    save_composer2_usd: float = 0.0
    save_cheapest_sibling_usd: float = 0.0  # new: family-aware headline
    medoid_text: str
    examples: list[str]
    label: str


class SessionCard(BaseModel):
    session_short: str
    cwd: str
    started_at: str
    n_prompts: int
    spend_usd: float
    dominant_model: str
    top_tool: str


class FunStat(BaseModel):
    title: str
    value: str
    detail: str


class WrappedReport(BaseModel):
    # corpus
    n_sessions: int
    n_messages: int
    n_prompts: int
    earliest: str
    latest: str
    span_days: int
    source: str = "claude_code"

    # money — legacy Anthropic-specific fields kept for render-layer back-compat
    total_spend_usd: float
    opus_spend_usd: float
    sonnet_spend_usd: float
    haiku_spend_usd: float
    cache_read_share: float

    # money — generalised family breakdown (new)
    spend_by_family: dict[str, float] = Field(default_factory=dict)
    premium_spend_usd: float = 0.0  # spend on models that have at least one cheaper sibling

    # gate
    n_mechanical_turns: int
    n_clusters: int
    n_noise: int
    spend_in_clusters_usd: float
    spend_excluded_usd: float

    # gated savings (downgrade ONLY mechanical-clustered turns)
    saving_haiku_usd: float
    saving_sonnet_usd: float
    saving_composer2_usd: float = 0.0
    saving_cheapest_sibling_usd: float = 0.0  # new: family-aware headline savings
    saving_haiku_pct_of_opus: float
    saving_cheapest_sibling_pct_of_premium: float = 0.0

    # ceilings (if 100% of premium had been routed elsewhere — NOT a tracer claim)
    ceiling_haiku_usd: float
    ceiling_haiku_pct_of_opus: float
    ceiling_sonnet_usd: float = 0.0
    ceiling_composer2_usd: float = 0.0
    ceiling_cheapest_sibling_usd: float = 0.0

    # cards
    top_clusters: list[ClusterCard]
    top_sessions: list[SessionCard]
    fun_stats: list[FunStat]

    # provenance
    generated_at: str
    backend_version: str
    cluster_backend: str = "lite"
    pricing_source: dict[str, Any] = Field(default_factory=dict)
    gate: str = "structural"  # "structural" (open) | "behavioural" (hosted)
