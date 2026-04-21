"""Pricing tables and counterfactual helpers.

Single source of truth for per-token rates, covering the providers most commonly
seen across Claude Code / Cursor / custom agentic systems.

Sources (verified 2026-04-17):
    - Anthropic:  https://www.anthropic.com/pricing
                  https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI:     https://openai.com/api/pricing
    - Google:     https://ai.google.dev/pricing
    - xAI:        https://x.ai/api (Grok)
    - Moonshot:   https://platform.moonshot.ai/docs/pricing (Kimi)
    - Mistral:    https://mistral.ai/technology (open-weight & API)
    - DeepSeek:   https://api-docs.deepseek.com/quick_start/pricing
    - Cursor:     https://cursor.com/pricing

Cursor's API pool bills at each provider's API rate, so Anthropic models in
Cursor map 1:1 onto Anthropic's published per-token pricing. OpenAI models in
Cursor likewise match OpenAI's list API prices.

## Naming convention

Model IDs from real API responses use a mix of dots (``gpt-5.2-2025-12-11``),
dashes (``gpt-5-2-codex``), and dated snapshot suffixes. The resolver below
canonicalises both sides (lower + dot→dash) before lookup, so you can write
keys either way and they'll still match. Snapshot suffixes resolve by longest
prefix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _canon(s: str) -> str:
    """Canonical key: lowercase, dots → dashes. Handles ``gpt-5.2`` vs ``gpt-5-2``."""
    return (s or "").strip().lower().replace(".", "-")


MODEL_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # ─────────── Anthropic ────────────────────────────────────────────────
    "claude-opus-4-7":   {"input":  5.0, "output": 25.0, "cache_write":  6.25, "cache_read": 0.50},
    "claude-sonnet-4-6": {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5":  {"input":  1.0, "output":  5.0, "cache_write":  1.25, "cache_read": 0.10},
    "claude-opus-4-6":   {"input":  5.0, "output": 25.0, "cache_write":  6.25, "cache_read": 0.50},
    "claude-sonnet-4-5": {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "claude-opus-4-5":   {"input":  5.0, "output": 25.0, "cache_write":  6.25, "cache_read": 0.50},
    "claude-opus-4-1":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4":   {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "claude-opus-4":     {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "claude-3-haiku":    {"input":  0.25,"output":  1.25,"cache_write":  0.30, "cache_read": 0.03},

    # ─────────── Cursor's "Fast mode" Opus (research preview, 6× normal) ──
    "claude-opus-4-6-fast":  {"input": 30.0, "output": 150.0, "cache_write": 37.5, "cache_read":  3.0},

    # ─────────── Cursor proprietary ────────────────────────────────────────
    "composer-2":        {"input":  0.5,  "output":  2.5,  "cache_write":  0.5,   "cache_read": 0.20},
    "composer-1-5":      {"input":  3.5,  "output": 17.5,  "cache_write":  3.5,   "cache_read": 0.35},
    "composer-1":        {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "auto":              {"input":  1.25, "output":  6.0,  "cache_write":  1.25,  "cache_read": 0.25},

    # ─────────── OpenAI ───────────────────────────────────────────────────
    # Note: canonicalised keys use dashes, but real model IDs (``gpt-5.2-…``)
    # resolve here too because the resolver lowercases + dot→dash before match.
    "gpt-5":              {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "gpt-5-fast":         {"input":  2.5,  "output": 20.0,  "cache_write":  2.5,   "cache_read": 0.25},
    "gpt-5-mini":         {"input":  0.25, "output":  2.0,  "cache_write":  0.25,  "cache_read": 0.025},
    "gpt-5-codex":        {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "gpt-5-1-codex":      {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "gpt-5-1-codex-max":  {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "gpt-5-1-codex-mini": {"input":  0.25, "output":  2.0,  "cache_write":  0.25,  "cache_read": 0.025},
    "gpt-5-2":            {"input":  1.75, "output": 14.0,  "cache_write":  1.75,  "cache_read": 0.175},
    "gpt-5-2-codex":      {"input":  1.75, "output": 14.0,  "cache_write":  1.75,  "cache_read": 0.175},
    "gpt-5-2-mini":       {"input":  0.35, "output":  2.8,  "cache_write":  0.35,  "cache_read": 0.035},
    "gpt-5-3-codex":      {"input":  1.75, "output": 14.0,  "cache_write":  1.75,  "cache_read": 0.175},
    "gpt-5-4":            {"input":  2.5,  "output": 15.0,  "cache_write":  2.5,   "cache_read": 0.25},
    "gpt-5-4-mini":       {"input":  0.75, "output":  4.5,  "cache_write":  0.75,  "cache_read": 0.075},
    "gpt-5-4-nano":       {"input":  0.2,  "output":  1.25, "cache_write":  0.2,   "cache_read": 0.02},
    "gpt-4-1":            {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,   "cache_read": 0.20},
    "gpt-4-1-mini":       {"input":  0.4,  "output":  1.6,  "cache_write":  0.4,   "cache_read": 0.04},
    "gpt-4-1-nano":       {"input":  0.1,  "output":  0.4,  "cache_write":  0.1,   "cache_read": 0.01},
    "gpt-4o":             {"input":  2.5,  "output": 10.0,  "cache_write":  2.5,   "cache_read": 1.25},
    "gpt-4o-mini":        {"input":  0.15, "output":  0.6,  "cache_write":  0.15,  "cache_read": 0.075},
    "o1":                 {"input": 15.0,  "output": 60.0,  "cache_write": 15.0,   "cache_read": 7.5},
    "o1-mini":            {"input":  1.1,  "output":  4.4,  "cache_write":  1.1,   "cache_read": 0.55},
    "o3":                 {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,   "cache_read": 0.5},
    "o3-mini":            {"input":  1.1,  "output":  4.4,  "cache_write":  1.1,   "cache_read": 0.55},
    "o4-mini":            {"input":  1.1,  "output":  4.4,  "cache_write":  1.1,   "cache_read": 0.275},

    # ─────────── Google ───────────────────────────────────────────────────
    "gemini-2-5-flash":   {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,   "cache_read": 0.03},
    "gemini-2-5-pro":     {"input":  1.25, "output": 10.0,  "cache_write":  1.25,  "cache_read": 0.125},
    "gemini-3-flash":     {"input":  0.5,  "output":  3.0,  "cache_write":  0.5,   "cache_read": 0.05},
    "gemini-3-pro":       {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,   "cache_read": 0.20},
    "gemini-3-1-pro":     {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,   "cache_read": 0.20},

    # ─────────── xAI (Grok) ───────────────────────────────────────────────
    "grok-4-20":          {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,   "cache_read": 0.20},
    "grok-4":             {"input":  3.0,  "output": 15.0,  "cache_write":  3.0,   "cache_read": 0.75},
    "grok-3":             {"input":  3.0,  "output": 15.0,  "cache_write":  3.0,   "cache_read": 0.75},
    "grok-3-mini":        {"input":  0.3,  "output":  0.5,  "cache_write":  0.3,   "cache_read": 0.075},

    # ─────────── Moonshot (Kimi) ──────────────────────────────────────────
    "kimi-k2-5":          {"input":  0.6,  "output":  3.0,  "cache_write":  0.6,   "cache_read": 0.10},
    "kimi-k2":            {"input":  0.6,  "output":  2.5,  "cache_write":  0.6,   "cache_read": 0.15},

    # ─────────── Mistral ──────────────────────────────────────────────────
    "mistral-large":      {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,   "cache_read": 0.20},
    "mistral-medium":     {"input":  0.4,  "output":  2.0,  "cache_write":  0.4,   "cache_read": 0.04},
    "mistral-small":      {"input":  0.1,  "output":  0.3,  "cache_write":  0.1,   "cache_read": 0.01},
    "codestral":          {"input":  0.2,  "output":  0.6,  "cache_write":  0.2,   "cache_read": 0.02},

    # ─────────── DeepSeek ─────────────────────────────────────────────────
    "deepseek-chat":      {"input":  0.27, "output":  1.1,  "cache_write":  0.27,  "cache_read": 0.07},
    "deepseek-reasoner":  {"input":  0.55, "output":  2.19, "cache_write":  0.55,  "cache_read": 0.14},
    "deepseek-v3":        {"input":  0.27, "output":  1.1,  "cache_write":  0.27,  "cache_read": 0.07},
    "deepseek-r1":        {"input":  0.55, "output":  2.19, "cache_write":  0.55,  "cache_read": 0.14},
}


# Family tree — used by the family-aware counterfactual picker. Each family is
# ordered cheapest → most expensive (or roughly by tier) so "find a cheaper
# sibling" is a straight walk.
MODEL_FAMILY_TREE: dict[str, list[str]] = {
    "claude-anthropic": [
        "claude-3-haiku", "claude-haiku-4-5",
        "claude-sonnet-4", "claude-sonnet-4-5", "claude-sonnet-4-6",
        "claude-opus-4", "claude-opus-4-1",
        "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
    ],
    "cursor-composer": ["composer-2", "auto", "composer-1", "composer-1-5"],
    "openai-gpt5": [
        "gpt-5-4-nano", "gpt-5-mini", "gpt-5-1-codex-mini",
        "gpt-5-2-mini", "gpt-5-4-mini",
        "gpt-5", "gpt-5-codex", "gpt-5-1-codex", "gpt-5-1-codex-max",
        "gpt-5-fast", "gpt-5-2", "gpt-5-2-codex", "gpt-5-3-codex",
        "gpt-5-4",
    ],
    "openai-gpt4": ["gpt-4-1-nano", "gpt-4-1-mini", "gpt-4o-mini", "gpt-4-1", "gpt-4o"],
    "openai-reasoning": ["o3-mini", "o4-mini", "o1-mini", "o3", "o1"],
    "google-gemini": [
        "gemini-2-5-flash", "gemini-3-flash",
        "gemini-2-5-pro", "gemini-3-pro", "gemini-3-1-pro",
    ],
    "xai-grok": ["grok-3-mini", "grok-4-20", "grok-3", "grok-4"],
    "moonshot-kimi": ["kimi-k2", "kimi-k2-5"],
    "mistral": ["mistral-small", "codestral", "mistral-medium", "mistral-large"],
    "deepseek": ["deepseek-chat", "deepseek-v3", "deepseek-reasoner", "deepseek-r1"],
}


PRICING_SOURCE: dict[str, Any] = {
    "anthropic_url": "https://www.anthropic.com/pricing",
    "openai_url":    "https://openai.com/api/pricing",
    "google_url":    "https://ai.google.dev/pricing",
    "xai_url":       "https://x.ai/api",
    "moonshot_url":  "https://platform.moonshot.ai/docs/pricing",
    "mistral_url":   "https://mistral.ai/technology",
    "deepseek_url":  "https://api-docs.deepseek.com/quick_start/pricing",
    "cursor_url":    "https://cursor.com/pricing",
    "fetched_at":    "2026-04-17",
    "notes": (
        "Keys are stored dash-canonicalised (``gpt-5-2``); the resolver normalises "
        "dots → dashes so real model IDs like ``gpt-5.2-2025-12-11`` match. "
        "Cursor's API pool bills at each provider's API rate, so Anthropic/OpenAI "
        "models in Cursor match their published per-token pricing 1:1. "
        "Cursor Opus 4.6 'Fast mode' is 6× normal Opus."
    ),
}


def resolve_pricing(model: str | None) -> dict[str, float] | None:
    """Longest-prefix match after dot-canonicalisation.

    Handles all of:
        resolve_pricing("gpt-5.2-2025-12-11") → gpt-5-2 rates
        resolve_pricing("gpt-5-2")            → gpt-5-2 rates
        resolve_pricing("claude-opus-4-7")    → claude-opus-4-7 rates
        resolve_pricing("openai/gpt-5-4")     → gpt-5-4 rates (strips vendor prefix)
        resolve_pricing("unknown-x")          → None
    """
    if not isinstance(model, str) or not model:
        return None
    c = _canon(model)
    # strip common vendor prefixes used by aggregators (OpenRouter, etc.)
    for prefix in ("openai/", "anthropic/", "google/", "xai/", "mistralai/",
                   "deepseek/", "moonshotai/", "openrouter/"):
        if c.startswith(prefix):
            c = c[len(prefix):]
            break
    if c in MODEL_PRICING_USD_PER_MTOK:
        return MODEL_PRICING_USD_PER_MTOK[c]
    candidates = sorted(
        ((k, v) for k, v in MODEL_PRICING_USD_PER_MTOK.items() if c.startswith(k)),
        key=lambda kv: len(kv[0]), reverse=True,
    )
    return candidates[0][1] if candidates else None


def model_family(model: str | None) -> str:
    """Map a raw model id onto one of ``MODEL_FAMILY_TREE``'s keys, else 'other'."""
    if not isinstance(model, str):
        return "other"
    c = _canon(model)
    for prefix in ("openai/", "anthropic/", "google/", "xai/", "mistralai/",
                   "deepseek/", "moonshotai/", "openrouter/"):
        if c.startswith(prefix):
            c = c[len(prefix):]
            break
    if c.startswith("claude"): return "claude-anthropic"
    if c.startswith("composer") or c == "auto": return "cursor-composer"
    if c.startswith("gpt-5"): return "openai-gpt5"
    if c.startswith("gpt-4") or c.startswith("gpt-3"): return "openai-gpt4"
    if c.startswith("o1") or c.startswith("o3") or c.startswith("o4"): return "openai-reasoning"
    if c.startswith("gemini"): return "google-gemini"
    if c.startswith("grok"): return "xai-grok"
    if c.startswith("kimi"): return "moonshot-kimi"
    if c.startswith("mistral") or c.startswith("codestral"): return "mistral"
    if c.startswith("deepseek"): return "deepseek"
    return "other"


def cheaper_siblings(model: str | None) -> list[str]:
    """Return family members strictly cheaper than ``model`` on blended input+output.

    Blended rate = input + output (treats both dimensions as rough tier proxy).
    Ordered cheapest-first. Empty list means no cheaper sibling exists in the
    family (model is already the floor, or model is out-of-family).
    """
    fam = model_family(model)
    tree = MODEL_FAMILY_TREE.get(fam, [])
    if not tree:
        return []
    target = resolve_pricing(model)
    if not target:
        return []
    target_blend = target["input"] + target["output"]
    out = []
    for sib in tree:
        p = MODEL_PRICING_USD_PER_MTOK.get(sib)
        if not p:
            continue
        if p["input"] + p["output"] < target_blend * 0.999:  # strict cheaper
            out.append(sib)
    return sorted(out, key=lambda s: (
        MODEL_PRICING_USD_PER_MTOK[s]["input"] + MODEL_PRICING_USD_PER_MTOK[s]["output"]
    ))


def cheapest_sibling(model: str | None) -> str | None:
    sibs = cheaper_siblings(model)
    return sibs[0] if sibs else None


def estimate_cost_row(row: dict | pd.Series) -> float:
    model = row.get("model")
    p = resolve_pricing(model)
    if not p:
        return 0.0
    inp = (row.get("input_tokens") or 0) or 0
    out = (row.get("output_tokens") or 0) or 0
    cw  = (row.get("cache_creation_input_tokens") or 0) or 0
    cr  = (row.get("cache_read_input_tokens") or 0) or 0
    return (
        inp * p["input"]         / 1_000_000
        + out * p["output"]      / 1_000_000
        + cw  * p["cache_write"] / 1_000_000
        + cr  * p["cache_read"]  / 1_000_000
    )


def reprice_row(row: dict | pd.Series, target_model: str) -> float:
    p = resolve_pricing(target_model)
    if not p:
        return 0.0
    inp = (row.get("input_tokens") or 0) or 0
    out = (row.get("output_tokens") or 0) or 0
    cw  = (row.get("cache_creation_input_tokens") or 0) or 0
    cr  = (row.get("cache_read_input_tokens") or 0) or 0
    return (
        inp * p["input"]         / 1_000_000
        + out * p["output"]      / 1_000_000
        + cw  * p["cache_write"] / 1_000_000
        + cr  * p["cache_read"]  / 1_000_000
    )


def add_counterfactual_columns(
    messages_df: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """Attach ``estimated_usd`` + per-target ``cost_if_<target>`` + per-row
    ``cheapest_sibling`` + ``cost_if_cheapest_sibling`` columns.

    The cheapest-sibling column is what powers the family-aware headline: for
    each message, we find the cheapest model strictly in the same family and
    reprice against it. This is what "could have been emitted by a cheaper
    sibling of the same vendor" translates to in dollar terms.
    """
    df = messages_df.copy()
    df["estimated_usd"] = df.apply(lambda r: estimate_cost_row(r.to_dict()), axis=1)
    for target in targets:
        col = f"cost_if_{_col(target)}"
        df[col] = df.apply(lambda r, t=target: reprice_row(r.to_dict(), t), axis=1)
    df["cheapest_sibling"] = df["model"].apply(cheapest_sibling)
    df["cost_if_cheapest_sibling"] = df.apply(
        lambda r: reprice_row(r.to_dict(), r["cheapest_sibling"])
        if r.get("cheapest_sibling") else 0.0,
        axis=1,
    )
    return df


def _col(target: str) -> str:
    """Sanitise a model id into a DataFrame-column-safe suffix.
    claude-sonnet-4-6 → sonnet_4_6 ; gpt-5-4-mini → gpt_5_4_mini
    """
    c = _canon(target).replace("claude-", "")
    return c.replace("-", "_")
