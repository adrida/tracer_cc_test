"""End-to-end analysis: redacted tables → priced → mechanical → embedded →
clustered → report.

The open structural gate. The pipeline:

    1. Price every assistant message at its actual model's rate.
    2. Identify "premium" assistants — those whose model has at least one
       strictly-cheaper sibling in the same family tree. (Generalises the old
       Opus-only filter: any gpt-5-4 turn now clusters against gpt-5-4-mini,
       any gemini-3-pro against gemini-3-flash, etc.)
    3. Among premium assistants, pick mechanical-only turns (no reasoning text,
       no thinking, tool_use-only pointing at MECHANICAL_TOOLS).
    4. Embed their first-tool-call texts.
    5. Cluster; re-route ONLY turns inside dense clusters; noise is excluded.
    6. Save = actual_cost - cost_if_cheapest_sibling for the routed turns.

The **behavioural** gate — replay-on-cheaper-model + measure-agreement —
is not in this module. See ``backend/__init__.py`` for the moat boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .clustering import cluster_turns, medoid_and_examples
from .embedding import embed_texts_async
from .mechanical import (
    MECHANICAL_TOOLS,
    is_mechanical_assistant_turn,
    label_from_medoid,
    mechanical_turn_text,
)
from .pricing import (
    PRICING_SOURCE,
    _col,
    add_counterfactual_columns,
    cheaper_siblings,
    cheapest_sibling,
    estimate_cost_row,
    model_family,
    resolve_pricing,
)
from .schema import (
    AnalyzeRequest,
    ClusterCard,
    ClusterLabel,
    ClusterMapPoint,
    FunStat,
    RoutingPolicy,
    RoutingRule,
    SessionCard,
    WrappedReport,
)


def _short_session(uuid: str) -> str:
    return uuid.split("-")[0] if uuid else "?"


def _to_df(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.model_dump() if hasattr(r, "model_dump") else r for r in rows])


async def analyze(req: AnalyzeRequest) -> WrappedReport:
    sessions = _to_df(req.data.sessions)
    messages = _to_df(req.data.messages)
    tool_calls = _to_df(req.data.tool_calls)
    prompts = _to_df(req.data.prompts)
    errors = _to_df(req.data.errors)

    if messages.empty:
        return _empty_report(req, sessions, prompts, "no messages in payload")

    targets = req.options.targets
    msgs = add_counterfactual_columns(messages, targets=targets)
    total_spend = float(msgs["estimated_usd"].fillna(0).sum())

    asst = msgs[msgs["type"] == "assistant"].copy()
    asst["family"] = asst["model"].apply(model_family)

    # legacy spend rollups for Anthropic-centric dashboards
    fam_lower = asst["model"].fillna("").str.lower()
    opus_spend = float(asst.loc[fam_lower.str.contains("opus"), "estimated_usd"].fillna(0).sum())
    sonnet_spend = float(asst.loc[fam_lower.str.contains("sonnet"), "estimated_usd"].fillna(0).sum())
    haiku_spend = float(asst.loc[fam_lower.str.contains("haiku"), "estimated_usd"].fillna(0).sum())

    # generalised per-family rollup
    spend_by_family: dict[str, float] = (
        asst.groupby("family")["estimated_usd"].sum().fillna(0).to_dict()
    )

    # "premium" = any assistant with at least one cheaper sibling in its family.
    # This is what replaces the old opus-only gate. An Anthropic Opus turn is
    # premium (cheaper siblings: sonnet, haiku); a GPT-5.4 turn is premium
    # (cheaper siblings: gpt-5-4-mini, gpt-5-4-nano); a haiku turn is NOT
    # (nothing cheaper in its family).
    asst["has_cheaper_sibling"] = asst["model"].apply(lambda m: bool(cheaper_siblings(m)))
    premium = asst[asst["has_cheaper_sibling"]].copy()
    premium_spend = float(premium["estimated_usd"].fillna(0).sum())

    # corpus span
    sess_starts = pd.to_datetime(
        sessions.get("first_event_at", pd.Series([], dtype=object)),
        errors="coerce", utc=True,
    ) if not sessions.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    sess_ends = pd.to_datetime(
        sessions.get("last_event_at", pd.Series([], dtype=object)),
        errors="coerce", utc=True,
    ) if not sessions.empty else pd.Series([], dtype="datetime64[ns, UTC]")
    earliest = sess_starts.min() if len(sess_starts) else pd.NaT
    latest = sess_ends.max() if len(sess_ends) and not sess_ends.isna().all() else (
        sess_starts.max() if len(sess_starts) else pd.NaT
    )
    span_days = max(1, int((latest - earliest).days)) if pd.notna(earliest) and pd.notna(latest) else 1

    cache_share = 0.0
    if total_spend > 0 and "cache_read_input_tokens" in msgs.columns:
        cache_tok = msgs["cache_read_input_tokens"].fillna(0).sum()
        if cache_tok > 0:
            # blended cache-read rate varies; pick the sessions' dominant-model
            # rate as proxy. Fall back to Opus cache-read ($0.50) for legacy.
            est_cache_cost = float(cache_tok) * 0.50 / 1e6
            cache_share = min(1.0, est_cache_cost / max(total_spend, 1e-9))

    if premium.empty or tool_calls.empty:
        return _no_premium_report(
            req, sessions, msgs, prompts, total_spend, opus_spend, sonnet_spend,
            haiku_spend, premium_spend, spend_by_family, cache_share,
            earliest, latest, span_days,
        )

    reasoning_threshold = int(getattr(req.options, "reasoning_threshold_chars", 0) or 0)
    premium["mech_only"] = premium["uuid"].apply(
        lambda u: is_mechanical_assistant_turn(
            u, messages, tool_calls,
            reasoning_threshold_chars=reasoning_threshold,
        )
    )
    mech = premium[premium["mech_only"]].copy()
    if mech.empty:
        return _no_premium_report(
            req, sessions, msgs, prompts, total_spend, opus_spend, sonnet_spend,
            haiku_spend, premium_spend, spend_by_family, cache_share,
            earliest, latest, span_days,
        )

    tc = tool_calls[tool_calls["parent_assistant_uuid"].isin(mech["uuid"])].copy()
    if "started_at" in tc.columns:
        tc = tc.sort_values(["parent_assistant_uuid", "started_at"])
    first_per_turn = tc.groupby("parent_assistant_uuid").first().reset_index()
    first_per_turn["embed_text"] = first_per_turn.apply(
        lambda r: mechanical_turn_text(r.get("tool_name"), r.get("input_preview")),
        axis=1,
    )
    mech_w = mech.merge(
        first_per_turn[["parent_assistant_uuid", "tool_name", "embed_text"]],
        left_on="uuid", right_on="parent_assistant_uuid", how="left",
    ).dropna(subset=["embed_text"]).reset_index(drop=True)

    n_mech = len(mech_w)
    if n_mech == 0:
        return _no_premium_report(
            req, sessions, msgs, prompts, total_spend, opus_spend, sonnet_spend,
            haiku_spend, premium_spend, spend_by_family, cache_share,
            earliest, latest, span_days,
        )

    emb = await embed_texts_async(mech_w["embed_text"].tolist())

    if req.options.min_cluster_size:
        min_cs = req.options.min_cluster_size
    else:
        min_cs = max(5, min(20, n_mech // 12))
    min_samples = max(2, min(5, min_cs // 3))
    labels, backend = cluster_turns(emb, min_cluster_size=min_cs, min_samples=min_samples)
    mech_w["turn_cluster"] = labels
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())

    gated = mech_w[mech_w["turn_cluster"] >= 0].copy()
    spend_in_clusters = float(gated["estimated_usd"].fillna(0).sum())
    spend_excluded = premium_spend - spend_in_clusters

    # family-aware headline saving
    save_cheapest_sibling = float(
        (gated["estimated_usd"] - gated.get("cost_if_cheapest_sibling", 0)).sum()
    )

    # legacy Anthropic-targeted savings (for back-compat with old renderers)
    def _save_at(target: str, df: pd.DataFrame) -> float:
        col = f"cost_if_{_col(target)}"
        if col not in df.columns:
            return 0.0
        return float((df["estimated_usd"] - df[col]).sum())

    save_haiku = _save_at("claude-haiku-4-5", gated)
    save_sonnet = _save_at("claude-sonnet-4-6", gated)
    save_composer2 = _save_at("composer-2", gated)

    ceiling_cheapest_sibling = float(
        (premium["estimated_usd"] - premium.apply(
            lambda r: (resolve_pricing(r["cheapest_sibling"]) or {}).get("input", 0) * (r["input_tokens"] or 0) / 1e6
            + (resolve_pricing(r["cheapest_sibling"]) or {}).get("output", 0) * (r["output_tokens"] or 0) / 1e6
            if r.get("cheapest_sibling") else 0,
            axis=1,
        )).sum()
    ) if not premium.empty else 0.0

    ceiling_haiku = _save_at("claude-haiku-4-5", premium)
    ceiling_sonnet = _save_at("claude-sonnet-4-6", premium)
    ceiling_composer2 = _save_at("composer-2", premium)

    # cluster cards — prefer the family-aware saving as the headline number
    top_clusters: list[ClusterCard] = []
    if not gated.empty:
        summary_agg: dict[str, Any] = {
            "n": ("uuid", "count"),
            "actual": ("estimated_usd", "sum"),
            "save_sibling": ("estimated_usd", "sum"),  # overwritten below
            "dominant_tool": ("tool_name", lambda s: s.mode().iloc[0] if not s.mode().empty else "?"),
        }
        if "cost_if_cheapest_sibling" in gated.columns:
            gated["delta_sibling"] = gated["estimated_usd"] - gated["cost_if_cheapest_sibling"]
            summary_agg["save_sibling"] = ("delta_sibling", "sum")
        for t, col_key in (("claude-haiku-4-5", "if_haiku"), ("claude-sonnet-4-6", "if_sonnet"), ("composer-2", "if_composer2")):
            c = f"cost_if_{_col(t)}"
            if c in gated.columns:
                summary_agg[col_key] = (c, "sum")

        summary = gated.groupby("turn_cluster").agg(**summary_agg).sort_values("actual", ascending=False)

        embed_texts_list = mech_w["embed_text"].tolist()
        for cid in summary.head(8).index:
            idx = mech_w.index[mech_w["turn_cluster"] == cid].to_numpy()
            medoid_text, examples = medoid_and_examples(emb, idx, embed_texts_list, k=3)
            row = summary.loc[cid]
            top_clusters.append(ClusterCard(
                cluster_id=int(cid),
                n_turns=int(row["n"]),
                dominant_tool=str(row["dominant_tool"]),
                actual_usd=float(row["actual"]),
                save_haiku_usd=float(row["actual"] - row["if_haiku"]) if "if_haiku" in row else 0.0,
                save_sonnet_usd=float(row["actual"] - row["if_sonnet"]) if "if_sonnet" in row else 0.0,
                save_composer2_usd=float(row["actual"] - row["if_composer2"]) if "if_composer2" in row else 0.0,
                save_cheapest_sibling_usd=float(row.get("save_sibling", 0.0)),
                medoid_text=(medoid_text or "")[:240],
                examples=[(e or "")[:240] for e in examples],
                label=label_from_medoid(medoid_text or "", str(row["dominant_tool"])),
            ))

    # session cards
    sess_spend = msgs.groupby("session_id")["estimated_usd"].sum().fillna(0).rename("spend")
    top_session_ids = sess_spend.sort_values(ascending=False).head(5).index.tolist()
    top_sessions: list[SessionCard] = []
    for sid in top_session_ids:
        if sessions.empty:
            continue
        sess_row = sessions[sessions["session_id"] == sid]
        if sess_row.empty:
            continue
        sess_row = sess_row.iloc[0]
        sess_msgs = msgs[msgs["session_id"] == sid]
        sess_prompt_count = int(((sess_msgs["type"] == "user")).sum())
        sess_tc = tool_calls[tool_calls["session_id"] == sid] if not tool_calls.empty else pd.DataFrame()
        top_tool = (sess_tc["tool_name"].mode().iloc[0] if not sess_tc.empty and not sess_tc["tool_name"].mode().empty else "—")
        dominant = (sess_msgs[sess_msgs["type"] == "assistant"]["model"].fillna("").mode())
        top_sessions.append(SessionCard(
            session_short=_short_session(sid),
            cwd=str(sess_row.get("decoded_cwd") or sess_row.get("project_dir") or "")[:80],
            started_at=str(sess_row.get("first_event_at") or "")[:19],
            n_prompts=sess_prompt_count,
            spend_usd=float(sess_spend.loc[sid]),
            dominant_model=(dominant.iloc[0] if not dominant.empty else "—")[:40],
            top_tool=str(top_tool),
        ))

    fun_stats = _build_fun_stats(sessions, errors, msgs, mech_w, top_clusters, tool_calls)

    # Build the interactive cluster map (2D projection + per-cluster labels).
    clustermap, cluster_labels = _build_clustermap(mech_w, emb, top_clusters)

    # Build the actionable routing policy from the gated clusters. Each rule
    # maps a tool-name signature to a cheaper-sibling target model with
    # per-cluster evidence. Runtime consumer: tracercc.runtime.Router.
    routing_policy = _build_routing_policy(
        gated=gated,
        mech_w=mech_w,
        emb=emb,
        premium=premium,
        premium_spend=premium_spend,
        sessions=sessions,
        n_sessions=int(len(sessions)),
        span_days=span_days,
        reasoning_threshold=reasoning_threshold,
    )

    saving_pct_of_premium = (100 * save_cheapest_sibling / premium_spend) if premium_spend > 0 else 0.0

    return WrappedReport(
        n_sessions=int(len(sessions)),
        n_messages=int(len(messages)),
        n_prompts=int(len(prompts)),
        earliest=str(earliest)[:19] if pd.notna(earliest) else "—",
        latest=str(latest)[:19] if pd.notna(latest) else "—",
        span_days=span_days,
        source=req.source,
        total_spend_usd=total_spend,
        opus_spend_usd=opus_spend,
        sonnet_spend_usd=sonnet_spend,
        haiku_spend_usd=haiku_spend,
        cache_read_share=cache_share,
        spend_by_family=spend_by_family,
        premium_spend_usd=premium_spend,
        n_mechanical_turns=int(n_mech),
        n_clusters=int(n_clusters),
        n_noise=int(n_noise),
        spend_in_clusters_usd=spend_in_clusters,
        spend_excluded_usd=spend_excluded,
        saving_haiku_usd=save_haiku,
        saving_sonnet_usd=save_sonnet,
        saving_composer2_usd=save_composer2,
        saving_cheapest_sibling_usd=save_cheapest_sibling,
        saving_haiku_pct_of_opus=(100 * save_haiku / opus_spend) if opus_spend > 0 else 0.0,
        saving_cheapest_sibling_pct_of_premium=saving_pct_of_premium,
        ceiling_haiku_usd=ceiling_haiku,
        ceiling_haiku_pct_of_opus=(100 * ceiling_haiku / opus_spend) if opus_spend > 0 else 0.0,
        ceiling_sonnet_usd=ceiling_sonnet,
        ceiling_composer2_usd=ceiling_composer2,
        ceiling_cheapest_sibling_usd=ceiling_cheapest_sibling,
        top_clusters=top_clusters,
        top_sessions=top_sessions,
        fun_stats=fun_stats,
        routing_policy=routing_policy,
        clustermap=clustermap,
        cluster_labels=cluster_labels,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backend_version=__version__,
        cluster_backend=backend,
        pricing_source=PRICING_SOURCE,
        gate="structural",
    )


def _build_fun_stats(
    sessions: pd.DataFrame,
    errors: pd.DataFrame,
    msgs: pd.DataFrame,
    mech_w: pd.DataFrame,
    top_clusters: list[ClusterCard],
    tool_calls: pd.DataFrame,
) -> list[FunStat]:
    out: list[FunStat] = []
    if top_clusters:
        c0 = top_clusters[0]
        save_shown = c0.save_cheapest_sibling_usd or c0.save_haiku_usd
        out.append(FunStat(
            title="Most expensive copy-paste",
            value=f"{c0.n_turns}× {c0.label}",
            detail=f"You paid ${c0.actual_usd:,.2f} on the premium tier to repeat one "
                   f"pattern. A cheaper sibling in the same family could have emitted "
                   f"it for ${c0.actual_usd - save_shown:,.2f}.",
        ))
    asst = msgs[msgs["type"] == "assistant"]
    out.append(FunStat(
        title="Assistant turns",
        value=f"{len(asst):,}",
        detail=f"Of which {len(mech_w):,} were mechanical-only "
               f"({100 * len(mech_w) / max(1, len(asst)):.0f}% — work that doesn't need reasoning).",
    ))
    if not tool_calls.empty and "tool_name" in tool_calls.columns:
        tc_counts = tool_calls["tool_name"].value_counts().head(3)
        tools_str = ", ".join(f"{n}× {t}" for t, n in tc_counts.items())
        out.append(FunStat(
            title="Top tools", value=tools_str,
            detail="Across every session in the window.",
        ))
    if not sessions.empty and "n_compactions" in sessions.columns:
        n_compact = int(sessions["n_compactions"].sum() or 0)
        if n_compact > 0:
            out.append(FunStat(
                title="Context compactions", value=f"{n_compact}",
                detail="Each compaction is a sign the session ran long enough to overflow context. "
                       "Often correlates with cache-read spend.",
            ))
    n_errors = int(len(errors))
    if n_errors > 0:
        out.append(FunStat(
            title="API / tool errors observed", value=f"{n_errors}",
            detail="Each one is a retry on your dollar.",
        ))
    return out


def _build_clustermap(
    mech_w: pd.DataFrame,
    emb: np.ndarray,
    top_clusters: list[ClusterCard],
) -> tuple[list[ClusterMapPoint], list[ClusterLabel]]:
    """Project mechanical-turn embeddings to 2D and pair with per-cluster labels.

    Uses TSNE (sklearn, always available in [serve]) for better visual
    separation than linear PCA. Falls back to PCA if TSNE fails or the corpus
    is too small (< 5 points). Each turn becomes one point on the scatter;
    cluster centroids get anchored labels.

    Costs are mech_w.estimated_usd per turn — the frontend uses them for
    point size so expensive patterns pop visually.
    """
    n = int(len(mech_w))
    if n == 0 or emb.shape[0] == 0:
        return [], []

    # 2D projection
    coords: np.ndarray
    try:
        if n >= 5:
            from sklearn.manifold import TSNE
            perp = max(5.0, min(30.0, (n - 1) / 3.0))
            coords = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=42,
                init="pca",
                learning_rate="auto",
                max_iter=500,
                metric="cosine",
            ).fit_transform(emb)
        else:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=min(2, emb.shape[0] - 1 or 1)).fit_transform(emb)
            if coords.shape[1] == 1:
                coords = np.column_stack([coords, np.zeros(n)])
    except Exception:
        try:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(emb)
        except Exception:
            return [], []

    # Rescale to a pleasant range for the frontend ([-50, 50] on each axis).
    coords = np.asarray(coords, dtype=float)
    for j in range(coords.shape[1]):
        col = coords[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            coords[:, j] = ((col - lo) / (hi - lo) - 0.5) * 100.0
        else:
            coords[:, j] = 0.0

    mech_w = mech_w.reset_index(drop=True)
    labels = mech_w["turn_cluster"].to_numpy() if "turn_cluster" in mech_w.columns else np.full(n, -1)
    tool_names = mech_w["tool_name"].fillna("?").to_numpy() if "tool_name" in mech_w.columns else np.full(n, "?")
    texts = mech_w["embed_text"].fillna("").astype(str).to_numpy() if "embed_text" in mech_w.columns else np.full(n, "")
    costs = mech_w["estimated_usd"].fillna(0).astype(float).to_numpy() if "estimated_usd" in mech_w.columns else np.zeros(n)

    points: list[ClusterMapPoint] = []
    for i in range(n):
        points.append(ClusterMapPoint(
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            cluster_id=int(labels[i]),
            tool_name=str(tool_names[i])[:40],
            cost_usd=float(costs[i]),
            text=str(texts[i])[:160],
        ))

    # Per-cluster labels (centroid + metadata from the ClusterCard cards already built).
    labels_by_cluster: dict[int, ClusterCard] = {c.cluster_id: c for c in top_clusters}
    cluster_labels: list[ClusterLabel] = []
    for cid in sorted({int(l) for l in labels} - {-1}):
        mask = labels == cid
        cx = float(coords[mask, 0].mean())
        cy = float(coords[mask, 1].mean())
        card = labels_by_cluster.get(cid)
        # pull confidence back off the routing-rule confidence policy bridge
        # (we don't have it yet here — just use size thresholds mirroring the
        # rule heuristic so the plot and the policy table agree).
        n_turns = int(mask.sum())
        if card is None:
            lab = f"cluster {cid}"
            dom = str(tool_names[mask][0]) if mask.any() else "?"
        else:
            lab = card.label
            dom = card.dominant_tool
        if n_turns >= 50:
            conf = "high"
        elif n_turns >= 20:
            conf = "medium"
        else:
            conf = "low"
        cluster_labels.append(ClusterLabel(
            cluster_id=cid, x=cx, y=cy, label=lab[:60],
            dominant_tool=dom[:40], n_turns=n_turns, confidence=conf,
        ))

    return points, cluster_labels


def _build_routing_policy(
    gated: pd.DataFrame,
    mech_w: pd.DataFrame,
    emb: np.ndarray,
    premium: pd.DataFrame,
    premium_spend: float,
    sessions: pd.DataFrame,
    n_sessions: int,
    span_days: int,
    reasoning_threshold: int,
) -> RoutingPolicy | None:
    """Assemble per-cluster routing rules from the gated clustering output.

    For every dense cluster that passed the structural gate, emit a rule
    that says "when the session's most-recent assistant tool call matches
    one of these names, route the NEXT call to <cheaper sibling>". The
    runtime consumes this list of rules and rewrites the ``model`` param
    on outgoing OpenAI calls whose signature matches.

    Confidence is a coarse function of cluster size (more evidence = higher)
    plus within-cluster embedding density (tighter = higher). This is the
    structural-gate proxy for the behavioural parity gate that the hosted
    tracerCC backend will add later.
    """
    if gated.empty:
        return None

    from .pricing import cheapest_sibling, resolve_pricing

    dominant_overall_model = (
        premium["model"].mode().iloc[0] if not premium["model"].mode().empty else ""
    )

    rules: list[RoutingRule] = []
    for cid, sub in gated.groupby("turn_cluster"):
        tool_names_seen = sorted({t for t in sub["tool_name"].dropna().unique()})
        dominant_tool = sub["tool_name"].mode().iloc[0] if not sub["tool_name"].mode().empty else "?"
        dominant_model = sub["model"].mode().iloc[0] if not sub["model"].mode().empty else dominant_overall_model
        target = cheapest_sibling(dominant_model) or "(no cheaper sibling)"

        n_turns = int(len(sub))
        actual_usd = float(sub["estimated_usd"].fillna(0).sum())
        if_cheap = float(sub.get("cost_if_cheapest_sibling", pd.Series([0] * n_turns)).fillna(0).sum())
        save = max(actual_usd - if_cheap, 0.0)
        per_call = save / n_turns if n_turns > 0 else 0.0
        # Window projection: scale the observed savings to 30 days assuming
        # steady-state traffic at the corpus' own rate.
        window_scale = 30.0 / max(span_days, 1)
        window_proj = save * window_scale

        # Within-cluster cosine density → confidence proxy. Tight cluster
        # with many turns = high; sparse or small = low.
        idx = sub.index.to_numpy()
        if len(idx) >= 2 and len(emb) > max(idx):
            from sklearn.metrics.pairwise import cosine_distances
            sub_emb = emb[idx]
            d = cosine_distances(sub_emb)
            iu = np.triu_indices_from(d, k=1)
            avg_dist = float(d[iu].mean()) if iu[0].size else 1.0
        else:
            avg_dist = 1.0
        # tight = avg_dist < 0.15, loose = > 0.35
        if n_turns >= 50 and avg_dist < 0.15:
            conf = "high"
        elif n_turns >= 20 and avg_dist < 0.25:
            conf = "medium"
        else:
            conf = "low"

        # Heuristic predicate. The simplest usable runtime signature is:
        # "the last assistant tool call in the session used one of these
        # tool names." Compose it as a JSON-serialisable dict.
        predicate = {
            "type": "last_tool_call_name_in",
            "tool_names": tool_names_seen,
            "cluster_label": sub.iloc[0].get("tool_name") or dominant_tool,
            "reasoning_threshold_chars": int(reasoning_threshold),
        }

        medoid = (
            sub["embed_text"].iloc[0] if "embed_text" in sub.columns else ""
        )
        medoid = (medoid or "")[:240]

        rules.append(RoutingRule(
            rule_id=f"rule-{int(cid):03d}",
            cluster_id=int(cid),
            label=f"{dominant_tool} pattern",
            predicate=predicate,
            source_model=str(dominant_model),
            target_model=str(target),
            confidence=conf,
            n_training_turns=n_turns,
            medoid_example=medoid,
            estimated_savings_per_call_usd=round(per_call, 6),
            estimated_window_savings_usd=round(window_proj, 4),
        ))

    # Sort by projected window savings — biggest levers first.
    rules.sort(key=lambda r: r.estimated_window_savings_usd, reverse=True)

    return RoutingPolicy(
        policy_version="1.0",
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        corpus_summary={
            "sessions": n_sessions,
            "assistant_turns": int(len(premium)),
            "span_days": int(span_days),
            "premium_spend_usd": round(float(premium_spend), 4),
        },
        default_model=str(dominant_overall_model) or "",
        rules=rules,
        gate="structural",
        reasoning_threshold_chars=int(reasoning_threshold),
    )


def _no_premium_report(req, sessions, msgs, prompts, total_spend, opus_spend, sonnet_spend,
                       haiku_spend, premium_spend, spend_by_family, cache_share,
                       earliest, latest, span_days) -> WrappedReport:
    return WrappedReport(
        n_sessions=int(len(sessions)),
        n_messages=int(len(req.data.messages)),
        n_prompts=int(len(prompts)),
        earliest=str(earliest)[:19] if pd.notna(earliest) else "—",
        latest=str(latest)[:19] if pd.notna(latest) else "—",
        span_days=span_days,
        source=req.source,
        total_spend_usd=float(total_spend),
        opus_spend_usd=float(opus_spend),
        sonnet_spend_usd=float(sonnet_spend),
        haiku_spend_usd=float(haiku_spend),
        cache_read_share=float(cache_share),
        spend_by_family=spend_by_family,
        premium_spend_usd=float(premium_spend),
        n_mechanical_turns=0, n_clusters=0, n_noise=0,
        spend_in_clusters_usd=0.0, spend_excluded_usd=float(premium_spend),
        saving_haiku_usd=0.0, saving_sonnet_usd=0.0, saving_composer2_usd=0.0,
        saving_cheapest_sibling_usd=0.0,
        saving_haiku_pct_of_opus=0.0, saving_cheapest_sibling_pct_of_premium=0.0,
        ceiling_haiku_usd=0.0, ceiling_haiku_pct_of_opus=0.0,
        ceiling_sonnet_usd=0.0, ceiling_composer2_usd=0.0, ceiling_cheapest_sibling_usd=0.0,
        top_clusters=[], top_sessions=[],
        fun_stats=[FunStat(
            title="No mechanical premium-tier work found",
            value="$0 saveable",
            detail="Either the corpus is too small, every premium turn carries reasoning, "
                   "or your dominant model has no cheaper sibling in its family. "
                   "Run more sessions and try again.",
        )],
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backend_version=__version__,
        cluster_backend="noop",
        pricing_source=PRICING_SOURCE,
        gate="structural",
    )


def _empty_report(req, sessions, prompts, reason: str) -> WrappedReport:
    return WrappedReport(
        n_sessions=int(len(sessions)),
        n_messages=0, n_prompts=int(len(prompts)),
        earliest="—", latest="—", span_days=1, source=req.source,
        total_spend_usd=0.0, opus_spend_usd=0.0, sonnet_spend_usd=0.0,
        haiku_spend_usd=0.0, cache_read_share=0.0,
        spend_by_family={}, premium_spend_usd=0.0,
        n_mechanical_turns=0, n_clusters=0, n_noise=0,
        spend_in_clusters_usd=0.0, spend_excluded_usd=0.0,
        saving_haiku_usd=0.0, saving_sonnet_usd=0.0, saving_composer2_usd=0.0,
        saving_cheapest_sibling_usd=0.0,
        saving_haiku_pct_of_opus=0.0, saving_cheapest_sibling_pct_of_premium=0.0,
        ceiling_haiku_usd=0.0, ceiling_haiku_pct_of_opus=0.0,
        ceiling_sonnet_usd=0.0, ceiling_composer2_usd=0.0, ceiling_cheapest_sibling_usd=0.0,
        top_clusters=[], top_sessions=[],
        fun_stats=[FunStat(title="Empty payload", value="—", detail=reason)],
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backend_version=__version__,
        cluster_backend="noop",
        pricing_source=PRICING_SOURCE,
        gate="structural",
    )
