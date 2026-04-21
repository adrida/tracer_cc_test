"""Runtime routing — lookup a fitted tracerCC policy at call time.

Philosophy mirrors the classification Tracer (``adrida/tracer``):
  1. Fit offline on the agent's own past traces → produces ``policy.json``
     with per-cluster rules that map signatures to cheaper-sibling models.
  2. Deploy the router inline at the OpenAI-call boundary.
  3. Each call checks the session's recent signature against the policy;
     if a rule matches with high-enough confidence, the ``model`` parameter
     is rewritten to the cheaper target. Otherwise pass-through.
  4. Record the decision so the next refit sees the outcome distribution
     (supports the flywheel; optional at first).

Safety by default:
  - Unknown tool names, empty sessions, or no-match signatures → default model.
  - The router never INVENTS a tool call. It only rewrites the ``model`` param.
  - ``min_confidence`` gate — defaults to "medium"; the client can tighten
    to "high" or loosen to "low" depending on appetite.
  - The matched rule must declare a non-empty ``target_model`` that differs
    from the ``source_model`` AND the current request's default model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class _Rule:
    rule_id: str
    cluster_id: int
    label: str
    predicate: dict
    source_model: str
    target_model: str
    confidence: str
    estimated_savings_per_call_usd: float


def load_policy(path: str | Path) -> dict:
    """Read a fitted policy JSON from disk. Raises on missing/malformed file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_policy(
    policy: dict,
    *,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    default_model: str | None = None,
    min_confidence: str = "medium",
) -> tuple[str, str | None]:
    """Functional form of the router — convenient for stateless callers.

    Returns ``(target_model, rule_id_or_None)``. If no rule matches, returns
    ``(default_model or policy_default, None)``.
    """
    return Router(policy, default_model=default_model, min_confidence=min_confidence).route(
        messages=messages, tools=tools,
    )


class Router:
    """Wraps a fitted policy and applies routing rules at call time."""

    def __init__(
        self,
        policy: dict,
        default_model: str | None = None,
        min_confidence: str = "medium",
    ) -> None:
        self.policy = policy
        self.default_model = default_model or policy.get("default_model") or ""
        self.min_confidence = min_confidence
        self._rules: list[_Rule] = [
            _Rule(
                rule_id=r.get("rule_id", ""),
                cluster_id=int(r.get("cluster_id", -1)),
                label=r.get("label", ""),
                predicate=r.get("predicate") or {},
                source_model=r.get("source_model") or "",
                target_model=r.get("target_model") or "",
                confidence=r.get("confidence") or "low",
                estimated_savings_per_call_usd=float(r.get("estimated_savings_per_call_usd") or 0.0),
            )
            for r in (policy.get("rules") or [])
        ]
        # Sort by (confidence desc, savings desc). Higher-confidence rules win
        # ties when multiple match.
        self._rules.sort(
            key=lambda r: (CONFIDENCE_RANK.get(r.confidence, -1), r.estimated_savings_per_call_usd),
            reverse=True,
        )
        self._min_rank = CONFIDENCE_RANK.get(min_confidence, 1)
        self._decisions: list[dict] = []

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "Router":
        return cls(load_policy(path), **kwargs)

    def route(
        self,
        *,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, str | None]:
        """Decide which model to call based on the session signature.

        Returns ``(target_model, rule_id_or_None)``. Pass-through if no rule
        matches or if every matching rule is below the ``min_confidence`` bar.
        """
        sig = self._signature(messages or [], tools or [])
        for rule in self._rules:
            if CONFIDENCE_RANK.get(rule.confidence, -1) < self._min_rank:
                continue
            if not rule.target_model or rule.target_model == rule.source_model:
                continue
            if rule.target_model.startswith("(no"):
                continue
            if self._predicate_matches(rule.predicate, sig):
                return rule.target_model, rule.rule_id
        return self.default_model, None

    def record(
        self,
        rule_id: str | None,
        response: Any,
        *,
        ok: bool | None = None,
        note: str | None = None,
    ) -> None:
        """Log a routing decision's outcome. No-op by default — override or
        subclass to ship these to your own telemetry.

        ``ok`` can be ``True`` (cheap model returned a useful tool call),
        ``False`` (response was nonsense and you had to re-call on the big
        model), or ``None`` (not evaluated). Used by the flywheel refit to
        downgrade rules that misbehaved.
        """
        self._decisions.append({
            "rule_id": rule_id,
            "ok": ok,
            "note": note,
        })

    def decisions(self) -> list[dict]:
        return list(self._decisions)

    # ------------------------------------------------------------------ #
    # Signature extraction + predicate matching
    # ------------------------------------------------------------------ #

    def _signature(self, messages: list[dict], tools: list[dict]) -> dict:
        """Extract the routing-relevant signals from the pending request.

        Today: the last tool call observed in the message history (usually
        the tool name the assistant used most recently). Sessions that have
        been grinding on ``skill_manage`` are likely about to do another
        ``skill_manage`` — that's the re-routing heuristic.
        """
        last_tool = None
        recent_tools: list[str] = []
        for m in reversed(messages):
            if m.get("role") != "assistant":
                continue
            tool_calls = m.get("tool_calls")
            if not tool_calls:
                continue
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name")
                if name:
                    recent_tools.append(name)
            if recent_tools:
                last_tool = recent_tools[-1]
                break
        return {
            "last_tool_name": last_tool,
            "recent_tool_names": recent_tools[-5:],
            "tool_schema_names": [
                (t.get("function") or {}).get("name") or t.get("name")
                for t in tools or []
                if isinstance(t, dict)
            ],
        }

    def _predicate_matches(self, predicate: dict, sig: dict) -> bool:
        ptype = predicate.get("type")
        if ptype == "last_tool_call_name_in":
            names = set(predicate.get("tool_names") or [])
            last = sig.get("last_tool_name")
            return bool(last) and last in names
        return False


def describe_policy(policy: dict) -> str:
    """Human-readable summary for eyeballing a fitted policy before deploying it."""
    lines = []
    lines.append(f"tracerCC routing policy v{policy.get('policy_version','?')}  "
                 f"fitted {policy.get('fitted_at','?')}")
    cs = policy.get("corpus_summary") or {}
    lines.append(
        f"  fit corpus: {cs.get('sessions','?')} sessions, "
        f"{cs.get('assistant_turns','?')} assistant turns, "
        f"{cs.get('span_days','?')} days, "
        f"${cs.get('premium_spend_usd','?')} priced"
    )
    lines.append(f"  default model: {policy.get('default_model')}")
    lines.append(f"  gate: {policy.get('gate','structural')} "
                 f"(reasoning threshold: {policy.get('reasoning_threshold_chars', 0)} chars)")
    lines.append(f"  rules ({len(policy.get('rules') or [])}):")
    for r in policy.get("rules") or []:
        lines.append(
            f"    {r['rule_id']}  [{r['confidence']}]  n={r['n_training_turns']}  "
            f"→ {r['target_model']}  "
            f"(~${r['estimated_savings_per_call_usd']:.4f}/call, "
            f"${r['estimated_window_savings_usd']:.2f} projected)"
        )
        lines.append(f"        medoid: {r.get('medoid_example','')[:100]}")
    return "\n".join(lines)
