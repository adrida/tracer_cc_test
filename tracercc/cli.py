"""tracerCC CLI entry point.

Run against your local Claude Code / Cursor / custom-agent data and get a
Spotify-Wrapped-style HTML dashboard out the other side.

The client is intentionally thin: parse sessions on your machine, redact raw
prompt + response text, POST the numeric+metadata payload to the hosted
analysis backend. Pricing, clustering, and counterfactuals all run on our
infra so users get the fastest startup and the latest rate tables without
having to upgrade a local install.

The reference backend source lives in ``tracercc/backend/`` for transparency
and is self-hostable (see Dockerfile + deploy.sh at the repo root). For most
users, just:

    $ tracercc                         # autodetect, use hosted backend
    $ tracercc --source cursor         # force source
    $ tracercc --json report.json
    $ tracercc --no-open
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__
from .api_client import (
    BackendError,
    DEFAULT_BACKEND_TOKEN,
    DEFAULT_BACKEND_URL,
    analyze,
    health,
)
from .redact import redact_tables
from .render import render_html
from .sources import SOURCES, autodetect, load


def _banner() -> None:
    print()
    print("  ╭───────────────────────────────────────────────╮")
    print("  │  \033[1mtracerCC\033[0m · your AI-coding wrapped           │")
    print("  ╰───────────────────────────────────────────────╯")
    print()


def cmd_serve(argv: list[str]) -> int:
    """Spin up the reference backend locally via uvicorn."""
    ap = argparse.ArgumentParser(prog="tracercc serve", description="Run the tracerCC analysis backend locally.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reload", action="store_true", help="Auto-reload on source changes.")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("\033[31m✗\033[0m uvicorn not installed. Install with: pip install 'tracercc[serve]'",
              file=sys.stderr)
        return 2
    try:
        import tracercc.backend.main  # noqa: F401
    except ImportError:
        print("\033[31m✗\033[0m backend not installed. Install with: pip install 'tracercc[serve]' or 'tracercc[local]'",
              file=sys.stderr)
        return 2

    print(f"  tracerCC backend serving on http://{args.host}:{args.port}")
    print(f"  point the client at it:  TRACERCC_BACKEND_URL=http://{args.host}:{args.port} tracercc")
    uvicorn.run("tracercc.backend.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Subcommand dispatch: ``tracercc serve ...`` routes to cmd_serve.
    if argv and argv[0] == "serve":
        return cmd_serve(argv[1:])

    p = argparse.ArgumentParser(
        prog="tracercc",
        description="Tracer · waste diagnostic for Claude Code, Cursor, and custom agents.",
    )
    p.add_argument("--source", choices=sorted(SOURCES), default=None,
                   help="Where to read sessions from. Default: autodetect.")
    p.add_argument("--out", type=Path, default=Path.cwd() / "tracer_wrapped.html",
                   help="Where to write the HTML dashboard. Default: ./tracer_wrapped.html")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to also dump the raw report as JSON.")
    p.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                   help=f"Analysis backend URL. Default: {DEFAULT_BACKEND_URL}")
    p.add_argument("--backend-token", default=DEFAULT_BACKEND_TOKEN,
                   help="Optional bearer token for a self-hosted backend.")
    p.add_argument("--no-open", action="store_true",
                   help="Do not auto-open the dashboard in a browser.")
    p.add_argument("--reasoning-threshold", type=int, default=0,
                   help=("Max chars of reasoning/thinking text an assistant turn can "
                         "carry and still be counted as mechanical. Default 0 (strict "
                         "tracer gate). Raise to e.g. 1000 for agents like GPT-5.2 that "
                         "auto-emit short chain-of-thought on every call."))
    p.add_argument("--policy-out", type=Path, default=None,
                   help=("Write the fitted routing policy to this JSON path. Defaults to "
                         "./tracer_policy.json next to --out. Drop into your agent's "
                         "OpenAI client via tracercc.runtime.Router.from_file(path)."))
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    p.add_argument("--version", action="version", version=f"tracerCC {__version__}")
    args = p.parse_args(argv)

    progress = not args.quiet
    if progress:
        _banner()

    source = args.source or autodetect()
    if not source:
        print("\n  \033[31m✗\033[0m no Claude Code (~/.claude/projects), "
              "Cursor (~/.cursor/projects), or custom (./traces/custom) data found.\n"
              "    Run something in either tool first, or drop a .jsonl into "
              "./traces/custom/, then re-run tracercc.",
              file=sys.stderr)
        return 2
    if progress:
        print(f"  source:           \033[1m{source}\033[0m")
        print(f"  backend:          {args.backend_url}")

    if progress:
        print()
        print(f"→ extracting {source} sessions ...")
    t0 = time.time()
    tables = load(source)
    if len(tables.sessions) == 0:
        print(f"\n  \033[31m✗\033[0m no sessions found for {source}.", file=sys.stderr)
        return 2
    if progress:
        print(f"  → {len(tables.sessions)} sessions, {len(tables.messages):,} messages, "
              f"{len(tables.prompts):,} prompts in {time.time()-t0:.1f}s")

    if progress:
        print("→ redacting prompt + response text (only tokens + tool summaries leave your machine)")
    payload = redact_tables(tables).to_dict()

    if progress:
        print("→ requesting analysis from backend ...")
        try:
            h = health(args.backend_url)
            if not h.get("cf_configured"):
                print("  \033[33m!\033[0m backend reports cf_configured=false; "
                      "embeddings will use the backend's hashing fallback.")
        except BackendError as e:
            print(f"\n  \033[31m✗\033[0m {e}\n"
                  f"    Backend may be cold-starting; retry in ~10s.\n"
                  f"    If self-hosted, verify: curl {args.backend_url.rstrip('/')}/v1/health",
                  file=sys.stderr)
            return 3
    try:
        report = analyze(
            payload,
            source=source,
            backend_url=args.backend_url,
            backend_token=args.backend_token,
            progress=progress,
            reasoning_threshold_chars=args.reasoning_threshold,
        )
    except BackendError as e:
        print(f"\n  \033[31m✗\033[0m {e}", file=sys.stderr)
        return 3

    out_path = render_html(report, args.out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_namespace_to_json(report))

    # Serialise the fitted routing policy to disk — this is the actionable
    # output Amine drops into his Hermes OpenAI client at runtime.
    policy_out = args.policy_out or args.out.with_name("tracer_policy.json")
    policy_ns = getattr(report, "routing_policy", None)
    policy_written: Path | None = None
    n_rules = 0
    if policy_ns is not None:
        from types import SimpleNamespace

        def _to_plain(o):
            if isinstance(o, SimpleNamespace):
                return {k: _to_plain(v) for k, v in vars(o).items()}
            if isinstance(o, list):
                return [_to_plain(v) for v in o]
            if isinstance(o, dict):
                return {k: _to_plain(v) for k, v in o.items()}
            return o

        policy_plain = _to_plain(policy_ns)
        if policy_plain.get("rules"):
            policy_out.parent.mkdir(parents=True, exist_ok=True)
            policy_out.write_text(json.dumps(policy_plain, indent=2, default=str))
            policy_written = policy_out
            n_rules = len(policy_plain["rules"])

    if progress:
        elapsed = time.time() - t0
        print()
        print(f"  \033[32m✓\033[0m dashboard ready in {elapsed:.1f}s → {out_path}")
        if args.json:
            print(f"  \033[32m✓\033[0m raw report          → {args.json}")
        if policy_written:
            print(f"  \033[32m✓\033[0m routing policy      → {policy_written}  ({n_rules} rules)")
        try:
            print()
            # Prefer the family-aware headline when populated; fall back to the
            # legacy Anthropic-specific one for old backends / renderers.
            save = getattr(report, "saving_cheapest_sibling_usd", 0.0) or getattr(report, "saving_haiku_usd", 0.0)
            premium = getattr(report, "premium_spend_usd", 0.0) or getattr(report, "opus_spend_usd", 0.0)
            pct = getattr(report, "saving_cheapest_sibling_pct_of_premium", 0.0) or getattr(report, "saving_haiku_pct_of_opus", 0.0)
            print(f"  \033[1mheadline\033[0m  ${save:,.0f} saveable on your "
                  f"${premium:,.0f} premium-tier spend ({pct:.1f}%)")
            print(f"  \033[2m         {report.n_clusters} clustered patterns of mechanical work · "
                  f"{report.n_noise} excluded · backend={report.cluster_backend}\033[0m")
            total = getattr(report, "total_spend_usd", 0) or 0
            print(f"  \033[2m         covered ${total:,.2f} of priced activity across "
                  f"{report.n_messages:,} messages. Provider dashboards may show a bigger number —\n"
                  f"  \033[2m         tracerCC prices only what's in the traces you handed it.\033[0m")
        except AttributeError:
            pass
        print()

    if not args.no_open:
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            pass

    return 0


def _namespace_to_json(obj) -> str:
    from types import SimpleNamespace

    def _convert(o):
        if isinstance(o, SimpleNamespace):
            return {k: _convert(v) for k, v in vars(o).items()}
        if isinstance(o, list):
            return [_convert(v) for v in o]
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        return o

    return json.dumps(_convert(obj), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
