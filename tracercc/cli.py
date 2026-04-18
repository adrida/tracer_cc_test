"""tracerCC CLI entry point.

Run against your local Claude Code or Cursor data and get a Spotify-Wrapped-
style HTML dashboard out the other side. The heavy analysis (pricing,
clustering, counterfactuals) runs on the hosted tracerCC backend; this
client only extracts, redacts, posts, and renders.

    $ tracercc                              # autodetect source
    $ tracercc --source cursor              # force one
    $ tracercc --json report.json           # also dump the raw report
    $ tracercc --no-open                    # don't pop a browser
    $ TRACERCC_BACKEND_URL=http://… tracercc   # self-host the backend
"""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tracercc",
        description="Tracer · Spotify-Wrapped-style waste diagnostic for Claude Code & Cursor.",
    )
    p.add_argument("--source", choices=sorted(SOURCES), default=None,
                   help="Where to read sessions from. Default: autodetect.")
    p.add_argument("--out", type=Path, default=Path.cwd() / "tracer_wrapped.html",
                   help="Where to write the HTML dashboard. Default: ./tracer_wrapped.html")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to also dump the raw report as JSON.")
    p.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                   help=f"tracerCC analysis backend URL. Default: {DEFAULT_BACKEND_URL} "
                        "(override via TRACERCC_BACKEND_URL env var).")
    p.add_argument("--backend-token", default=DEFAULT_BACKEND_TOKEN,
                   help="Optional bearer token for a self-hosted backend "
                        "(or set TRACERCC_BACKEND_TOKEN).")
    p.add_argument("--no-open", action="store_true",
                   help="Do not auto-open the dashboard in a browser.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    p.add_argument("--version", action="version", version=f"tracerCC {__version__}")
    args = p.parse_args(argv)

    progress = not args.quiet
    if progress:
        _banner()

    source = args.source or autodetect()
    if not source:
        print("\n  \033[31m✗\033[0m no Claude Code (~/.claude/projects) "
              "or Cursor (~/.cursor/projects) data found.\n"
              "    Run something in either tool first, then re-run tracercc.",
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
        # Quick liveness probe so we fail fast with a useful error
        try:
            h = health(args.backend_url)
            if not h.get("cf_configured"):
                print("  \033[33m!\033[0m backend reports cf_configured=false; analysis will fail.")
        except BackendError as e:
            print(f"\n  \033[31m✗\033[0m {e}\n"
                  "    If you're running your own backend, check it's reachable:\n"
                  f"      curl {args.backend_url.rstrip('/')}/healthz",
                  file=sys.stderr)
            return 3

    try:
        report = analyze(
            payload,
            source=source,
            backend_url=args.backend_url,
            backend_token=args.backend_token,
            progress=progress,
        )
    except BackendError as e:
        print(f"\n  \033[31m✗\033[0m {e}", file=sys.stderr)
        return 3

    out_path = render_html(report, args.out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_namespace_to_json(report))

    if progress:
        elapsed = time.time() - t0
        print()
        print(f"  \033[32m✓\033[0m dashboard ready in {elapsed:.1f}s → {out_path}")
        if args.json:
            print(f"  \033[32m✓\033[0m raw report          → {args.json}")
        try:
            print()
            print(f"  \033[1mheadline\033[0m  ${report.saving_haiku_usd:,.0f} saveable on your "
                  f"${report.opus_spend_usd:,.0f} Opus spend ({report.saving_haiku_pct_of_opus:.1f}%)")
            print(f"  \033[2m         {report.n_clusters} clustered patterns of mechanical work · "
                  f"{report.n_noise} excluded · backend={report.cluster_backend}\033[0m")
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
    """SimpleNamespace tree → pretty JSON string."""
    from types import SimpleNamespace

    def _convert(o):
        if isinstance(o, SimpleNamespace):
            return {k: _convert(v) for k, v in vars(o).items()}
        if isinstance(o, list):
            return [_convert(v) for v in o]
        return o

    return json.dumps(_convert(obj), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
