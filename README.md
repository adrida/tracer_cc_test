# tracerCC — your AI coding, wrapped

> One command. Reads your local Claude Code sessions (`~/.claude/projects/`) or
> Cursor history. Returns a Spotify-Wrapped-style HTML showing exactly how much
> money your model choice cost you, with a per-cluster receipt so you can audit
> every dollar of the claim.

```
  ╭───────────────────────────────────────────────╮
  │  tracerCC · your AI-coding wrapped            │
  ╰───────────────────────────────────────────────╯

  source:           cursor   (auto-detected)
  backend:          https://tracercc-backend-…run.app

→ extracting cursor sessions ...
  → 42 sessions, 6,777 messages, 1,218 prompts in 0.7s
→ redacting prompt + response text (only tokens + tool summaries leave your machine)
→ requesting analysis from backend ...
  → POST … (340 KB gzip)  ← 200 in 8.5s

  ✓ dashboard ready in 9.8s → ./tracer_wrapped.html

  headline  $9 saveable on your $442 Opus spend (2.0%)
            3 clustered patterns of mechanical work · 0 excluded · backend=kmeans
```

…and a self-contained `tracer_wrapped.html` opens in your browser. Share it.
Print it. Tweet it.

---

## One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/adrida/tracer_cc_test/main/install.sh | bash
```

That's it. The script installs [`uv`](https://docs.astral.sh/uv/) if you don't
already have it, then runs `tracercc` via `uvx`. No global install, no venv to
manage, no credentials to configure, no Docker.

If you already have `uv`:

```bash
uvx --python 3.11 --from git+https://github.com/adrida/tracer_cc_test.git tracercc
```

Or with `pipx`:

```bash
pipx run --spec git+https://github.com/adrida/tracer_cc_test.git tracercc
```

---

## What gets sent off your machine — and what doesn't

| Leaves your machine                                                | Stays on your machine            |
| ------------------------------------------------------------------ | -------------------------------- |
| Per-message token counts (input/output/cache)                      | **Your prompt text**             |
| Model name + timestamp                                              | **Assistant response text**      |
| Tool call names + a 200-char preview of the tool input             | **Thinking / reasoning blocks**  |
| Per-session metadata (uuid, cwd, message count)                     | **File contents read by tools**  |

The hosted analysis backend is stateless: it embeds the tool-call previews,
clusters them, computes counterfactual costs against your actual token usage,
and returns the report. Nothing is logged or persisted. Source for the backend
is private; source for this client is right here.

If you want a fully self-hosted setup, see *Self-hosting* below.

---

## Sources

| Source        | Auto-detected when                                             | Notes                                                                                  |
| ------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `claude_code` | `~/.claude/projects/` exists                                   | Reads JSONL session logs.                                                              |
| `cursor`      | `~/Library/Application Support/Cursor` (or `.config/Cursor/`)  | Reads agent transcripts + the `ai-code-tracking.db` SQLite file.                       |

Force one with `--source claude_code` or `--source cursor`.

---

## Common flags

```bash
tracercc                              # autodetect, open browser
tracercc --source cursor              # force the source
tracercc --no-open                    # don't pop the browser
tracercc --out ./report.html          # custom HTML path
tracercc --json ./report.json         # also dump the raw report JSON
```

Env overrides:

```bash
TRACERCC_BACKEND_URL=https://my.backend.example   # self-hosted analysis backend
TRACERCC_BACKEND_TOKEN=…                          # bearer token if backend requires it
```

---

## How the saveable number is computed

One paragraph:

> We extract every assistant turn, identify the ones whose *only* output is a
> mechanical tool call (`Bash` / `Read` / `Glob` / `Grep` / `Edit` /
> `TodoWrite`), normalise the tool input into a short text (`Bash: ls -la
> /path`), embed it with BGE-M3, and cluster (PCA + sklearn-HDBSCAN, falling
> back to silhouette-tuned KMeans on small corpora). Only turns inside dense
> clusters are re-priced at smaller-model rates. Sparse / noisy turns are
> excluded — visible in the dashboard's trust gauge. The result is strictly
> smaller than naive arithmetic but defensible per dollar: if 327 commands are
> all near-identical to one medoid, by definition a smaller model can emit any
> one of them.

Pricing comes from Anthropic's published per-token rates, Cursor's
[models-and-pricing docs](https://cursor.com/docs/models-and-pricing) for
Auto / Composer-2 / Max-mode upcharges, and per-provider rates for everything
in Cursor's API pool.

---

## Self-hosting the backend

If you don't trust a hosted backend (or you want to run this on a corporate
network), the `tracerCC_backend/` folder in the parent repo ships a single
FastAPI service deployable to GCloud Cloud Run, Fly, Render, or any container
host. It needs a Cloudflare Workers AI account (free tier handles thousands of
runs).

```bash
# in tracerCC_backend/
gcloud run deploy tracercc-backend --source . \
  --set-secrets CLOUDFLARE_ACCOUNT_ID=…:latest,CLOUDFLARE_AUTH_TOKEN=…:latest
TRACERCC_BACKEND_URL="https://your-service.run.app" tracercc
```

---

## Local development

```bash
git clone https://github.com/adrida/tracer_cc_test.git
cd tracer_cc_test
uv pip install -e .
tracercc --no-open
```

## License

MIT.
