# tracerCC — your AI coding, wrapped

> One command. Reads your local Claude Code, Cursor, or custom-agent sessions.
> Returns a Spotify-Wrapped-style HTML showing how much your model choice cost
> you, with a per-cluster receipt so you can audit every dollar.

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

  headline  $15 saveable on your $24 premium-tier spend (60.1%)
            2 clustered patterns of mechanical work · 0 excluded · backend=lite
```

---

## Install & run

Not yet on PyPI — install straight from the git repo:

```bash
pip install 'tracercc @ git+https://github.com/adrida/tracer_cc_test.git'
tracercc
```

Or with `uv` (faster, no venv management needed):

```bash
uv tool install 'tracercc @ git+https://github.com/adrida/tracer_cc_test.git'
tracercc
```

Or one-shot via `uvx` (no install at all — runs from cache):

```bash
uvx --from 'git+https://github.com/adrida/tracer_cc_test.git' tracercc
```

The client parses your session data locally, redacts prompt + response text,
posts the redacted metadata to the hosted analysis backend, and renders a
self-contained HTML in your browser. No extras, no torch, no local compute —
all heavy lifting runs on our infra (Cloud Run + Cloudflare Workers AI) for
the fastest possible first click.

One-liner install (installs `uv` if needed):

```bash
curl -fsSL https://raw.githubusercontent.com/adrida/tracer_cc_test/main/install.sh | bash
```

---

## What gets sent off your machine — and what doesn't

| Leaves your machine                                                | Stays on your machine            |
| ------------------------------------------------------------------ | -------------------------------- |
| Per-message token counts (input/output/cache)                      | **Your prompt text**             |
| Model name + timestamp                                              | **Assistant response text**      |
| Tool call names + a 200-char preview of the tool input             | **Thinking / reasoning blocks**  |
| Per-session metadata (uuid, cwd, message count)                     | **File contents read by tools**  |

The backend is stateless: it embeds the tool-call previews on Cloudflare
Workers AI (BGE-M3, 1024-dim), clusters them, computes counterfactual costs
against your actual token usage, and returns the report. Nothing is logged or
persisted. The backend source is in `tracercc/backend/` for full transparency.

---

## Sources

| Source        | Auto-detected when                                             | Notes                                                                                  |
| ------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `claude_code` | `~/.claude/projects/` exists                                   | Reads JSONL session logs.                                                              |
| `cursor`      | `~/Library/Application Support/Cursor` (or `.config/Cursor/`)  | Reads agent transcripts + the `ai-code-tracking.db` SQLite file.                       |
| `custom`      | `./traces/custom/*.jsonl` exists                               | Any agent framework. See `tracercc/sources/custom.py` for the event schema.            |

Force one with `--source claude_code|cursor|custom`.

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
TRACERCC_BACKEND_URL=https://my.backend.example   # point at a self-hosted backend
TRACERCC_BACKEND_TOKEN=…                          # bearer token if backend requires it
```

---

## How the saveable number is computed

For every assistant turn, we price it at its actual model's rate using a
pricing table covering Anthropic + OpenAI + Google + xAI + Moonshot + Mistral
+ DeepSeek + Cursor. A turn enters the "mechanical" pool iff its output is
**only** tool calls (no reasoning text, no thinking blocks) AND all tool names
are in a generous whitelist (Bash / Read / Grep / Edit / TodoWrite /
skill_manage / memory / read_file / ...). Each mechanical turn is embedded by
its first tool-call text; density clustering groups near-identical operations.
**Only turns inside dense clusters are re-priced** at the cheapest sibling in
the same model family (gpt-5-4 → gpt-5-4-mini, claude-opus → claude-haiku,
gemini-pro → gemini-flash). Sparse / noisy turns are excluded — visible in the
dashboard's trust gauge. Result: a strictly smaller number than naive
arithmetic, but defensible per dollar.

The structural gate (what this backend runs today) guarantees *evidence of
repetition*. A future behavioural gate will add *evidence of agreement on
held-out inputs* — mirror of the classification parity-gate in
[`adrida/tracer`](https://github.com/adrida/tracer).

---

## Self-hosting the backend

The backend source is in `tracercc/backend/`. To run your own instance:

```bash
pip install 'tracercc[serve]'
tracercc serve --host 0.0.0.0 --port 8080

# From another machine (or the same one):
TRACERCC_BACKEND_URL=http://your-host:8080 tracercc
```

Docker / Cloud Run:

```bash
docker build -t tracercc-backend .
docker run -p 8080:8080 \
  -e CLOUDFLARE_ACCOUNT_ID=… \
  -e CLOUDFLARE_AUTH_TOKEN=… \
  tracercc-backend

# Or deploy to Cloud Run with the bundled script (requires gcloud auth):
./deploy.sh
```

Without Cloudflare creds, the backend falls back to a sklearn HashingVectorizer
embedder — coarser than BGE-M3 but good enough for the structural gate.

---

## Local development

```bash
git clone https://github.com/adrida/tracer_cc_test.git
cd tracer_cc_test
uv pip install -e '.[serve]'
tracercc serve --port 8080          # in one terminal
TRACERCC_BACKEND_URL=http://localhost:8080 tracercc --no-open   # in another
```

## License

MIT.
