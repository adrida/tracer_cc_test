# tracerCC — your AI coding, wrapped

> One command. Reads your local Claude Code / Cursor / custom-agent sessions.
> Returns a Spotify-Wrapped-style HTML showing how much your model choice cost
> you, with a per-cluster receipt so you can audit every dollar.

```
  ╭───────────────────────────────────────────────╮
  │  tracerCC · your AI-coding wrapped            │
  ╰───────────────────────────────────────────────╯

  source:           cursor   (auto-detected)
  backend:          in-process (local)

→ extracting cursor sessions ...
  → 42 sessions, 6,777 messages, 1,218 prompts in 0.7s
→ redacting prompt + response text (only tokens + tool summaries leave your machine)
→ running analysis in-process ...

  ✓ dashboard ready in 9.8s → ./tracer_wrapped.html

  headline  $15 saveable on your $24 premium-tier spend (60.1%)
            2 clustered patterns of mechanical work · 0 excluded · backend=kmeans
```

…and a self-contained `tracer_wrapped.html` opens in your browser. Share it.
Print it. Tweet it.

---

## Open-core layout

tracerCC is split into two parts with a sharp line between them:

| Layer | What's in it | Where it lives |
|---|---|---|
| **Client** | Extract from `~/.claude/projects/`, `~/.cursor/`, or local JSONL. Redact prompt + response text. Render HTML. | This repo, `tracercc/` |
| **Structural gate (open reference backend)** | Per-token pricing across Anthropic, OpenAI (incl. dot-named `gpt-5.2-…`), Google, xAI, Moonshot, Mistral, DeepSeek, Cursor. Mechanical-turn detection. Embedding (local MiniLM / optional BGE-M3). Density clustering. Family-aware counterfactual re-pricing. | This repo, `tracercc/backend/` |
| **Behavioural parity gate** | Replay-on-cheaper-model + measure-agreement. The real Tracer thesis ported to agents. Mirrors the classification parity gate in [`adrida/tracer`](https://github.com/adrida/tracer). | **Hosted only** — not in this repo. |
| **Cross-user pattern library** | Patterns observed across many tenants feed back into safer routing recommendations. | **Hosted only** — not in this repo. |

The structural gate alone produces a defensible savings number. The behavioural
gate upgrades "clustering says these 75 skill-manage calls are near-identical"
to "…and we verified gpt-5-4-mini emits the same thing on held-out inputs." The
latter is the tracerCC roadmap's moat, not a gate for the first click.

---

## Install

Four tiers, pick by how much you want to download on first run:

| Extras | What you get | First-run DL | Cluster quality |
|---|---|---|---|
| `tracercc` | Client only, uses hosted backend | ~15 MB | depends on hosted |
| `tracercc[local]` | Local pipeline, sklearn hashing embedder + sklearn HDBSCAN | ~15 MB | good |
| `tracercc[embed]` | [local] + sentence-transformers MiniLM (better clusters) | +500 MB torch, +90 MB model | very good |
| `tracercc[full]` | [embed] + UMAP + HDBSCAN heavy clusterer | +100 MB numba/llvmlite | best (>1k turns) |
| `tracercc[serve]` | [embed] + FastAPI for self-hosting | +500 MB | — |

Recommended: **`pip install 'tracercc[local]'`** — fastest start, no torch, no downloads beyond
the sklearn install. The hashing embedder is noticeably coarser than MiniLM in theory, but the
structural gate (mechanical-turn filter → density clustering) is robust enough that the
resulting clusters are indistinguishable in practice on single-developer corpora.

```bash
pip install 'tracercc[local]'
tracercc                          # autodetect, open browser — ~10s total

# If your clusters look underdelineated, upgrade to MiniLM:
pip install 'tracercc[embed]'
tracercc                          # same command, now uses sentence-transformers

# Self-host for a team:
pip install 'tracercc[serve]'
tracercc serve --port 8080
TRACERCC_BACKEND_URL=http://localhost:8080 tracercc --hosted
```

One-liner (installs `uv` if needed):

```bash
curl -fsSL https://raw.githubusercontent.com/adrida/tracer_cc_test/main/install.sh | bash
```

---

## What gets sent off your machine — and what doesn't

| Leaves your machine (only with `--hosted`)                       | Stays on your machine           |
| ---------------------------------------------------------------- | ------------------------------- |
| Per-message token counts (input/output/cache)                    | **Your prompt text**            |
| Model name + timestamp                                           | **Assistant response text**     |
| Tool call names + a 200-char preview of the tool input           | **Thinking / reasoning blocks** |
| Per-session metadata (uuid, cwd, message count)                  | **File contents read by tools** |

With `tracercc[local]` (the default), nothing leaves the machine at all —
analysis runs in-process and embeddings use a 384-dim `all-MiniLM-L6-v2`
downloaded once into `~/.cache/huggingface`.

---

## Sources

| Source        | Auto-detected when                                             | Notes                                                                                  |
| ------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `claude_code` | `~/.claude/projects/` exists                                   | Reads JSONL session logs.                                                              |
| `cursor`      | `~/Library/Application Support/Cursor` (or `.config/Cursor/`)  | Reads agent transcripts + the `ai-code-tracking.db` SQLite file.                       |
| `custom`      | `./traces/custom/*.jsonl` exists                               | Any agent framework. See `tracercc/sources/custom.py` docstring for the event schema.  |

Force one with `--source claude_code|cursor|custom`.

---

## Common flags

```bash
tracercc                              # autodetect, local analysis if extras installed
tracercc --local                      # force in-process analysis (needs [local] extras)
tracercc --hosted                     # force the hosted backend even if [local] is available
tracercc --source cursor              # force the source
tracercc --no-open                    # don't pop the browser
tracercc --out ./report.html          # custom HTML path
tracercc --json ./report.json         # also dump the raw report JSON
tracercc serve --port 8080            # spin up the reference backend locally
```

Env overrides:

```bash
TRACERCC_BACKEND_URL=https://my.backend.example   # self-hosted or custom backend
TRACERCC_BACKEND_TOKEN=…                          # bearer token if backend requires it
TRACERCC_FORCE_LOCAL_EMBED=1                      # skip cloudflare even if creds are set
```

---

## How the saveable number is computed

> For every assistant turn, we price it at its actual model's rate using the
> bundled pricing table (Anthropic + OpenAI + Google + xAI + Moonshot + Mistral
> + DeepSeek + Cursor). A turn enters the "mechanical" pool iff its output is
> **only** tool calls (no reasoning text, no thinking blocks) AND all tool
> names are in a generous whitelist (Bash / Read / Grep / Edit / TodoWrite /
> skill_manage / memory / read_file / ...). Each mechanical turn is embedded
> by its first tool-call text; density clustering groups near-identical
> operations. **Only turns inside dense clusters are re-priced** at the
> cheapest sibling in the same model family (gpt-5-4 → gpt-5-4-mini,
> claude-opus → claude-haiku, gemini-pro → gemini-flash). Sparse / noisy turns
> are excluded — visible in the dashboard's trust gauge. Result: a strictly
> smaller number than naive arithmetic, but defensible per dollar.

The structural gate (what this repo ships) guarantees *evidence of repetition*.
The behavioural gate (hosted) adds *evidence of agreement on held-out inputs* —
that's the upgrade path that mirrors the classification Tracer's parity-gate.

---

## Self-hosting the reference backend

```bash
pip install 'tracercc[serve]'
tracercc serve --host 0.0.0.0 --port 8080

# From another machine (or the same one):
TRACERCC_BACKEND_URL=http://your-host:8080 tracercc --hosted
```

Docker / Cloud Run:

```bash
docker build -t tracercc-backend tracercc/backend
docker run -p 8080:8080 tracercc-backend
# or
gcloud run deploy tracercc-backend --source tracercc/backend/
```

With no Cloudflare creds the backend uses `sentence-transformers/all-MiniLM-L6-v2`
automatically. Set `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_AUTH_TOKEN` to upgrade
to BGE-M3 (1024-dim, sharper clusters).

---

## Local development

```bash
git clone https://github.com/adrida/tracer_cc_test.git
cd tracer_cc_test
uv pip install -e '.[local]'
tracercc --local --no-open
```

## License

MIT.
