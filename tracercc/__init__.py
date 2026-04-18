"""tracerCC — Spotify-Wrapped-style waste diagnostic for Claude Code & Cursor.

The public client is intentionally thin: it extracts your local Claude Code
or Cursor session data, redacts every prompt and assistant response, sends
the resulting per-message tokens + tool-call summaries to the tracerCC
analysis backend, and renders the returned report as a self-contained HTML
dashboard you can open in your browser.

Usage:
    $ tracercc
    $ tracercc --source cursor
    $ tracercc --source claude-code --json report.json
    $ TRACERCC_BACKEND_URL=http://localhost:8766 tracercc

What leaves your machine:
    * Per-message token counts + model name + timestamps
    * Tool-call names + truncated tool-call inputs
      (e.g. "Bash: git status", "Read: /path", "Grep: pattern=...")
    * Per-session metadata (cwd, message counts)

What never leaves your machine:
    * User prompt text
    * Assistant response text
    * Thinking blocks
    * File contents read by tools
"""

__version__ = "0.2.0"
