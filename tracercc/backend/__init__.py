"""tracerCC analysis backend — open reference implementation.

This subpackage is the **open-source structural gate**: it consumes a redacted
Tables payload (tokens + tool metadata, no prompt/response text) and returns a
WrappedReport with per-model spend, mechanical-turn detection, clustering of
repeated mechanical patterns, and counterfactual re-pricing against cheaper
siblings in the same model family.

Runs two ways:
    1. In-process, via ``tracercc.local_analyze`` — zero network, default.
    2. As a FastAPI service, via ``tracercc serve`` — same logic, HTTP surface.

What this module intentionally does NOT contain:

* **Behavioural parity gate** — the replay-on-cheaper-model + measure-agreement
  step that upgrades the structural gate to a safety-proving one. This is the
  Tracer thesis's real contribution to agentic routing (mirror of the classification
  parity gate that already exists in ``adrida/tracer``). It lives server-side on the
  hosted tracerCC backend; this reference implementation stops at the structural
  pre-gate by design.
* **Cross-user pattern library** — patterns learned across many tenants are not
  part of a single-machine diagnostic.

If you need either of those, point the client at the hosted backend (default) or
use the ``--hosted`` flag.
"""

__version__ = "0.3.0"
