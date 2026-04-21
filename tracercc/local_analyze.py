"""In-process analysis — runs the backend pipeline without any HTTP round-trip.

Used when the CLI is invoked with ``--local`` (the default when the backend
extras are installed). This is what makes ``pip install tracercc[local]``
a zero-network experience.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from . import __version__


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def analyze_local(payload: dict, source: str, targets: list[str] | None = None) -> SimpleNamespace:
    """Run analysis in-process. Requires the ``[local]`` extras installed."""
    try:
        from .backend.analyze import analyze
        from .backend.schema import AnalyzeRequest
    except ImportError as e:
        raise RuntimeError(
            "local analysis requires the [local] extras. Install with: "
            "pip install 'tracercc[local]'"
        ) from e

    body = AnalyzeRequest(
        schema_version="1.0",
        source=source,
        client_version=__version__,
        redacted_prompts=True,
        data=payload,
        options={"targets": targets} if targets else {},
    )
    report = asyncio.run(analyze(body))
    return _to_namespace(report.model_dump())
