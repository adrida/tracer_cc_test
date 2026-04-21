"""HTTP client for the tracerCC analysis backend.

Sends a redacted payload to ``POST /v1/analyze`` and returns the
``WrappedReport`` as a ``types.SimpleNamespace`` tree (so the existing
Jinja template can keep using attribute access on ``r.foo``).
"""

from __future__ import annotations

import gzip
import json
import os
import time
from types import SimpleNamespace
from typing import Any

import requests

from . import __version__


DEFAULT_BACKEND_URL = os.environ.get(
    "TRACERCC_BACKEND_URL",
    "https://tracercc-backend-331966625866.us-central1.run.app",
)
DEFAULT_BACKEND_TOKEN = os.environ.get("TRACERCC_BACKEND_TOKEN", "")

REQUEST_TIMEOUT_S = 600  # the analysis can take a couple of minutes for big corpora
GZIP_THRESHOLD_BYTES = 32 * 1024  # gzip if the body is bigger than 32 KB


class BackendError(RuntimeError):
    pass


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def analyze(
    payload: dict,
    source: str,
    backend_url: str = DEFAULT_BACKEND_URL,
    backend_token: str = DEFAULT_BACKEND_TOKEN,
    targets: list[str] | None = None,
    progress: bool = True,
    reasoning_threshold_chars: int = 0,
) -> SimpleNamespace:
    """POST a redacted Tables payload to the backend, return a WrappedReport namespace."""
    options: dict = {
        "targets": targets or ["claude-sonnet-4-6", "claude-haiku-4-5", "composer-2"],
    }
    if reasoning_threshold_chars:
        options["reasoning_threshold_chars"] = int(reasoning_threshold_chars)
    body = {
        "schema_version": "1.0",
        "source": source,
        "client_version": __version__,
        "redacted_prompts": True,
        "data": payload,
        "options": options,
    }
    raw = json.dumps(body, default=str).encode("utf-8")

    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": f"tracercc/{__version__}"}
    if backend_token:
        headers["Authorization"] = f"Bearer {backend_token}"

    if len(raw) > GZIP_THRESHOLD_BYTES:
        raw = gzip.compress(raw)
        headers["Content-Encoding"] = "gzip"

    url = backend_url.rstrip("/") + "/v1/analyze"
    if progress:
        size_kb = len(raw) / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        print(f"  → POST {url}  ({size_str}{'  gzip' if 'Content-Encoding' in headers else ''})")
    t0 = time.time()
    try:
        r = requests.post(url, data=raw, headers=headers, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        raise BackendError(f"backend unreachable at {url}: {e}") from e
    elapsed = time.time() - t0
    if r.status_code != 200:
        raise BackendError(
            f"backend returned {r.status_code} after {elapsed:.1f}s: {r.text[:400]}"
        )
    if progress:
        print(f"  ← 200 in {elapsed:.1f}s")
    return _to_namespace(r.json())


def health(backend_url: str = DEFAULT_BACKEND_URL, timeout: float = 10.0) -> dict:
    """Cheap healthcheck so the CLI can fail fast with a useful message.

    GCloud's frontend intercepts ``/healthz`` on Cloud Run, so we use
    ``/v1/health`` (the local backend exposes both for convenience).
    """
    base = backend_url.rstrip("/")
    last_err: Exception | None = None
    for path in ("/v1/health", "/healthz"):
        try:
            r = requests.get(base + path, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = BackendError(f"{path} returned {r.status_code}")
        except requests.RequestException as e:
            last_err = e
    raise BackendError(f"backend healthcheck failed at {base}: {last_err}")
