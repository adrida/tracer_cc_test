"""FastAPI entry point for the tracerCC analysis backend (open reference).

Endpoints:
    GET  /            — service banner
    GET  /v1/health   — liveness + embedding-backend presence (no values leaked)
    POST /v1/analyze  — accept a redacted Tables payload, return a WrappedReport

Environment variables:
    CLOUDFLARE_ACCOUNT_ID    — optional, enables BGE-M3 embeddings
    CLOUDFLARE_AUTH_TOKEN    — optional (or CLOUDFLARE_API_TOKEN)
    TRACERCC_API_TOKEN       — optional shared bearer token
    TRACERCC_RATE_PER_MIN    — per-IP rate limit, default 30
    TRACERCC_MAX_MESSAGES    — max messages per analyze request, default 200000
    PORT                     — default 8080

Run locally: ``tracercc serve`` or ``uvicorn tracercc.backend.main:app --port 8080``
"""

import gzip
import logging
import os
import time
import uuid as _uuid
from typing import Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    _HAVE_SLOWAPI = True
except ImportError:
    _HAVE_SLOWAPI = False

from . import __version__
from .analyze import analyze
from .embedding import credentials_ok, _pick_backend
from .pricing import PRICING_SOURCE
from .schema import AnalyzeRequest, WrappedReport


SHARED_TOKEN = os.environ.get("TRACERCC_API_TOKEN", "")
RATE_PER_MIN = int(os.environ.get("TRACERCC_RATE_PER_MIN", "30"))
MAX_MESSAGES = int(os.environ.get("TRACERCC_MAX_MESSAGES", "200000"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tracercc.backend")


app = FastAPI(
    title="tracerCC analysis backend",
    version=__version__,
    description="Pricing + clustering + counterfactuals for Claude Code, Cursor, and custom agents.",
)

if _HAVE_SLOWAPI:
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_PER_MIN}/minute"])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)


class GzipDecodeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-encoding", "").lower() == "gzip":
            raw = await request.body()
            try:
                decoded = gzip.decompress(raw)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": "bad_gzip", "detail": str(e)})
            request._body = decoded  # type: ignore[attr-defined]

            async def _replay() -> dict:
                return {"type": "http.request", "body": decoded, "more_body": False}
            request._receive = _replay  # type: ignore[attr-defined]
            mutable_headers = [
                (k, v) for (k, v) in request.scope["headers"]
                if k.lower() not in (b"content-encoding", b"content-length")
            ]
            mutable_headers.append((b"content-length", str(len(decoded)).encode()))
            request.scope["headers"] = mutable_headers
        return await call_next(request)


app.add_middleware(GzipDecodeMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _check_auth(authorization: Optional[str]) -> None:
    if not SHARED_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization.split(" ", 1)[1] != SHARED_TOKEN:
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/")
async def root():
    return {
        "name": "tracerCC analysis backend",
        "version": __version__,
        "endpoints": ["/", "/v1/health", "/v1/analyze"],
        "rate_limit_per_min": RATE_PER_MIN,
        "max_messages_per_request": MAX_MESSAGES,
        "auth_required": bool(SHARED_TOKEN),
        "embedding_backend": _pick_backend(),
        "gate": "structural",  # behavioural gate is hosted-only, not in this reference impl
        "pricing_source": PRICING_SOURCE,
    }


@app.get("/healthz")
@app.get("/v1/health")
async def healthz():
    return {
        "status": "ok",
        "cf_configured": credentials_ok(),
        "embedding_backend": _pick_backend(),
        "version": __version__,
    }


@app.post("/v1/analyze", response_model=WrappedReport)
async def analyze_endpoint(
    request: Request,
    body: AnalyzeRequest = Body(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    if len(body.data.messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"too many messages: {len(body.data.messages)} > {MAX_MESSAGES}",
        )

    rid = _uuid.uuid4().hex[:8]
    t0 = time.time()
    log.info(
        "analyze rid=%s source=%s n_sessions=%d n_messages=%d n_tool_calls=%d "
        "client_version=%s redacted=%s",
        rid, body.source, len(body.data.sessions), len(body.data.messages),
        len(body.data.tool_calls), body.client_version, body.redacted_prompts,
    )
    try:
        report = await analyze(body)
    except RuntimeError as e:
        log.warning("analyze rid=%s failed: %s", rid, e)
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.time() - t0
    log.info(
        "analyze rid=%s done in %.1fs total_spend=$%.2f premium=$%.2f n_clusters=%d "
        "save_sibling=$%.2f cluster_backend=%s",
        rid, elapsed, report.total_spend_usd, report.premium_spend_usd,
        report.n_clusters, report.saving_cheapest_sibling_usd, report.cluster_backend,
    )
    return report


if _HAVE_SLOWAPI:
    @app.exception_handler(RateLimitExceeded)
    async def _ratelimit_handler(request: Request, exc: RateLimitExceeded):  # type: ignore[misc]
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "detail": str(exc.detail)},
        )


@app.exception_handler(ValidationError)
async def _validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )
