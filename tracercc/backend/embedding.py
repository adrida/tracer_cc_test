"""Embedding layer — three backends in preference order:

    1. Local sentence-transformers (``all-MiniLM-L6-v2`` by default) — ships
       in-process, no network, ~15 MB of weights cached in HF hub. This is the
       default when you run ``tracercc`` without a Cloudflare account.

    2. Cloudflare Workers AI BGE-M3 (1024-dim) — higher quality, requires
       ``CLOUDFLARE_ACCOUNT_ID`` + ``CLOUDFLARE_AUTH_TOKEN``. Used automatically
       when those env vars are set.

    3. sklearn ``HashingVectorizer`` + L2 norm — ultimate fallback when
       sentence-transformers isn't installed and Cloudflare isn't configured.
       Rough but always works (~10 KB of code, zero downloads).

The clustering layer only needs pairwise cosine distances, so (1) and (3) are
both "good enough" for the structural gate. (2) is what the hosted backend uses
because BGE-M3 outperforms MiniLM on tool-call-text grouping, but it's a
quality upgrade, not a dependency.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np


log = logging.getLogger("tracercc.embedding")


CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_AUTH_TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN", "") or os.environ.get(
    "CLOUDFLARE_API_TOKEN", ""
)
CF_BGE_MODEL = os.environ.get("TRACERCC_BGE_MODEL", "@cf/baai/bge-m3")

LOCAL_MODEL_NAME = os.environ.get("TRACERCC_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FORCE_LOCAL = os.environ.get("TRACERCC_FORCE_LOCAL_EMBED") == "1"

BATCH_SIZE = 96
MAX_RETRIES = 4
TIMEOUT_S = 60


def credentials_ok() -> bool:
    """True iff Cloudflare creds are configured (BGE-M3 path available).

    Kept for backwards-compat with the hosted backend's healthcheck.
    """
    return bool(CF_ACCOUNT_ID and CF_AUTH_TOKEN)


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #

def _pick_backend() -> str:
    """Return ``"local"``, ``"cloudflare"``, or ``"hashing"``."""
    if FORCE_LOCAL:
        if _have("sentence_transformers"):
            return "local"
        return "hashing"
    if credentials_ok():
        return "cloudflare"
    if _have("sentence_transformers"):
        return "local"
    return "hashing"


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


async def embed_texts_async(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Picks the best available backend."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    backend = _pick_backend()
    log.info("embedding %d texts via %s", len(texts), backend)
    if backend == "cloudflare":
        try:
            return await _cf_embed(texts)
        except Exception as e:
            log.warning("cloudflare embed failed (%s); falling back to local", e)
    if backend == "local" or (backend == "cloudflare"):  # cloudflare may have fallen through
        if _have("sentence_transformers"):
            return _st_embed(texts)
    return _hashing_embed(texts)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Synchronous wrapper."""
    return asyncio.run(embed_texts_async(texts))


# --------------------------------------------------------------------------- #
# Cloudflare Workers AI BGE-M3 (hosted-backend default)
# --------------------------------------------------------------------------- #

async def _cf_embed(texts: list[str]) -> np.ndarray:
    import aiohttp  # local import so lite install doesn't require aiohttp
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/"
        f"{CF_BGE_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {CF_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    out_parts: list[np.ndarray] = []
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S * 4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            vecs = await _cf_batch(session, url, headers, batch)
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"cloudflare returned {len(vecs)} vectors for {len(batch)} inputs"
                )
            out_parts.append(np.asarray(vecs, dtype=np.float32))
    return np.concatenate(out_parts, axis=0) if out_parts else np.zeros((0, 1024), dtype=np.float32)


async def _cf_batch(session, url, headers, texts: list[str]) -> list[list[float]]:
    import aiohttp
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json={"text": texts}) as r:
                body = await r.json()
                if r.status == 200 and body.get("success"):
                    result = body.get("result", {})
                    vecs = result.get("data") or result.get("embedding")
                    if not isinstance(vecs, list):
                        raise RuntimeError(f"unexpected upstream payload: {body}")
                    return vecs
                if r.status in (429, 502, 503, 504):
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"cloudflare error {r.status}: {str(body)[:300]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"cloudflare unreachable: {last_err}")


# --------------------------------------------------------------------------- #
# Local sentence-transformers (default for `pip install tracercc`)
# --------------------------------------------------------------------------- #

_ST_CACHE = {}

def _st_embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if LOCAL_MODEL_NAME not in _ST_CACHE:
        log.info("loading sentence-transformers model %s", LOCAL_MODEL_NAME)
        _ST_CACHE[LOCAL_MODEL_NAME] = SentenceTransformer(LOCAL_MODEL_NAME)
    model = _ST_CACHE[LOCAL_MODEL_NAME]
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pure-sklearn fallback — always works
# --------------------------------------------------------------------------- #

def _hashing_embed(texts: list[str]) -> np.ndarray:
    """HashingVectorizer + L2 norm. 512 dims, zero downloads, always works.

    Coarser than BGE-M3 or MiniLM but the clustering layer only needs pairwise
    cosine to detect near-identical tool calls, which hashing does fine.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    vec = HashingVectorizer(
        n_features=512, alternate_sign=False, norm=None,
        analyzer="char_wb", ngram_range=(3, 5),
    )
    X = vec.transform(texts).astype(np.float32).toarray()
    return normalize(X, norm="l2", axis=1)
