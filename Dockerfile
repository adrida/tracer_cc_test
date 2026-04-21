# tracerCC reference backend — Cloud Run image.
#
# Build context: the repo root (Cloud Run `gcloud run deploy --source .`
# picks this Dockerfile up automatically).
#
# Build:  docker build -t tracercc-backend .
# Run:    docker run -p 8080:8080 \
#           -e CLOUDFLARE_ACCOUNT_ID=... \
#           -e CLOUDFLARE_AUTH_TOKEN=... \
#           tracercc-backend
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the package source and install with [serve] extras (fastapi + uvicorn
# + aiohttp + pydantic + sklearn + slowapi + dotenv + torch + sentence-transformers).
# We skip [full] — UMAP+HDBSCAN is optional and pulls numba/llvmlite which
# balloons the cold-start time on Cloud Run.
COPY pyproject.toml README.md ./
COPY tracercc ./tracercc
RUN pip install '.[serve]'

# Cloud Run injects PORT (default 8080).
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn tracercc.backend.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
