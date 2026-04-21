# tracerCC reference backend — containerised self-host.
#
# Build:  docker build -t tracercc-backend tracercc/backend
# Run:    docker run -p 8080:8080 tracercc-backend
#         # optional Cloudflare BGE-M3 upgrade:
#         docker run -p 8080:8080 \
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

# Install the `serve` extras from PyPI (or from source if you pin to a git rev).
RUN pip install 'tracercc[serve]'

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn tracercc.backend.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
