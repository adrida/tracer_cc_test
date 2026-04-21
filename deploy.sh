#!/usr/bin/env bash
# Deploy the tracerCC analysis backend to GCloud Cloud Run.
#
# Prereqs (one-time):
#   - gcloud CLI authed:        gcloud auth login
#   - Project selected:         gcloud config set project <PROJECT>
#   - APIs enabled:             gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
#   - Two secrets created (Cloudflare Workers AI):
#       * tracercc-cf-account-id  → your Cloudflare account id
#       * tracercc-cf-auth-token  → Cloudflare API token with Workers AI scope
#
# Usage:
#   ./deploy.sh                       # defaults
#   REGION=us-central1 SERVICE=tracercc-backend ./deploy.sh

set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-tracercc-backend}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
CONCURRENCY="${CONCURRENCY:-20}"
TIMEOUT="${TIMEOUT:-300s}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: no GCloud project selected. Run: gcloud config set project <PROJECT>" >&2
  exit 1
fi

echo "→ Project:      $PROJECT"
echo "→ Region:       $REGION"
echo "→ Service:      $SERVICE"
echo "→ Min/Max:      $MIN_INSTANCES / $MAX_INSTANCES instances"
echo "→ Resources:    $CPU CPU, $MEMORY"
echo "→ Concurrency:  $CONCURRENCY"
echo

EXTRA_FLAGS=()
if [[ "$ALLOW_UNAUTH" == "true" ]]; then
  EXTRA_FLAGS+=(--allow-unauthenticated)
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source . \
  --platform managed \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --port 8080 \
  --set-secrets "CLOUDFLARE_ACCOUNT_ID=tracercc-cf-account-id:latest,CLOUDFLARE_AUTH_TOKEN=tracercc-cf-auth-token:latest" \
  --set-env-vars "TRACERCC_RATE_PER_MIN=30,TRACERCC_MAX_MESSAGES=200000" \
  "${EXTRA_FLAGS[@]}"

echo
echo "→ done. Service URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
    --format="value(status.url)"
