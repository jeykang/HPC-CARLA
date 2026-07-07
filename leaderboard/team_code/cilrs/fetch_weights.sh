#!/usr/bin/env bash
# Fetch the pretrained CILRS weights (best_model.pth) for the modular pipeline.
#
# Source: autonomousvision/transfuser pretrained model bundle (public S3, ~697 MB
# zip containing all baselines; we extract only cilrs/best_model.pth, ~48 MB).
# Referenced in the repo README (cvpr2021 branch):
#   https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models.zip
#
# The download is a direct S3 object (no Google Drive), so it works through the
# HPC login-node proxy (HTTPS_PROXY). This script is idempotent: it skips the
# download if best_model.pth is already present.
#
# NOTE: at the time this agent ran, best_model.pth was already fetched and placed
# next to this script — this file documents provenance / allows re-fetching.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/best_model.pth"
URL="https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models.zip"

if [ -f "${DEST}" ]; then
    echo "[fetch_weights] ${DEST} already exists ($(du -h "${DEST}" | cut -f1)); nothing to do."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "[fetch_weights] downloading models bundle (~697 MB) via proxy ..."
# curl honors HTTPS_PROXY from the environment. -L follows redirects.
curl -fSL --retry 3 -o "${TMP}/models.zip" "${URL}"

echo "[fetch_weights] extracting cilrs/best_model.pth ..."
# -j: flatten (drop the leading cilrs/ dir) so the file lands directly here.
unzip -o -j "${TMP}/models.zip" 'cilrs/best_model.pth' -d "${HERE}"

echo "[fetch_weights] done -> ${DEST} ($(du -h "${DEST}" | cut -f1))"
