#!/usr/bin/env bash
# Fetch pretrained NEAT weights (best_encoder.pth, best_decoder.pth, args.txt)
# into this directory (leaderboard/team_code/neat/), i.e. container path
#   /workspace/leaderboard/team_code/neat/
#
# Source: official NEAT model bundle from autonomousvision/neat README:
#   https://s3.eu-central-1.amazonaws.com/avg-projects/neat/models.zip
# The bundle (~818 MB) contains all 5 pretrained agents; we extract ONLY the
# `neat/` sub-folder and discard the rest.
#
# Honors HTTP(S)_PROXY from the environment (curl/wget pick it up automatically).
# Usage:  bash fetch_weights.sh
set -euo pipefail

DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="https://s3.eu-central-1.amazonaws.com/avg-projects/neat/models.zip"
TMPDIR="$(mktemp -d)"
ZIP="${TMPDIR}/models.zip"

cleanup() { rm -rf "${TMPDIR}"; }
trap cleanup EXIT

echo "[neat] downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --continue-at - -o "${ZIP}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${ZIP}" "${URL}"
else
    echo "[neat] ERROR: need curl or wget" >&2
    exit 1
fi

echo "[neat] extracting neat/* -> ${DST}"
# Extract only the neat/ entries, flatten into DST.
unzip -o -j "${ZIP}" "neat/best_encoder.pth" "neat/best_decoder.pth" "neat/args.txt" -d "${DST}"

echo "[neat] done:"
ls -la "${DST}/best_encoder.pth" "${DST}/best_decoder.pth" "${DST}/args.txt"
