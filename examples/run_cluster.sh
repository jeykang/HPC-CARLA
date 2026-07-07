#!/usr/bin/env bash
# =============================================================================
# run_cluster.sh — one-file launcher for running HPC-CARLA collection on YOUR
# cluster. Purpose: reproduce the collection on different GPU hardware (e.g.
# L40S) so results can be compared against the A100 baseline.
#
#   ./examples/run_cluster.sh setup     # check prerequisites (SIF, weights, configs)
#   ./examples/run_cluster.sh smoke     # tiny ~1h validation run  <-- DO THIS FIRST
#   ./examples/run_cluster.sh run       # the real comparison run
#   ./examples/run_cluster.sh monitor   # watch progress (from the login node)
#   ./examples/run_cluster.sh summary   # print this cluster's comparison metrics
#   ./examples/run_cluster.sh export    # write a labeled results bundle to share
#
# `start --slurm` submits the SLURM job for you — do NOT `sbatch` this file.
# See examples/CLUSTER_SETUP.md for the full walkthrough.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # repo root

# ╔══════════════════ EDIT THIS BLOCK FOR YOUR CLUSTER ══════════════════╗
CLUSTER_NAME="${CLUSTER_NAME:-l40s-example}"   # label for your results bundle, e.g. "l40s-siteX"
SLURM_NODES="${SLURM_NODES:-1}"                # number of nodes to acquire
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}" # GPUs/node: one CARLA server + one agent each
SLURM_TIME="${SLURM_TIME:-48:00:00}"           # walltime
SLURM_PARTITION="${SLURM_PARTITION:-}"         # e.g. "gpu"  ("" = cluster default)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"             # e.g. "myproj" ("" = not required)
SLURM_QOS="${SLURM_QOS:-}"                     # e.g. "normal" ("" = skip)
SLURM_NODELIST="${SLURM_NODELIST:-}"           # pin nodes e.g. "gpu[01-02]" ("" = SLURM picks)
CARLA_SIF="${CARLA_SIF:-$PWD/carla_official.sif}"  # path to the ~6 GB CARLA image
JOB_TIMEOUT_SEC="${JOB_TIMEOUT_SEC:-3600}"     # per-job cap. NOTE: a route FILE is a SUITE of many
                                               # routes run in sequence (checkpointed per route); a
                                               # stable GPU banks routes until this cap, then cycles.

# What to collect. KEEP THESE EQUAL ACROSS CLUSTERS for a fair comparison.
AGENTS=(tcp interfuser cilrs neat roach)       # A100 baseline set (LAV excluded — see note below)
ROUTES_KIND="${ROUTES_KIND:-short_tiny}"       # short_tiny | short | tiny | all  (short_tiny = A100 baseline)
# ╚══════════════════════════════════════════════════════════════════════╝
#
# NOTE on LAV: LAV is excluded from the A100 baseline because it crashes the
# CARLA server on A100 (a GPU-raster/driver issue — see PAPER_REFERENCE §9). On
# L40S (RTX-class rasteriser) it may well run — if you want that comparison data
# point, add `lav` to AGENTS. Just note the sets then differ across clusters.

export CARLA_SIF PROJECT_ROOT="$PWD"

_slurm_args() {
  local a=(--slurm-nodes "$SLURM_NODES" --slurm-gpus "$SLURM_GPUS_PER_NODE"
           --slurm-time "$SLURM_TIME" --job-timeout "$JOB_TIMEOUT_SEC" --sif "$CARLA_SIF")
  [ -n "$SLURM_PARTITION" ] && a+=(--slurm-partition "$SLURM_PARTITION")
  [ -n "$SLURM_ACCOUNT"   ] && a+=(--slurm-account   "$SLURM_ACCOUNT")
  [ -n "$SLURM_QOS"       ] && a+=(--slurm-qos       "$SLURM_QOS")
  [ -n "$SLURM_NODELIST"  ] && a+=(--slurm-nodelist  "$SLURM_NODELIST")
  printf '%s\n' "${a[@]}"
}
_routes() {   # emit the route-file basenames for ROUTES_KIND (empty => all)
  case "$ROUTES_KIND" in
    short_tiny) find leaderboard/data \( -name 'routes_*_short.xml' -o -name 'routes_*_tiny.xml' \) ;;
    short)      find leaderboard/data -name 'routes_*_short.xml' ;;
    tiny)       find leaderboard/data -name 'routes_*_tiny.xml' ;;
    all)        : ;;
    *) echo "unknown ROUTES_KIND=$ROUTES_KIND" >&2; exit 2 ;;
  esac 2>/dev/null | sed 's#.*/##' | sort -u | tr '\n' ' '
}

case "${1:-help}" in
  setup)
    echo "== prerequisite check for '$CLUSTER_NAME' =="
    command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 \
      && echo "  [ok] singularity/apptainer found" || echo "  [MISSING] singularity/apptainer"
    command -v squeue >/dev/null 2>&1 && echo "  [ok] SLURM (squeue) found" || echo "  [MISSING] SLURM"
    if [ -f "$CARLA_SIF" ]; then echo "  [ok] CARLA image: $CARLA_SIF"
    else echo "  [MISSING] CARLA image at $CARLA_SIF — build it: singularity build carla_official.sif carla_official_u2004.def"; fi
    echo "  -- agent weights (fetched from public sources if absent) --"
    for ag in "${AGENTS[@]}"; do
      w=$(ls leaderboard/team_code/"$ag"/*.pth leaderboard/team_code/"$ag"/*.ckpt \
             leaderboard/team_code/"$ag"/weights/* 2>/dev/null | head -1)
      if [ -n "$w" ]; then echo "    [ok]   $ag ($(basename "$w"))"
      elif [ -x leaderboard/team_code/"$ag"/fetch_weights.sh ]; then
        echo "    [fetch] $ag: running fetch_weights.sh"; bash leaderboard/team_code/"$ag"/fetch_weights.sh || echo "      (fetch failed — see script)"
      else echo "    [??]   $ag: no weights and no fetch_weights.sh"; fi
    done
    echo "  -- validating agent pipeline configs --"
    python3 continuous_cli.py validate-config --all
    echo "== setup check done. Next: ./examples/run_cluster.sh smoke =="
    ;;

  smoke)
    mapfile -t SA < <(_slurm_args)
    echo "== SMOKE on '$CLUSTER_NAME' ($SLURM_NODES node × $SLURM_GPUS_PER_NODE GPU) =="
    python3 continuous_cli.py --persistent start --slurm --reset --smoke \
      --agents "${AGENTS[@]}" "${SA[@]}"
    echo "watch with: ./examples/run_cluster.sh monitor"
    ;;

  run)
    mapfile -t SA < <(_slurm_args)
    R="$(_routes)"
    echo "== RUN on '$CLUSTER_NAME': agents=[${AGENTS[*]}] routes=$ROUTES_KIND nodes=$SLURM_NODES gpus/node=$SLURM_GPUS_PER_NODE =="
    # shellcheck disable=SC2086
    python3 continuous_cli.py --persistent start --slurm --reset \
      --agents "${AGENTS[@]}" ${R:+--routes $R} "${SA[@]}"
    echo "watch with: ./examples/run_cluster.sh monitor"
    ;;

  monitor)
    python3 continuous_cli.py monitor --persistent
    ;;

  summary)
    echo "== '$CLUSTER_NAME' comparison metrics =="
    python3 continuous_cli.py summary --persistent 2>/dev/null || true
    echo "-- per-ROUTE evals harvested (the real metric: routes done, incl. from crashed/timed-out files) --"
    python3 tools/harvest_results.py 2>/dev/null || echo "   (tools/harvest_results.py not present)"
    echo "-- per-agent score & pass-rate --";      python3 tools/verification_report.py 2>/dev/null || true
    echo "-- outcome taxonomy (agent-result vs infra) --"; python3 tools/classify_outcomes.py 2>/dev/null || true
    echo "-- sim-vs-realtime speed --";             python3 tools/parse_sim_ratio.py 2>/dev/null || true
    echo "-- difficulty vs performance --";         python3 tools/difficulty_validation.py 2>/dev/null || echo "   (tools/difficulty_validation.py not present)"
    ;;

  export)
    OUT="results/${CLUSTER_NAME}"; mkdir -p "$OUT"
    echo "== exporting labeled bundle to $OUT/ =="
    python3 continuous_cli.py export --output "$OUT/collection_results.json" 2>/dev/null || true
    # per-ROUTE harvest is the primary comparison artifact (-> $OUT/per_route_results.csv)
    python3 tools/harvest_results.py --out-dir "$OUT" > "$OUT/harvest_summary.txt" 2>&1 || true
    python3 tools/verification_report.py   > "$OUT/verification_report.txt"   2>&1 || true
    python3 tools/classify_outcomes.py     > "$OUT/outcomes.txt"              2>&1 || true
    python3 tools/parse_sim_ratio.py       > "$OUT/sim_ratio.txt"             2>&1 || true
    python3 tools/difficulty_validation.py --per-route "$OUT/per_route_results.csv" > "$OUT/difficulty_validation.md" 2>&1 \
      || python3 tools/difficulty_validation.py > "$OUT/difficulty_validation.md" 2>&1 || true
    # hardware/run metadata for the comparison
    { echo "cluster_name: $CLUSTER_NAME"
      echo "nodes: $SLURM_NODES   gpus_per_node: $SLURM_GPUS_PER_NODE"
      echo "agents: ${AGENTS[*]}"
      echo "routes_kind: $ROUTES_KIND   job_timeout_sec: $JOB_TIMEOUT_SEC"
      echo "gpu_model:"; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort -u | sed 's/^/  /' || echo "  (nvidia-smi unavailable on login node — fill in your GPU model)"
    } > "$OUT/cluster_meta.txt"
    echo "bundle ready: $OUT/  (share this directory for A100-vs-$CLUSTER_NAME comparison)"
    ;;

  *)
    sed -n '2,18p' "$0"   # print the header/usage
    ;;
esac
