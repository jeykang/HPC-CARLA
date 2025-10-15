#!/usr/bin/env bash
# Emits per-GPU health JSONs at: collection_state/health/<node>/gpu<N>.json
# Works alongside your existing persistent worker. Safe to run in background.
#
# Env (all optional with sane defaults):
#   PROJECT_ROOT       : repo root (defaults to script's directory parent)
#   STATE_DIR          : "$PROJECT_ROOT/collection_state"
#   LOG_DIR            : "$PROJECT_ROOT/logs"
#   GPU_ID             : REQUIRED for correct port calc (tries to infer, else 0)
#   BASE_RPC_PORT      : 2000   | PORT_SPACING : 100   | TM_OFFSET : 5000
#   HEARTBEAT_SECS     : 10     | STALE_SECS   : 90
#   STATUS             : "idle" | MESSAGE      : ""    | JOBS_COMPLETED : 0
#
# It also *reads* collection_state/gpu_status.json (v2 schema) on each tick
# to enrich "status/message/jobs_completed" when available for this (node,gpu).

set -euo pipefail

# --- resolve paths/env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/collection_state}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$STATE_DIR/health" "$LOG_DIR"

# Node & GPU id
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s || hostname)}"

# Try to infer GPU_ID if not set (best-effort; prefer explicit GPU_ID)
if [[ -z "${GPU_ID:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Take first entry from CUDA_VISIBLE_DEVICES; this might be a logical map.
    GPU_ID="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | head -n1)"
  else
    GPU_ID="0"
  fi
fi

# Ports
BASE_RPC_PORT="${BASE_RPC_PORT:-2000}"
PORT_SPACING="${PORT_SPACING:-100}"
TM_OFFSET="${TM_OFFSET:-5000}"
RPC_PORT=$(( BASE_RPC_PORT + GPU_ID * PORT_SPACING ))
TM_PORT=$(( RPC_PORT + TM_OFFSET ))

HEARTBEAT_SECS="${HEARTBEAT_SECS:-10}"
STALE_SECS="${STALE_SECS:-90}"

STATUS="${STATUS:-idle}"
MESSAGE="${MESSAGE:-}"
JOBS_COMPLETED="${JOBS_COMPLETED:-0}"

HEALTH_NODE_DIR="$STATE_DIR/health/$NODE_NAME"
HEALTH_FILE="$HEALTH_NODE_DIR/gpu${GPU_ID}.json"
mkdir -p "$HEALTH_NODE_DIR"

# --- helpers ---

# Pull per-GPU info from v2 gpu_status.json (if present) and override fields.
read_v2_status() {
  local v2="$STATE_DIR/gpu_status.json"
  [[ -f "$v2" ]] || return 0
  python3 - "$NODE_NAME" "$GPU_ID" "$v2" <<'PY' || true
import json, sys, time, datetime
node = sys.argv[1]
gpu  = sys.argv[2]
path = sys.argv[3]
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

nodes = data.get("nodes", {})
gmap = nodes.get(node, {})
info = gmap.get(str(gpu))
if not isinstance(info, dict):
    sys.exit(0)

status = info.get("status")
cj = info.get("current_job") or {}
# Build compact message
parts=[]
if cj.get("agent") or cj.get("agent_name"):
    parts.append(str(cj.get("agent") or cj.get("agent_name")))
town = cj.get("town") or cj.get("map") or cj.get("town_num")
if town: parts.append(f"T{town}")
route = cj.get("route") or cj.get("route_id") or cj.get("route_name")
if route: parts.append(str(route))
w = cj.get("weather") or cj.get("weather_idx") or cj.get("weather_index")
if w not in (None, ""): parts.append(f"W{w}")
message = " / ".join(parts)

jobs_completed = info.get("jobs_completed", 0)
hb = info.get("last_heartbeat")
ts_unix = None
if isinstance(hb, (int, float)):
    ts_unix = float(hb)
elif isinstance(hb, str):
    try:
        s = hb.replace("T", " ").split(".")[0]
        ts_unix = datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        ts_unix = None

# Emit as shell exports to override outer vars
if status:
    print(f"STATUS_OVR={json.dumps(status)}")
if message:
    print(f"MESSAGE_OVR={json.dumps(message)}")
print(f"JOBS_OVR={int(jobs_completed)}")
if ts_unix:
    print(f"TS_OVR={ts_unix}")
PY
}

# Atomic JSON write via python (prevents partial reads)
write_health_json() {
  local tmp="${HEALTH_FILE}.tmp.$$"
  local now_unix
  now_unix="$(date +%s)"

  # Try enrich from v2
  local status_ovr="" msg_ovr="" jobs_ovr="" ts_ovr=""
  while IFS='=' read -r k v; do
    case "$k" in
      STATUS_OVR) status_ovr="$v" ;;
      MESSAGE_OVR) msg_ovr="$v" ;;
      JOBS_OVR) jobs_ovr="$v" ;;
      TS_OVR) ts_ovr="$v" ;;
    esac
  done < <(read_v2_status || true)

  local status="$STATUS"
  local message="$MESSAGE"
  local jobs="$JOBS_COMPLETED"
  [[ -n "$status_ovr" ]] && status="$(python3 - <<PY 2>/dev/null || echo "$status"
import json,sys; print(json.loads(sys.stdin.read()))
PY
<<<"$status_ovr")"
  [[ -n "$msg_ovr" ]] && message="$(python3 - <<PY 2>/dev/null || echo "$message"
import json,sys; print(json.loads(sys.stdin.read()))
PY
<<<"$msg_ovr")"
  [[ -n "$jobs_ovr" ]] && jobs="$jobs_ovr"
  local ts_use="$now_unix"
  [[ -n "$ts_ovr" ]] && ts_use="$ts_ovr"

  python3 - "$tmp" <<PY
import json, sys, time, datetime
path = sys.argv[1]
payload = {
  "gpu_id": int(${GPU_ID}),
  "node": ${NODE_NAME!r},
  "status": ${status!r},
  "message": ${message!r},
  "jobs_completed": int(${jobs}),
  "rpc_port": int(${RPC_PORT}),
  "tm_port": int(${TM_PORT}),
  "timestamp": datetime.datetime.fromtimestamp(float(${ts_use})).strftime("%Y-%m-%d %H:%M:%S"),
  "timestamp_unix": float(${ts_use}),
}
with open(path, "w") as f:
    json.dump(payload, f, ensure_ascii=False)
PY
  mv -f "$tmp" "$HEALTH_FILE"
}

# --- loop ---

echo "[healthbeat] node=${NODE_NAME} gpu=${GPU_ID} -> $HEALTH_FILE (every ${HEARTBEAT_SECS}s)"
while true; do
  write_health_json
  sleep "$HEARTBEAT_SECS"
done
