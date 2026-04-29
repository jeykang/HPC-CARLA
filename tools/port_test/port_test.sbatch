#!/bin/bash
# Toy RPC-port-binding diagnostic. Submits to one node, runs three independent
# probes to localize where TCP listeners fail (or don't):
#
#   probe A : Python (inside Singularity) binds a port and self-connects.
#             -> tests container -> listener path; CARLA-independent.
#
#   probe B : nc on host binds a port; Python (inside Singularity) connects.
#             -> tests host->container loopback path.
#
#   probe C : python -c http.server (inside container) bound + curl from host.
#             -> tests container->host visibility (is listener reachable from
#                outside the container's network namespace?).
#
# All probes use ephemeral ports; nothing collides with CARLA's 2000-range.
# Pass --nodelist on submit (e.g. sbatch -w hpc-pr-a-pod09 port_test.sbatch).

#SBATCH --job-name=port_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=port_test_%A_%N.out
#SBATCH --error=port_test_%A_%N.err

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/autodr_test/HPC-CARLA-persistent}"
SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
NODE="${SLURMD_NODENAME:-$(hostname)}"

# Random ephemeral ports in the 50000-59999 range (low collision risk)
PORT_A=$(( 50000 + RANDOM % 1000 ))
PORT_B=$(( 51000 + RANDOM % 1000 ))
PORT_C=$(( 52000 + RANDOM % 1000 ))

echo "=========================================="
echo "RPC port binding diagnostic"
echo "Node     : ${NODE}"
echo "SIF      : ${SIF}"
echo "Ports    : A=${PORT_A} B=${PORT_B} C=${PORT_C}"
echo "Date     : $(date -Is)"
echo "=========================================="

# Sanity: can we even reach nvidia-smi inside the container?
echo
echo "--- nvidia-smi inside container ---"
singularity exec --nv "${SIF}" bash -lc 'nvidia-smi -L || echo NVIDIA_SMI_FAILED' 2>&1 | head -10

echo
echo "--- /dev/dri inside container ---"
singularity exec --nv "${SIF}" bash -lc 'ls -la /dev/dri 2>&1 | head -5; echo; ldconfig -p 2>/dev/null | grep -iE "vulkan|GLX|EGL" | head -5'

echo
echo "############### PROBE A : container-internal bind+connect ###############"
singularity exec --nv "${SIF}" python3 - "${PORT_A}" <<'PY'
import socket, sys, threading, time
port = int(sys.argv[1])
ok = {"server": False, "client": False, "err": None}

def srv():
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        ok["server"] = True
        c, _ = s.accept()
        c.send(b"hello\n"); c.close(); s.close()
    except Exception as e:
        ok["err"] = f"server: {e!r}"

t = threading.Thread(target=srv, daemon=True); t.start()
time.sleep(1.0)
try:
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    data = c.recv(64); c.close()
    ok["client"] = (data == b"hello\n")
except Exception as e:
    ok["err"] = f"client: {e!r}"
t.join(timeout=3)
print(f"PROBE_A: server_bind={ok['server']} client_recv_ok={ok['client']} err={ok['err']}")
PY
echo "(probe A done)"

echo
echo "############### PROBE B : host-side bind, container connect ###############"
# Listen on host with nc; have container connect.
nc -l -p "${PORT_B}" -q 2 >/dev/null 2>&1 &
NC_PID=$!
sleep 1
if ! kill -0 "${NC_PID}" 2>/dev/null; then
    echo "PROBE_B: HOST_NC_FAILED_TO_BIND (no nc, busy port, or policy)"
else
    singularity exec "${SIF}" python3 - "${PORT_B}" <<'PY'
import socket, sys
port = int(sys.argv[1])
try:
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.send(b"ping\n"); c.close()
    print(f"PROBE_B: container_connect_to_host_ok=True port={port}")
except Exception as e:
    print(f"PROBE_B: container_connect_to_host_ok=False err={e!r} port={port}")
PY
    wait "${NC_PID}" 2>/dev/null || true
fi
echo "(probe B done)"

echo
echo "############### PROBE C : container bind, host connects ###############"
# Background a tiny HTTP server inside the container, curl from host.
( singularity exec "${SIF}" python3 -m http.server "${PORT_C}" --bind 127.0.0.1 ) >/tmp/probe_c_$$.log 2>&1 &
SRV_PID=$!
sleep 2
if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "PROBE_C: CONTAINER_HTTP_SRV_DIED"
    cat /tmp/probe_c_$$.log 2>/dev/null | head -10
else
    if curl -s --max-time 5 "http://127.0.0.1:${PORT_C}/" -o /dev/null -w 'PROBE_C: host_curl_to_container_http=%{http_code} time=%{time_total}\n'; then :; fi
    # ss snapshot to see which namespace owns the listener
    echo "--- host ss view ---"
    ss -ltn 2>/dev/null | grep ":${PORT_C}\b" || echo "(no listener visible to host on ${PORT_C})"
    kill "${SRV_PID}" 2>/dev/null || true
    wait "${SRV_PID}" 2>/dev/null || true
fi
rm -f /tmp/probe_c_$$.log
echo "(probe C done)"

echo
echo "############### SUMMARY ###############"
echo "If A fails  -> container can't even talk to itself (loopback / namespace problem inside container)."
echo "If A passes but B fails -> host->container loopback broken (unlikely; would also break SLURM steps)."
echo "If A passes but C fails -> container's listener invisible to host (bind-mount or netns separation)."
echo "If A,B,C all pass       -> generic TCP is fine; CARLA/UE4 specifically isn't reaching the bind() call."
echo "Date     : $(date -Is)"
