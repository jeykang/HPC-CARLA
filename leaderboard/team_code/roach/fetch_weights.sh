#!/usr/bin/env bash
# Fetch the Roach CILRS (camera IL agent) pretrained checkpoint.
#
# The trained models live on Weights & Biases (public project
# iccv21-roach/trained-models). This repo's roach.yaml targets the L_K LeaderBoard
# checkpoint, W&B run 12uzu2lu, file ckpt/ckpt_24.pth (~287 MB), and its config /
# policy_init_kwargs pin the exact architecture the pipeline mirrors.
#
# The W&B public GraphQL API + signed directUrl are reachable through the cluster
# proxy without a W&B account (no auth needed for this public project). If that ever
# changes, install `wandb`, run `wandb login`, and use `wandb.Api()` as CilrsAgent.setup
# does (agents/cilrs/cilrs_agent.py:22-40).
#
# Usage:  bash fetch_weights.sh              # fetch default (L_K LeaderBoard)
#         RUN=nw226h5h bash fetch_weights.sh # fetch a different run (see table below)
#
# LeaderBoard IL checkpoints (DAGGER iter 5), from README.md:
#   1myvm4mw L_A(AP) | nw226h5h L_A | 12uzu2lu L_K | 3ar2gyqw L_K+L_V
#   9rcwt5fh L_K+L_F | 2qq2rmr1 L_K+L_V+L_F | zwadqx9z L_K+L_F(c) | 21trg553 L_K+L_V+L_F(c)
# NoCrash IL checkpoints exist too (39o1h862, v5kqxe3i, t3x557tv, ...); see README.md.
#
# NOTE: different runs may use different input_states / action_distribution / heads.
# Each ckpt is self-describing (torch.load(...)['policy_init_kwargs']); if you switch
# RUN, reconcile roach.yaml's TorchModelRunner model.args with that ckpt's
# policy_init_kwargs (and the CilrsStateVector input_states / CilrsActionFromBranches
# action_distribution), or the state_dict load will shape-mismatch.
set -euo pipefail

ENTITY="iccv21-roach"
PROJECT="trained-models"
RUN="${RUN:-12uzu2lu}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT_DIR}/cilrs_ckpt_lk_${RUN}.pth"

if [[ -f "${OUT}" ]]; then
  echo "Already present: ${OUT}"
  exit 0
fi

echo "Resolving ckpt directUrl for ${ENTITY}/${PROJECT}/${RUN} ..."
GQL="$(cat <<EOF
{"query":"query { project(name:\"${PROJECT}\", entityName:\"${ENTITY}\"){ run(name:\"${RUN}\"){ files(first:50){ edges{ node{ name sizeBytes directUrl } } } } } }"}
EOF
)"

RESP="$(curl -sSL --max-time 60 -H 'Content-Type: application/json' \
  -X POST --data "${GQL}" https://api.wandb.ai/graphql)"

DIRECT_URL="$(python3 - "$RESP" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
edges = d["data"]["project"]["run"]["files"]["edges"]
ckpts = [e["node"] for e in edges if e["node"]["name"].startswith("ckpt/")]
if not ckpts:
    sys.exit("no ckpt file found for run")
# pick the highest step: ckpt/ckpt_<n>.pth
def step(n):
    try: return int(n["name"].split("_")[1].split(".")[0])
    except Exception: return -1
best = max(ckpts, key=step)
print(best["directUrl"])
PY
)"

echo "Downloading ${OUT} ..."
curl -sSL --max-time 600 -o "${OUT}" "${DIRECT_URL}"
echo "Done: ${OUT}"
python3 - "$OUT" <<'PY'
import os, sys
p = sys.argv[1]
print("size: %.1f MB" % (os.path.getsize(p) / 1e6))
PY
