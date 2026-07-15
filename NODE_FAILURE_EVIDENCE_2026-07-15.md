# Node-failure diagnosis — storage (WekaFS), not memory

**Prepared 2026-07-15 from login-node evidence (no root needed). For the cluster admin.**

## Verdict
The recurrent `hpc-pr-a-pod09` / `hpc-pr-a-pod17` "Not responding" crashes are a **storage
(WekaFS) fault**, not a memory fault. The Weka cluster is **fencing our compute nodes' Weka
processes** when their DriverFrontend IOs hang, which kills `/scratch` access on that node and
leaves every process touching it in uninterruptible **D-state** → SLURM `NODE_FAIL`.

## Evidence

**1. `/scratch` is WekaFS (network storage), cluster `gist.ac.kr`.**
`df -T` → `wekafs`, 14 backend hosts (10.3.8.x/10.3.9.x), InfiniBand, `_netdev`, `weka-agent.service`.
pod09/pod17 are **converged** nodes: they run Weka backend processes *and* our compute jobs.

**2. Weka fenced our nodes at the exact SLURM NODE_FAIL timestamps.**
`weka events` around job 167116 (SLURM NODE_FAIL 2026-07-14 19:13→20:23 on pod[09,17]):
```
19:10:11  NodeUnreachable  weka node 17620 on host 881 ("hpc-pr-a-pod17") removed — unreachable
19:10:13  NodeStopped      17620 stopped, TerminationReason "FENCING"
19:13:21  NodeStarted/Rejoined
20:22:35  NodeUnreachable  weka node 17480 on host 874 ("hpc-pr-a-pod09") removed — unreachable
20:22:37  NodeStopped      17480 stopped, TerminationReason "FENCING"
20:24:04  NodeRejoined
... bracketed by repeated CRITICAL HangingDriverFrontendIosDetected (IOs hung 7,700–16,700 s)
```

**3. Standing Weka alerts (`weka alerts`, 31 active) describe the same mechanism:**
- `HangingIos` — IOs stuck for *weeks* on `WEKAFS_SET_FILE_ACCESS` / `AcquiringFileLease`
  (= the D-state wedge; a process blocked forever acquiring a file lease is unkillable).
- `HangingCacheSync` — cache sync hanging 9 weeks (can block other clients / lose writes).
- `AgentNotRunning` — weka-agent down on hpc-pr-a-pod20.
- `LowDiskSpace` (×21), `BucketCapacityExhausting` (273 buckets), `SSDCapacityTooHigh`,
  `QuotasHardLimitReached` — capacity pressure (/scratch is **88% full**, 838T/955T).

**4. Memory is excluded.**
- Nodes have **~1 TB RAM** (`RealMemory=1031711M`); historical node metrics show **~38–48 GB
  used (<5%)**. Even a full 16-GPU load (16 CARLA + 16 agents) is nowhere near 1 TB.
- **Zero** `OUT_OF_MEMORY` SLURM states in months of `sacct`; no OOM traces in any of our logs.
- Weka's client reserves only ~8 GB of hugepages. A GPU-VRAM exhaustion would crash a *process*,
  not `NODE_FAIL` the host.

## Likely trigger (converged-node contention)
Our jobs are GL/CPU-heavy (CARLA + agents, 8/GPU-node). On a converged Weka backend node, heavy
compute can starve the Weka process enough to miss its cluster heartbeat → the leader fences it →
`/scratch` dies on that node → D-state wedge → NODE_FAIL. The metadata-heavy write pattern
(millions of small `.npy` files) adds file-lease/metadata pressure that feeds the HangingIos.

## What to check / fix (admin side)
- `weka debug fs resolve-inode` on the HangingDriverFrontend IOs to find the stuck file(s).
- Why pod09/pod17 Weka processes miss heartbeats under compute load (CPU/core pinning, cgroup
  isolation of the weka client vs SLURM jobs, IB health).
- Capacity: 88% full + buckets exhausting + hard-quota tenants (the over-quota dirs are other
  tenants' k8s CSI volumes `cheetah-container:/csi-volumes/pvc-*`, not ours).
- Add a SLURM `UnkillableStepProgram` so a D-state step doesn't drain the node on cleanup.

## What we can do (our side, to reduce contribution)
- Batch/shard the per-frame `.npy` writes (or stage to node-local `/tmp` then bulk-copy) to cut
  small-file metadata + file-lease pressure on Weka.
- Throttle concurrent writers; our `park-on-unkillable` fix already contains the drain mode but
  cannot prevent a Weka fencing of the whole node.
