# HPC-CARLA: Single-Node, Multi-GPU Data Generation & Dashboard

This repo contains the code for (paper name stand-in), and provides:

1. a `.def` file for building the CARLA container via Singularity,
2. `generate_data_consolidated.sh`, which is the main distributed SLURM launch orchestration script, and
3. a lightweight Streamlit dashboard for browsing generated data.

---

## 0) Prerequisites

* **GPU node** with recent NVIDIA drivers.
* **Singularity/Apptainer** with GPU support (`--nv`).

  * Use `apptainer` on newer clusters, `singularity` on older ones. Commands below show both.
* **SLURM** (for job submission).
* **Python 3.8+** on your workstation (or a compute node) for the dashboard:

  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install streamlit plotly pandas pillow numpy
  ```

Recommended layout (you can adjust via env vars):

```
PROJECT_ROOT/
├─ carla_official.def
├─ carla_official.sif         # created after build
├─ generate_data_consolidated.sh
├─ local_carla_dashboard.py
├─ leaderboard/
│  ├─ ...
│  └─ team_code/
│     ├─ ...
│     ├─ consolidated_agent.py
│     └─ configs/
│        └─ interfuser.yaml   # example agent config
│     └─ interfuser
│        └─ interfuser_agent.py
└─ dataset/
```

---

## 1) Build the CARLA image from the `.def`

From `PROJECT_ROOT`:

### With **Apptainer**

```bash
module load apptainer  # if your cluster uses modules
apptainer build carla_official.sif carla_official.def
```

### With **Singularity**

```bash
module load singularity # if needed
singularity build carla_official.sif carla_official.def
```

> Tip: If your cluster requires unprivileged builds, use the provided builder (`apptainer build …`) on the cluster. If you must build remotely, check `apptainer remote` / `singularity remote`.

**Quick check** (should print nothing and exit 0):

```bash
apptainer exec --nv carla_official.sif python -c "import carla"
# or
singularity exec --nv carla_official.sif python -c "import carla"
```

---

## 2) Run data generation as a SLURM job

The script launches **one CARLA simulator per GPU** and runs routes in parallel using the **ConsolidatedAgent** wrapper. All hard-coded paths have been replaced with **environment variables** (see table below).

### Key environment variables

| Var                              | Meaning                                                                      | Default                                      |
| -------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------- |
| `PROJECT_ROOT`                   | Host path to mount at `/workspace` inside the container                      | `$PWD`                                       |
| `CARLA_SIF`                      | Path to the built image                                                      | `$PROJECT_ROOT/carla_official.sif`           |
| `AGENT_DIR_HOST`                 | Host path to your agent code (mounted at `/workspace/leaderboard/team_code`) | `$PROJECT_ROOT/leaderboard/team_code`        |
| `AGENT_TYPE`                     | Logical name for agent/config set (affects output path)                      | `interfuser`                                 |
| `AGENT_YAML_CONFIG_HOST`         | Host path to agent YAML                                                      | `$AGENT_DIR_HOST/configs/${AGENT_TYPE}.yaml` |
| `GPUS`                           | Number of GPUs to use (one CARLA per GPU)                                    | `8`                                          |
| `WEATHER_START` / `WEATHER_END`  | Weather index range to evaluate (1-based)                                    | `1` / `8`                                    |
| `BASE_RPC_PORT` / `BASE_TM_PORT` | Starting RPC / Traffic Manager ports                                         | `2000` / `8000`                              |
| `LOG_DIR`                        | Host path for per-GPU logs and launchers                                     | `$PROJECT_ROOT/logs`                         |

> Weather indices map to:
> `1:ClearNoon, 2:CloudyNoon, 3:WetNoon, 4:WetCloudyNoon, 5:MidRainyNoon, 6:HardRainNoon, 7:SoftRainNoon, 8:ClearSunset, 9:CloudySunset, 10:WetSunset, 11:WetCloudySunset, 12:MidRainySunset, 13:HardRainSunset, 14:SoftRainSunset`
> (The script uses 1..8 by default.)

### Submit a job (typical)

From `PROJECT_ROOT`:

```bash
export PROJECT_ROOT=$PWD
export CARLA_SIF=$PROJECT_ROOT/carla_official.sif
export AGENT_DIR_HOST=$PROJECT_ROOT/leaderboard/team_code
export AGENT_TYPE=interfuser

# Optional overrides:
# export GPUS=4 WEATHER_START=1 WEATHER_END=4
# export BASE_RPC_PORT=2500 BASE_TM_PORT=8500
# export LOG_DIR=$PROJECT_ROOT/logs

sbatch generate_data_consolidated.sh
```

Grab the **JobID** from `sbatch` output. Check status:

```bash
squeue -j <JOBID>
# or
scontrol show job <JOBID>
```

Logs:

* SLURM stdout/stderr in the submission directory: `carla_collect_<JOBID>.out|err`
* Per-GPU worker logs in `$LOG_DIR/`:

  * `gpu${ID}_w${WEATHER_IDX}.out|err`
  * The script also generates `launch_gpu${ID}.sh` for reproducibility/debugging.

### Output location & structure

Data are saved under:

```
$PROJECT_ROOT/dataset/agent-${AGENT_TYPE}/weather-${W}/routes_townXX_YYY/
├─ rgb*/0000.png, 0001.png, ...
├─ lidar/0000.npy, ...
├─ depth_*/0000.npy, ...
├─ semantic_*/0000.png, ...
├─ gps/0000.json, imu/0000.json, ...
└─ results.json, metadata.json
```

---

## 3) Explore results with `local_carla_dashboard.py`

The dashboard is a Streamlit app. You can run it:

* **On your workstation** after copying or mounting the `dataset/` folder locally, or
* **On a cluster node** (with SSH port-forwarding).

### A) Run locally (recommended)

1. (One-time) Install dependencies in a virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install streamlit plotly pandas pillow numpy
```

2. Launch the app and point it at your dataset:

```bash
# In the repo root, for example:
export CARLA_DATA_DIR=/path/to/dataset  # or pass --data-dir below
streamlit run local_carla_dashboard.py -- --data-dir "$CARLA_DATA_DIR"
```

The app will tell you the local URL (usually `http://localhost:8501`).

### B) Run on a remote cluster with SSH tunneling

On the **cluster** (interactive shell on a GPU or CPU node is fine):

```bash
source .venv/bin/activate  # your Python env with Streamlit
export CARLA_DATA_DIR=/path/to/dataset
streamlit run --server.port 8501 --server.headless true local_carla_dashboard.py -- --data-dir "$CARLA_DATA_DIR"
```

On your **laptop/workstation**, open a tunnel:

```bash
ssh -N -L 8501:<compute-node-hostname>:8501 <user>@<cluster-login-host>
```

Then visit `http://localhost:8501`.

> If your cluster restricts inbound ports, run the dashboard on a login node only if allowed by policy; otherwise, request an interactive compute allocation (e.g., `salloc`/`srun`) and tunnel to that node.

---

## Acknowledgements

This codebase reuses and adapts components from the following excellent open-source projects. Huge thanks to the authors and maintainers:

* **dotchen/LAV**
* **opendilab/InterFuser**
* **OpenDriveLab/TCP**
* **carla-simulator/leaderboard**

All original copyrights remain with their respective owners, and usage here follows the terms of their licenses as provided in each project’s repository. This project is not affiliated with or endorsed by the above maintainers.