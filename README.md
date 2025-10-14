Version of the entire system with:

- Persistent CARLA instances per-GPU
- Automatic, continuous scheduling of all possible agent-route-scenario combinations
- Per-instance healthchecks

Example usage:

`python3 continuous_cli.py reset` to regenerate agent-route-scenario combos

`python3 continuous_cli.py --persistent start --slurm --slurm-gpus 8 --slurm-time 96:00:00 --slurm-nodes 1 --slurm-nodelist hpc-pr-a-pod09` where everything after start defines various SLURM options

`python3 continuous_cli.py --persistent monitor` to use healthchecks to monitor status of each instance
