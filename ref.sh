#disable metrics
export HPC_CARLA_SERVER_METRICS=0 HPC_CARLA_WRITE_RUN_META=0 HPC_CARLA_WRITE_STATIC_INFO=0 HPC_CARLA_DISABLE_METRICS_DAEMON=1

rm -r logs && rm -r dataset

python3 continuous_cli.py stop && python3 continuous_cli.py reset

python3 continuous_cli.py --persistent start --slurm \
  --slurm-nodelist=hpc-pr-a-pod09,hpc-pr-a-pod17 \
  --slurm-nodes=2 --slurm-gpus=8 --slurm-time=336:00:00