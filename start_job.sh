python3 continuous_cli.py reset
python3 continuous_cli.py --persistent start --slurm --slurm-nodelist=hpc-pr-a-pod10,hpc-pr-a-pod11 --slurm-gpus=8 --slurm-nodes=2 --slurm-time=168:00:00
