#disable metrics
export HPC_CARLA_SERVER_METRICS=0 HPC_CARLA_WRITE_RUN_META=0 HPC_CARLA_WRITE_STATIC_INFO=0 HPC_CARLA_DISABLE_METRICS_DAEMON=1

rm -r logs && rm -r dataset

python3 continuous_cli.py stop && python3 continuous_cli.py reset

./start_job.sh