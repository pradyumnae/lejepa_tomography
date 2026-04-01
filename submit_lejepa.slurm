#!/bin/bash
#SBATCH -J lejepa_train
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

module load pytorch/2.3.1-cu121
export LOGLEVEL=INFO

# Number of GPUs per node (Perlmutter has 4 per node)
export GPUS_PER_NODE=4

srun bash -c "
    export MASTER_ADDR=\$(hostname)
    export MASTER_PORT=29500
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    export WORLD_SIZE=\$SLURM_NTASKS
    python main.py
"
