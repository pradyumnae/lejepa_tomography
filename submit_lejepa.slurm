#!/bin/bash
#SBATCH -A als
#SBATCH -J lejepa_train
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

module load gpu
export IMAGE="nersc/pytorch:25.02.01"

# Using the local environment installed for this project
export PYTHONUSERBASE=$PSCRATCH/lejepa_tomography/env
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=4

srun shifter --image=$IMAGE bash -c "
    export MASTER_ADDR=\$(hostname)
    export MASTER_PORT=29500
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=0
    export WORLD_SIZE=\$SLURM_NTASKS
    python main.py
"
