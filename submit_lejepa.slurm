#!/bin/bash
#SBATCH -A als
#SBATCH -J lejepa_train
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
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
    torchrun --nnodes 1 --nproc_per_node 4 main.py
"
