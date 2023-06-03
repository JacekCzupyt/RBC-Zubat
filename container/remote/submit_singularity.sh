#!/usr/bin/env bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gpus=1
#SBATCH --time=0-14:00:00
#SBATCH --partition=short
#SBATCH --export=ALL
#SBATCH --account=mandziuk-lab
#SBATCH --output="./logs/%j.out"

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "singularity version: $(singularity version)"
echo "image: ${1}"
echo "Training command: python ${2}" "${@:3}"
singularity run \
  --nv \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --bind /home2/faculty/jczupyt/projects/zubat:/app:ro \
  "${1}" \
  python "${2}" "${@:3}"
date