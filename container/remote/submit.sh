#!/usr/bin/env bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --time=0-1:00:00
#SBATCH --partition=short
#SBATCH --export=ALL
#SBATCH --account=mandziuk-lab

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Training command: python ${1}" "${@:2}"
echo "Enroot version: $(enroot version)"
enroot start \
  --rw \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -m /home2/faculty/jczupyt/projects/zubat:/app \
  -m /etc/slurm:/etc/slurm \
  jczupyt-zubat-latest \
  python "${1}" "${@:2}"
date
