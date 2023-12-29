#!/usr/bin/env bash

#SBATCH --constraint=sr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gpus=0
#SBATCH --time=0-24:00:00
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
echo "Training command: python ${3}" "${@:4}"
echo "Run command ${2} times"
for i in $(seq 1 ${2})
do
  singularity run \
    --nv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}",STOCKFISH_EXECUTABLE=/home2/faculty/jczupyt/projects/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --bind /home2/faculty/jczupyt/projects/zubat:/app:ro \
    "${1}" \
    python "${3}" "${@:4}"
  date
done