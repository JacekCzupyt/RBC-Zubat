#!/bin/bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=0-2:00:00
#SBATCH --export=ALL
#SBATCH --account=mandziuk-lab

DOCKER_BUILD_DIR='/vagrant'
DOCKER_FILE_PATH='/vagrant/container/dockerfiles/tensorflow.Dockerfile'
DOCKER_IMAGE_URI='jczupyt/zubat:latest'
ENROOT_IMAGE_NAME='jczupyt-zubat-latest'
OUTPUT_DIR_HOST='/home2/faculty/jczupyt/images/enroot'
OUTPUT_DIR_GUEST='/output'
OUTPUT_FILENAME="${ENROOT_IMAGE_NAME}_$(date +'%Y-%m-%d_%H-%M-%S').sqsh"

set -e

env \
  DOCKER_BUILD_DIR="${DOCKER_BUILD_DIR}" \
  DOCKER_FILE_PATH="${DOCKER_FILE_PATH}" \
  DOCKER_IMAGE_URI="${DOCKER_IMAGE_URI}" \
  OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST}" \
  OUTPUT_DIR_GUEST="${OUTPUT_DIR_GUEST}" \
  OUTPUT_FILENAME="${OUTPUT_FILENAME}" \
  CONVERT_TO_ENROOT='true' \
  vagrant up --provision
echo "Provisioned virtual machine and exported docker image to enroot image: ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"

vagrant suspend
echo "Suspended virtual machine"

enroot create --force --name "${ENROOT_IMAGE_NAME}" "${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
echo "Created new enroot image ${ENROOT_IMAGE_NAME} from ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
