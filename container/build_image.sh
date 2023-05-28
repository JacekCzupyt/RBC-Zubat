#!/usr/bin/env bash

IMAGE_URI="jacekczupyt/zubat:latest"

docker build -t ${IMAGE_URI} -f container/dockerfiles/tensorflow.Dockerfile .
