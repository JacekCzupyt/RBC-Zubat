#!/usr/bin/env bash

rsync -vazP --delete --exclude-from .dockerignore --exclude 'slurm-*' . eden:~/projects/zubat
