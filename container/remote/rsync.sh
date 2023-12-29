#!/usr/bin/env bash

rsync -vazP --delete \
--exclude-from .dockerignore \
--exclude 'slurm-*' \
--exclude "game_logs/*" \
--exclude "logs/*" \
. eden:~/projects/zubat
