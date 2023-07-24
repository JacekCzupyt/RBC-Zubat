#!/usr/bin/env bash

rsync -vazP --delete --exclude-from .dockerignore --exclude 'slurm-*' --exclude gamelogs/test_games . eden:~/projects/zubat
