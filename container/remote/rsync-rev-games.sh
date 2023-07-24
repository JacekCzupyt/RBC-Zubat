#!/usr/bin/env bash

rsync -vazP --delete --exclude 'slurm-*' eden:~/projects/zubat/game_logs/test_games/ ./game_logs/test_games
