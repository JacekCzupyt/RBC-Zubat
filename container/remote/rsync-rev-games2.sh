#!/usr/bin/env bash

rsync -vazP --exclude 'slurm-*' eden:~/projects/zubat/game_logs/ranked_games/ game_logs/ranked_games
