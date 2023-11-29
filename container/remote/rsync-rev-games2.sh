#!/usr/bin/env bash

rsync -vazP --exclude 'slurm-*' eden:~/projects/zubat/game_logs/ranked_games/ game_logs/ranked_games
rsync -vazP --exclude 'slurm-*' eden:~/projects/zubat/game_logs/unranked_games/ game_logs/unranked_games
