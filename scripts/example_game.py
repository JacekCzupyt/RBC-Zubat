"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import random
import sys
from time import time

import click
from reconchess import play_local_game, LocalGame
from reconchess.bots.trout_bot import TroutBot
from strangefish.strangefish_strategy import StrangeFish2

from strangefish.zubat_strategy.zubat_strategy import Zubat
from strangefish.models.uncertainty_lstm import uncertainty_lstm_1


@click.command()
@click.option("--batch", "batch", type=str, default=int(time()), help="Color to play as")
@click.option("--uncertainty_multiplier", "uncertainty_multiplier", type=float, default=0)
@click.option("--risk_taker_multiplier", "risk_taker_multiplier", type=float, default=0)
@click.option("--risk_taker_state_offset", "risk_taker_state_offset", type=float, default=0)
@click.option("--risk_taker_state_weight", "risk_taker_state_weight", type=float, default=0)
def play_game(
    batch,
    uncertainty_multiplier,
    risk_taker_multiplier,
    risk_taker_state_offset,
    risk_taker_state_weight
):
    bot_opts = {
        "uncertainty_multiplier": uncertainty_multiplier,
        "risk_taker_multiplier": risk_taker_multiplier,
        "risk_taker_state_offset": risk_taker_state_offset,
        "risk_taker_state_weight": risk_taker_state_weight,
    }

    game = LocalGame(seconds_per_player=900)
    zubat_color = random.random() > 0.5
    id = int(time())

    zubat = Zubat(
        uncertainty_model=uncertainty_lstm_1('uncertainty_model/uncertainty_lstm_3/weights'),
        game_id=id,
        log_dir=f"game_logs/unranked_games/{batch}/StrangeFish2/{id}",
        **bot_opts
    )
    strangefish = StrangeFish2(
        game_id=f"stragefish2_{id}_{'b' if zubat_color else 'w'}",
        log_to_file=False,
        log_dir=None,
    )

    try:
        winner_color, win_reason, game_history = play_local_game(
            zubat if zubat_color else strangefish,
            strangefish if zubat_color else zubat,
            game,
        )
    except Exception as e:
        logging.exception(e)

    game_history = game.get_game_history()
    game_history.save(f'game_logs/unranked_games/{batch}/StrangeFish2/{id}/game_{id}.log')


if __name__ == "__main__":
    try:
        play_game()
    except Exception as e:
        logging.exception(e)
