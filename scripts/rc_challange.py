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

    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
"""

import json
from datetime import datetime
import logging
import multiprocessing
import signal
import sys
import time
import traceback
import click
import requests

import chess
from reconchess import Player, RemoteGame, play_turn, ChessJSONDecoder, ChessJSONEncoder
from reconchess.scripts.rc_connect import RBCServer, check_package_version

from strangefish.models.uncertainty_lstm import uncertainty_lstm_1
from strangefish.strangefish_strategy import StrangeFish2
from strangefish.utilities import ignore_one_term
from strangefish.utilities.player_logging import create_file_handler, create_stream_handler
from strangefish.zubat_strategy.zubat_strategy import Zubat


class OurRemoteGame(RemoteGame):

    def __init__(self, server_url, game_id, auth):
        self.logger = logging.getLogger(f"game-{game_id}.server-comms")
        super().__init__(server_url, game_id, auth)

    def is_op_turn(self):
        status = self._get("game_status")
        return not status["is_over"] and not status["is_my_turn"]

    def _get(self, endpoint, decoder_cls=ChessJSONDecoder):
        self.logger.debug(f"Getting '{endpoint}'")
        done = False
        url = "{}/{}".format(self.game_url, endpoint)
        while not done:
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    done = True
                elif response.status_code >= 500:
                    time.sleep(0.5)
                else:
                    self.logger.error(response.text)
                    raise ValueError(response.text)
            except requests.RequestException as e:
                self.logger.error(e)
                time.sleep(0.5)
        response = response.json(cls=decoder_cls)
        self.logger.debug(f"response: {response}")
        return response

    def _post(self, endpoint, obj):
        self.logger.debug(f"Posting '{endpoint}' -> {obj}")
        url = "{}/{}".format(self.game_url, endpoint)
        data = json.dumps(obj, cls=ChessJSONEncoder)
        done = False
        while not done:
            try:
                response = self.session.post(url, data=data)
                if response.status_code == 200:
                    done = True
                elif response.status_code >= 500:
                    time.sleep(0.5)
                else:
                    self.logger.error(response.text)
                    raise ValueError(response.text)
            except requests.RequestException as e:
                self.logger.error(e)
                time.sleep(0.5)
        response = response.json(cls=ChessJSONDecoder)
        self.logger.debug(f"response: {response}")
        return response


def our_play_remote_game(server_url, game_id, auth, player: Player):
    game = OurRemoteGame(server_url, game_id, auth)
    logger = logging.getLogger(f"game-{game_id}.game-mod")

    op_name = game.get_opponent_name()
    our_color = game.get_player_color()

    logger.debug("Setting up remote game %d playing %s against %s.",
                 game_id, chess.COLOR_NAMES[our_color], op_name)

    player.handle_game_start(our_color, game.get_starting_board(), op_name)
    game.start()

    turn_num = 0
    while not game.is_over():
        turn_num += 1
        logger.info("Playing turn %2d. (%3.0f seconds left.)", turn_num, game.get_seconds_left())
        play_turn(game, player, end_turn_last=False)
        logger.info("   Done turn %2d. (%d boards.)", turn_num, len(player.boards))

        if hasattr(player, "while_we_wait") and getattr(player, "while_we_wait"):
            while game.is_op_turn():
                player.while_we_wait()

    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    logger.debug("Ending remote game %d against %s.", game_id, op_name)

    player.handle_game_end(winner_color, win_reason, game_history)

    return winner_color, win_reason, game_history


def play_remote_game(server_url, auth, opponent, color, batch, bot_opts):
    # make sure this process doesn't react to the first interrupt signal
    signal.signal(signal.SIGINT, ignore_one_term)

    logger = logging.getLogger("rc-connect")

    server = RBCServer(server_url, auth)

    # send invitation
    game_id = server.send_invitation(opponent, color)

    logger.info("Playing game %d.", game_id)

    player = Zubat(uncertainty_model=uncertainty_lstm_1(
        'uncertainty_model/uncertainty_lstm_3/weights'),
        game_id=game_id,
        log_dir=f"game_logs/unranked_games/{batch}/{opponent}/{game_id}",
        **bot_opts
    )

    try:
        _, _, game_history = our_play_remote_game(server_url, game_id, auth, player)
        logger.debug("Finished game %d.", game_id)
        game_history.save(f'game_logs/unranked_games/{batch}/{opponent}/{game_id}/game_{game_id}.log')
    except:
        logging.getLogger(f"game-{game_id}").exception("Fatal error in game %d.", game_id)
        traceback.print_exc()
        server.error_resign(game_id)
        player.handle_game_end(None, None, None)
        logger.critical("Game %d closed on account of error.", game_id)
    # finally:
        # server.finish_invitation(invitation_id)
        # logger.debug("Game %d ended. Invitation %d closed.", game_id, invitation_id)


def send_invitations(server, limit_games, opponent, color, batch, bot_opts):
    logger = logging.getLogger("rc-connect")
    i = 0
    t0 = 2
    t = t0

    while i < limit_games:
        try:
            # play
            play_remote_game(server.server_url, server.session.auth, opponent, color, batch, bot_opts)
            i+=1
            t=t0
        except requests.RequestException as e:
            logger.exception("Failed to connect to server")
            print(e)
            t *= 2
        except Exception:
            logger.exception("Error in invitation processing: ")
            traceback.print_exc()
            t *= 2

        if t >= t0 * 2**10:
            break
        time.sleep(t)



@click.command()
@click.argument("username")
@click.argument("password")
@click.option("--server-url", "server_url", default="https://rbc.jhuapl.edu", help="URL of the server.")
@click.option("--limit-games", "limit_games", type=int, default=None, help="Optional limit to number of games played.")
@click.option("--challenge", "challenge", type=str, default=None, help="Player to challenge")
@click.option("--color", "color", type=str, default=None, help="Color to play as")
@click.option("--batch", "batch", type=str, default=int(time.time()), help="Color to play as")
@click.option("--uncertainty_multiplier", "uncertainty_multiplier", type=float, default=0)
@click.option("--risk_taker_multiplier", "risk_taker_multiplier", type=float, default=0)
@click.option("--risk_taker_state_offset", "risk_taker_state_offset", type=float, default=0)
@click.option("--risk_taker_state_weight", "risk_taker_state_weight", type=float, default=0)
def main(
    username,
    password,
    server_url,
    limit_games,
    challenge,
    color,
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

    logger = logging.getLogger("rc-connect")
    logger.setLevel(logging.INFO)
    logger.addHandler(create_stream_handler())
    logger.addHandler(create_file_handler(f"connection_logs/rc_connect_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"))

    logger.debug(
        f"Running modified_rc_connect to play RBC games online."
        f" {server_url=}"
        f" {limit_games=}"
        f" {challenge=}"
        f" {color=}"
    )

    auth = username, password
    server = RBCServer(server_url, auth)

    # verify we have the correct version of reconchess package
    check_package_version(server)

    def handle_term(signum, frame):
        # reset to default response to interrupt signals
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        logger.warning("Received terminate signal, waiting for games to finish and then exiting.")
        server.set_ranked(False)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_term)

    # tell the server whether we want to do ranked matches or not
    server.set_ranked(False)

    send_invitations(server, limit_games, challenge, color, batch, bot_opts)


if __name__ == "__main__":
    main()
