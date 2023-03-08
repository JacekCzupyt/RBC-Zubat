"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

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

import logging
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from time import sleep, time
from typing import List, Tuple, Optional

import chess.engine
import numpy as np
from reconchess import Square, Color
from tqdm import tqdm

from strangefish.strangefish_mht_core import StrangeFish, RC_DISABLE_PBAR
from strangefish.utilities import (
    SEARCH_SPOTS,
    stockfish,
    simulate_move,
    sense_masked_bitboards,
    rbc_legal_moves,
    rbc_legal_move_requests,
    sense_partition_leq,
    PASS,
    fast_copy_board,
    print_sense_result,
)
from strangefish.utilities.player_logging import create_file_handler
from strangefish.utilities.rbc_move_score import calculate_score, ScoreConfig, ENGINE_CACHE_STATS


SCORE_ROUNDOFF = 1e-5
SENSE_SAMPLE_LIMIT = 2500
SCORE_SAMPLE_LIMIT = 250


# @dataclass
# class RunningEst:
#     num_samples: int = 0
#     total_weight: float = 0
#     minimum: float = None
#     maximum: float = None
#     average: float = None
#
#     def update(self, value, weight=1):
#         self.num_samples += 1
#         self.total_weight += weight
#         if self.num_samples == 1:
#             self.minimum = self.maximum = self.average = value
#         else:
#             self.average += (value - self.average) * weight / self.total_weight
#             if value < self.minimum:
#                 self.minimum = value
#             elif value > self.maximum:
#                 self.maximum = value


# @dataclass
# class SenseConfig:
#     boards_per_centipawn: float = 50  # The scaling factor for combining decision-impact and set-reduction sensing
#     expected_outcome_coef: float = 1.0  # The scaling factor for sensing to maximize the expected turn outcome
#     worst_outcome_coef: float = 0.2  # The scaling factor for sensing to maximize the worst turn outcome
#     outcome_variance_coef: float = -0.3  # The scaling factor for sensing based on turn outcome variance
#     score_variance_coef: float = 0.15  # The scaling factor for sensing based on move score variance
#
#
# @dataclass
# class MoveConfig:
#     mean_score_factor: float = 0.7  # relative contribution of a move's average outcome on its compound score
#     min_score_factor: float = 0.3  # relative contribution of a move's worst outcome on its compound score
#     max_score_factor: float = 0.0  # relative contribution of a move's best outcome on its compound score
#     threshold_score: float = 10  # centipawns below best compound score in which any move will be considered
#     sense_by_move: bool = False  # Use bonus score to encourage board set reduction by attempted moves
#     force_promotion_queen: bool = True  # change all pawn-promotion moves to choose queen, otherwise it's often a knight
#     sampling_exploration_coef: float = 1_000.0  # UCT coef for sampling moves to eval  # TODO: tune
#     move_sample_rep_limit: int = 100  # Number of consecutive iterations with same best move before breaking loop  # TODO: tune
#
#
# @dataclass
# class TimeConfig:
#     turns_to_plan_for: int = 16  # fixed number of turns over which the remaining time will be divided
#     min_time_for_turn: float = 3.0  # minimum time to allocate for a turn
#     max_time_for_turn: float = 40.0  # maximum time to allocate for a turn
#     time_for_sense: float = 0.7  # fraction of turn spent in choose_sense
#     time_for_move: float = 0.3  # fraction of turn spent in choose_move
#     calc_time_per_move: float = 0.005  # starting time estimate for move score calculation
#
#
# # Create a cache key for the requested board and move
# def make_cache_key(board: chess.Board, move: chess.Move = PASS, prev_turn_score: int = None):
#     return board, move, prev_turn_score


class OracleFish(StrangeFish):

    def __init__(
        self,

        log_to_file=True,
        game_id=None,
        rc_disable_pbar=RC_DISABLE_PBAR
        #
        # load_score_cache: bool = True,
        # load_opening_book: bool = True,
        #
        # sense_config: SenseConfig = SenseConfig(),
        # move_config: MoveConfig = MoveConfig(),
        # score_config: ScoreConfig = ScoreConfig(),
        # time_config: TimeConfig = TimeConfig(),
        #
        # board_weight_90th_percentile: float = 3_000,
        # min_board_weight: float = 0.02,
        #
        # while_we_wait_extension: bool = True,
    ):
        """
        Constructs an instance of the StrangeFish2 agent.

        :param log_to_file: A boolean flag to turn on/off logging to file game_logs/game_<game_id>.log
        :param game_id: Any printable identifier for logging (typically, the game number given by the server)
        :param rc_disable_pbar: A boolean flag to turn on/off the tqdm progress bars

        :param load_score_cache: A boolean flag to turn on/off pre-computed score cache
        :param load_opening_book: A boolean flag to turn on/off pre-computed opening book

        :param sense_config: A dataclass of parameters which determine the sense strategy's score calculation
        :param move_config: A dataclass of parameters which determine the move strategy's score calculation
        :param score_config: A dataclass of parameters which determine the score assigned to a board's strength
        :param time_config: A dataclass of parameters which determine how time is allocated between turns

        :param board_weight_90th_percentile: The centi-pawn score associated with a 0.9 weight in the board set
        :param min_board_weight: A lower limit on relative board weight `w = max(w, min_board_weight)`

        :param while_we_wait_extension: A bool that toggles the scoring of boards that could be reached two turns ahead
        """
        super().__init__(log_to_file=log_to_file, game_id=game_id, rc_disable_pbar=rc_disable_pbar)

        # self.logger.debug("Creating new instance of StrangeFish2.")
        #
        # engine_logger = logging.getLogger("chess.engine")
        # engine_logger.setLevel(logging.DEBUG)
        # file_handler = create_file_handler(f"engine_logs/game_{game_id}_engine.log", 10_000)
        # engine_logger.addHandler(file_handler)
        # engine_logger.debug("File handler added to chess.engine logs")
        #
        # self.load_score_cache = load_score_cache
        # self.load_opening_book = load_opening_book
        # self.opening_book = None
        #
        # self.sense_config = sense_config
        # self.move_config = move_config
        # self.score_config = score_config
        # self.time_config = time_config
        #
        # self.swap_sense_time = 90
        # self.swap_sense_size = 10_000
        # self.swap_sense_min_size = 150
        #
        # self.time_switch_aggro = 60
        #
        # self.extra_move_time = False
        #
        # self.board_weight_90th_percentile = board_weight_90th_percentile
        # self.min_board_weight = min_board_weight
        # self.while_we_wait_extension = while_we_wait_extension
        #
        # # Initialize a list to store calculation time data for dynamic time management
        # self.score_calc_times = []
        #
        # self.score_cache = dict()
        # self.boards_in_cache = set()
        #
        self.engine = None

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        super().handle_game_start(color, board, opponent_name)

        self.engine = stockfish.create_engine()

    def sense_strategy(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        # Don't sense if there is nothing to learn from it
        if len(self.boards) == 1:
            return None

        return self.sense_min_states(sense_actions, moves, seconds_left)

    def sense_min_states(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a sense square to minimize the expected board set size."""

        sample_size = min(len(self.boards), SENSE_SAMPLE_LIMIT)

        self.logger.debug(f"In sense phase with {seconds_left:.2f} seconds left. Set size is {sample_size}.")

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        sense_results = defaultdict(lambda: defaultdict(set))
        sense_possibilities = defaultdict(set)

        # Get a random sampling of boards from the board set
        board_sample = random.sample(self.boards, sample_size)

        self.logger.debug(f"Sampled {len(board_sample)} boards out of {len(self.boards)} for sensing.")

        for board in tqdm(board_sample, disable=self.rc_disable_pbar,
                          desc="Sense quantity evaluation", unit="boards"):

            # Gather information about sense results for each square on each board
            for square in SEARCH_SPOTS:
                sense_result = sense_masked_bitboards(board, square)
                num_occurances[square][sense_result] += 1
                sense_results[board][square] = sense_result
                sense_possibilities[square].add(sense_result)

        # Calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(self.boards) *
                (1 - (1 / len(board_sample) ** 2) *
                 sum([num_occurances[square][sense_result] ** 2 for sense_result in sense_possibilities[square]]))
            for square in SEARCH_SPOTS
        }

        max_sense_score = max(expected_set_reduction.values())
        sense_choice = random.choice(
            [
                square
                for square, score in expected_set_reduction.items()
                if abs(score - max_sense_score) < SCORE_ROUNDOFF
            ]
        )
        return sense_choice

    def move_strategy(self, moves: List[chess.Move], seconds_left: float):

        move_votes = defaultdict(int)

        for board in tqdm(self.boards, desc='Evaluating best moves for each board'):
            move = self._get_engine_move(next(iter(self.boards)))
            move_votes[move] += 1

        return max(move_votes, key=move_votes.get)

    def _get_engine_move(self, board: chess.Board):
        self.logger.debug("Only one possible board; using win-or-engine-move method")

        # Capture the opponent's king if possible
        if board.was_into_check():
            op_king_square = board.king(not board.turn)
            king_capture_moves = [
                move for move in board.pseudo_legal_moves
                if move and move.to_square == op_king_square
            ]
            return random.choice(king_capture_moves)

        # If in checkmate or stalemate, return
        if not board.legal_moves:
            return PASS

        # Otherwise, let the engine decide

        # For Stockfish versions based on NNUE, seem to need to screen illegal en passant
        replaced_ep_square = None
        if board.ep_square is not None and not board.has_legal_en_passant():
            replaced_ep_square, board.ep_square = board.ep_square, replaced_ep_square
        # Then analyse the position
        move = self.engine.play(board, limit=chess.engine.Limit(time=0.4)).move
        # Then put back the replaced ep square if needed
        if replaced_ep_square is not None:
            board.ep_square = replaced_ep_square

        self.logger.debug(f"Engine chose to play {move}")
        return move

    def gameover_strategy(self):
        """
        Quit the StockFish engine instance(s) associated with this strategy once the game is over.
        """

        # Shut down StockFish
        self.logger.debug("Terminating engine.")
        self.engine.quit()
        self.logger.debug("Engine exited.")
