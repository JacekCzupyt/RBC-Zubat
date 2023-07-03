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
import json
import os.path
import random
from collections import defaultdict
from time import time
from typing import List, Optional, Tuple

import chess.engine
import numpy as np
from reconchess import Square, Color
from tqdm import tqdm

from strangefish.strangefish_mht_core import StrangeFish, RC_DISABLE_PBAR
from strangefish.utilities import (
    SEARCH_SPOTS,
    stockfish,
    sense_masked_bitboards,
    PASS, fast_copy_board, rbc_legal_move_requests, simulate_move, rbc_legal_moves,
)
from strangefish.utilities.chess_model_embedding import chess_model_embedding
from strangefish.utilities.rbc_move_score import calculate_score
from strangefish.zubat_strategy.risk_taker import get_high_risk_moves

SCORE_ROUNDOFF = 1e-5
SENSE_SAMPLE_LIMIT = 2500
SCORE_SAMPLE_LIMIT = 250


class Zubat(StrangeFish):

    def __init__(
            self,

            log_to_file=True,
            game_id=int(time()),
            rc_disable_pbar=RC_DISABLE_PBAR,

            uncertainty_model=None,
            move_vote_value=100,
            uncertainty_multiplier=50,
            log_move_scores=True,
            log_path="game_logs/move_score_logs"
    ):
        """
        Constructs an instance of the StrangeFish2 agent.

        :param log_to_file: A boolean flag to turn on/off logging to file game_logs/game_<game_id>.log
        :param game_id: Any printable identifier for logging (typically, the game number given by the server)
        :param rc_disable_pbar: A boolean flag to turn on/off the tqdm progress bars
        """
        super().__init__(log_to_file=log_to_file, game_id=game_id, rc_disable_pbar=rc_disable_pbar)
        self.engine = None
        self.uncertainty_model = uncertainty_model

        self.previous_requested_move = None
        self.previous_move_piece_type = None
        self.previous_taken_move = None
        self.previous_move_capture = None
        self.opponent_capture = None
        self.sense_position = None

        self.move_vote_value = move_vote_value
        self.uncertainty_multiplier = uncertainty_multiplier

        self.network_input_sequence = []

        self.log_move_scores = log_move_scores
        self.move_scores_log = []
        self.log_path = log_path

        self.game_id = game_id

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        super().handle_game_start(color, board, opponent_name)

        self.engine = stockfish.create_engine()

    def sense_strategy(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        # Don't sense if there is nothing to learn from it
        if len(self.boards) == 1:
            return None

        self.sense_position = self.sense_min_states(sense_actions, moves, seconds_left)

        return self.sense_position

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
        uncertainty_results = defaultdict(float)

        if self.uncertainty_model:
            inputs, uncertainty_results = self.uncertainty_strategy(moves, seconds_left)

        analytical_results = self.analytical_strategy(moves, seconds_left)

        gamble_results = get_high_risk_moves(
            self.engine, tuple(self.boards), moves
        )

        results = {
            move:
                uncertainty_results[move] * self.uncertainty_multiplier +
                analytical_results[move] +
                gamble_results[move] * uncertainty_results[move]
            for move in moves
        }

        move = max(results, key=results.get)

        if self.log_move_scores:
            move_dict = {
                "move_number": len(self.move_scores_log) + 1,
                "chosen_move": str(move),
                "moves": sorted([
                    {
                        "move": str(move),
                        "score": results[move],
                        "analytical": analytical_results[move],
                        "uncertainty": uncertainty_results[move],
                        "gamble": gamble_results[move],
                    }
                    for move in moves
                ], key=lambda d: -d["score"])
            }
            self.move_scores_log.append(move_dict)

        if self.uncertainty_model:
            self.network_input_sequence += [inputs[moves.index(move)]]

        return move

    def analytical_strategy(self, moves: List[chess.Move], seconds_left: float):
        move_votes = defaultdict(float)

        for board in tqdm(self.boards, desc='Evaluating best moves for each board'):
            move = self._get_engine_move(board)
            move_votes[move] += self.move_vote_value / len(self.boards)

        return move_votes

    def uncertainty_strategy(self, moves: List[chess.Move], seconds_left: float):
        move_votes = defaultdict(float)

        inputs, uncertainties = self.measure_uncertainty(moves)

        for i in range(len(moves)):
            move_votes[moves[i]] += uncertainties[i]

        return inputs, move_votes

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        self.previous_requested_move = requested_move
        if requested_move is not None:
            self.previous_move_piece_type = next(iter(self.boards)).piece_at(requested_move.from_square).piece_type
        else:
            self.previous_move_piece_type = None
        self.previous_taken_move = taken_move
        self.previous_move_capture = capture_square

        super().handle_move_result(requested_move, taken_move, captured_opponent_piece, capture_square)

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.opponent_capture = capture_square
        super().handle_opponent_move_result(captured_my_piece, capture_square)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        self.sense_result = sense_result

        super().handle_sense_result(sense_result)

    def measure_uncertainty(self, moves: List[chess.Move]):
        # TODO: split next requested move from rest

        inputs = [
            chess_model_embedding(
                self.color,
                self.previous_requested_move,
                self.previous_move_piece_type,
                self.previous_taken_move,
                self.previous_move_capture,
                self.opponent_capture,
                self.sense_position,
                self.sense_result,
                next(iter(self.boards)),
                move
            ) for move in moves]

        results = np.array(
            [self.uncertainty_model.predict(np.array([self.network_input_sequence + [input]])) for input in inputs]
        )

        return inputs, results.flatten()

    def _get_engine_move(self, board: chess.Board):
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

        if self.log_move_scores:
            with open(os.path.join(self.log_path, f"{self.game_id}.json"), "w") as outfile:
                json.dump(self.move_scores_log, outfile)

        # Shut down StockFish
        self.logger.debug("Terminating engine.")
        self.engine.quit()
        self.logger.debug("Engine exited.")
