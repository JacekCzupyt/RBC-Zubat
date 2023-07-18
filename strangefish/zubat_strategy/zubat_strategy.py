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
import csv
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
from strangefish.strangefish_strategy import RunningEst, make_cache_key, MoveConfig
from strangefish.utilities import (
    SEARCH_SPOTS,
    stockfish,
    sense_masked_bitboards,
    PASS, fast_copy_board, rbc_legal_move_requests, simulate_move, rbc_legal_moves,
)
from strangefish.utilities.chess_model_embedding import chess_model_embedding
from strangefish.utilities.rbc_move_score import calculate_score, ScoreConfig
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
        self.log_path = log_path

        self.game_id = game_id

        self.score_cache = dict()
        self.boards_in_cache = set()

        # TODO
        self.move_config = MoveConfig()
        self.score_config = ScoreConfig()

        if self.log_move_scores:
            self.move_score_log_path = os.path.join(self.log_path, f"{self.game_id}_{'w' if self.color else 'b'}.csv")
            with open(self.move_score_log_path, "w", newline='') as outfile:
                writer = csv.DictWriter(
                    outfile,
                    fieldnames=["move_number", "move", "score", "analytical", "uncertainty", "gamble"]
                )
                writer.writeheader()




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
        moves = list(analytical_results.keys())

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
            moves_log = sorted([
                {
                    "move_number": self.turn_num,
                    "move": str(move),
                    "score": results[move],
                    "analytical": analytical_results[move],
                    "uncertainty": uncertainty_results[move],
                    "gamble": gamble_results[move],
                }
                for move in moves
            ], key=lambda d: -d["score"])
            with open(self.move_score_log_path, "a", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=["move_number", "move", "score", "analytical", "uncertainty", "gamble"])
                writer.writerows(moves_log)

        if self.uncertainty_model:
            self.network_input_sequence += [inputs[moves.index(move)]]

        return move

    def allocate_time(self, seconds_left: float, fraction_turn_passed: float = 0):
        """Determine how much of the remaining time should be spent on (the rest of) the current turn."""
        # TODO: improve, split time among modules?
        return 10
        # turns_left = self.time_config.turns_to_plan_for - fraction_turn_passed  # account for previous parts of turn
        # equal_time_split = seconds_left / turns_left
        # return min(max(equal_time_split, self.time_config.min_time_for_turn), self.time_config.max_time_for_turn)

    # def analytical_strategy(self, moves: List[chess.Move], seconds_left: float):
    #     move_votes = defaultdict(float)
    #
    #     for board in tqdm(self.boards, desc='Evaluating best moves for each board'):
    #         move = self._get_engine_move(board)
    #         move_votes[move] += self.move_vote_value / len(self.boards)
    #
    #     return move_votes

    def analytical_strategy(self, moves: List[chess.Move], seconds_left: float):
        """
        Choose the move with the maximum score calculated from a combination of mean, min, and max possibilities.

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. Each move is scored on each board, and the
        resulting scores are assessed together by looking at the worst-case score, the average score, and the best-case
        score. The relative contributions of these components to the compound score are determined by a config object.
        Deterministic move patterns are reduced by randomly choosing a move that is within a configurable range of the
        maximum score.
        """

        # TODO: agro
        # if seconds_left < self.time_switch_aggro:
        #     self.time_switch_aggro = -100
        #     self.last_ditch_plan()

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_phase = self.allocate_time(seconds_left)
        # time_for_turn = self.allocate_time(seconds_left)
        # if self.extra_move_time:
        #     time_for_phase = time_for_turn
        #     self.extra_move_time = False
        # else:
        #     time_for_phase = time_for_turn * self.time_config.time_for_move

        # self.logger.debug(f"In move phase with {seconds_left:.2f} seconds left. "
        #                   f"Allowing up to {time_for_phase:.2f} seconds for this move step.")

        # First compute valid move requests and taken move -> requested move mappings
        possible_move_requests = set(moves)
        valid_move_requests = set()
        all_maps_to_taken_move = {}
        all_maps_from_taken_move = {}
        for board in tqdm(self.boards, desc="Writing move maps", unit="boards", disable=self.rc_disable_pbar):
            legal_moves = set(rbc_legal_moves(board))
            valid_move_requests |= legal_moves
            map_to_taken_move = {}
            map_from_taken_move = defaultdict(set)
            for requested_move in moves:
                taken_move = simulate_move(board, requested_move, legal_moves) or PASS
                map_to_taken_move[requested_move] = taken_move
                map_from_taken_move[taken_move].add(requested_move)
            all_maps_to_taken_move[board] = map_to_taken_move
            all_maps_from_taken_move[board] = map_from_taken_move
        # Filter main list of moves and all move maps
        moves = possible_move_requests & valid_move_requests
        all_maps_from_taken_move = {
            board:
                {
                    taken_move: requested_moves & moves
                    for taken_move, requested_moves in map_from_taken_move.items()
                }
            for board, map_from_taken_move in all_maps_from_taken_move.items()
        }

        # Initialize move score estimates and populate with any pre-computed scores
        move_scores = defaultdict(RunningEst)
        boards_to_sample = {move: set() for move in moves}
        for board in tqdm(self.boards, desc="Reading pre-computed move scores", unit="boards", disable=self.rc_disable_pbar):
            try:
                score_before_move = self.score_cache[make_cache_key(board)]
            except KeyError:
                for move in moves:
                    boards_to_sample[move].add(board)
            else:
                # weight = self.weight_board_probability(score_before_move)
                for taken_move, requested_moves in all_maps_from_taken_move[board].items():
                    try:
                        score = self.score_cache[make_cache_key(board, taken_move, -score_before_move)]
                    except KeyError:
                        for move in requested_moves:
                            boards_to_sample[move].add(board)
                    else:
                        for move in requested_moves:
                            move_scores[move].update(score)
        incomplete_moves = {move for move in moves if boards_to_sample[move]}

        # Until stop time, compute board scores
        start_time = time()
        phase_end_time = start_time + time_for_phase
        total_evals = len(self.boards) * len(moves)  # TODO: change this to match new mapping
        num_evals_done = num_precomputed = sum(est.num_samples for est in move_scores.values())
        with tqdm(desc="Computing move scores", unit="evals", disable=self.rc_disable_pbar, total=total_evals) as pbar:
            pbar.update(num_precomputed)

            sorted_priorities = sorted(self.board_sample_priority.keys(), reverse=True)
            top_move_repetition = (None, 0)

            while incomplete_moves and time() < phase_end_time:

                # On each iteration, choose a move to evaluate using a similar scheme to UCT
                exploration_const = np.sqrt(np.log(num_evals_done + 1)) * self.move_config.sampling_exploration_coef
                values = {
                    move: np.inf if move not in move_scores or move_scores[move].num_samples == 0 else (
                        exploration_const * np.sqrt(1 / move_scores[move].num_samples)
                        + move_scores[move].minimum * self.move_config.min_score_factor
                        + move_scores[move].maximum * self.move_config.max_score_factor
                        + move_scores[move].average * self.move_config.mean_score_factor
                    ) for move in moves
                }
                # # First evaluate current best move for possible early stopping
                # top_move = max(moves, key=values.get)
                # prev_top_move, num_reps = top_move_repetition
                # if top_move == prev_top_move:
                #     top_move_repetition = (top_move, num_reps + 1)
                #     if num_reps >= self.move_config.move_sample_rep_limit and top_move not in incomplete_moves:
                #         self.logger.debug("Move choice seems to be converged; breaking loop")
                #         break
                # else:
                #     top_move_repetition = (top_move, 0)
                # Otherwise, sample a move to evaluate
                move_to_eval = max(incomplete_moves, key=values.get)

                # Then iterate through boards in descending priority to get one for eval
                needed_boards_for_eval = boards_to_sample[move_to_eval]
                for priority in sorted_priorities:
                    priority_boards = needed_boards_for_eval & self.board_sample_priority[priority]
                    if priority_boards:
                        board_to_eval = priority_boards.pop()

                        # Get the score for the corresponding taken move, then map back to all equivalent move requests
                        taken_move_to_eval = all_maps_to_taken_move[board_to_eval][move_to_eval]

                        # Get the board position score before moving
                        score_before_move, _ = self.memo_calc_score(board_to_eval, key=make_cache_key(board_to_eval))

                        # Get the score and update the estimate
                        score, _ = self.memo_calc_score(
                            board_to_eval, taken_move_to_eval, -score_before_move,
                            make_cache_key(board_to_eval, taken_move_to_eval, -score_before_move),
                        )
                        for requested_move in all_maps_from_taken_move[board_to_eval][taken_move_to_eval]:
                            move_scores[requested_move].update(score)

                            boards_to_sample[requested_move].remove(board_to_eval)
                            if not boards_to_sample[requested_move]:
                                incomplete_moves.remove(requested_move)

                            num_evals_done += 1
                            pbar.update()

                        break

                else:
                    raise AssertionError("This can only be reached if a move eval is requested when already completed")

                pbar.update()

        self.logger.debug(f"Had {num_precomputed} of {total_evals} already cached")
        self.logger.debug(f"Spent {time() - start_time:0.1f} seconds computing new move scores")
        self.logger.debug(f"Sampled {num_evals_done} of {total_evals} move+board pairs.")

        # Combine the mean, min, and max possible scores based on config settings
        compound_score = {
            move: (
                    est.minimum * self.move_config.min_score_factor
                    + est.maximum * self.move_config.max_score_factor
                    + est.average * self.move_config.mean_score_factor
            ) for move, est in move_scores.items()
        }

        return compound_score

    def uncertainty_strategy(self, moves: List[chess.Move], seconds_left: float):
        move_votes = defaultdict(float)

        inputs, uncertainties = self.measure_uncertainty(moves)

        for i in range(len(moves)):
            move_votes[moves[i]] += uncertainties[i]

        return inputs, move_votes

    def memo_calc_score(
        self,
        board: chess.Board,
        move: chess.Move = chess.Move.null(),
        prev_turn_score: int = None,
        key = None,
    ):
        """Memoized calculation of the score associated with one move on one board"""
        if key is None:
            key = make_cache_key(board, simulate_move(board, move) or PASS, prev_turn_score)
        if key in self.score_cache:
            return self.score_cache[key], False

        score = calculate_score(
            board=board,
            move=move,
            prev_turn_score=prev_turn_score or 0,
            engine=self.engine,
            score_config=self.score_config,
            is_op_turn=prev_turn_score is None,
        )
        return score, True

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
            self.uncertainty_model.predict(np.array([self.network_input_sequence + [input]]) for input in inputs)
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

        # Shut down StockFish
        self.logger.debug("Terminating engine.")
        self.engine.quit()
        self.logger.debug("Engine exited.")
