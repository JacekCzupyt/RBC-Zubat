import random
from time import time
from typing import List, Tuple

import chess
import numpy as np
from tqdm import tqdm

from strangefish.strangefish_strategy import make_cache_key
from strangefish.utilities import fast_copy_board, rbc_legal_moves, simulate_move, PASS
from strangefish.utilities.rbc_move_score import ScoreConfig, calculate_score, score_material


class RiskTakerModule:
    def __init__(self, engine, score_cache, score_config=ScoreConfig(), depth=1, samples=3000, recapture_weight=5, rc_disable_pbar=False):
        self.score_cache = score_cache
        self.engine = engine
        self.rc_disable_pbar = rc_disable_pbar
        self.recapture_weight = recapture_weight
        self.samples = samples
        self.depth = depth
        self.score_config = score_config

    def get_high_risk_moves(
            self,
            boards: Tuple[chess.Board],
            moves: List[chess.Move],
            time_limit=None,
    ):
        results = {move: [] for move in moves}
        start_time = time()
        for _ in tqdm(range(self.samples), desc="Sampling for gambles", unit="Samples", disable=self.rc_disable_pbar):
            try:
                board = fast_copy_board(random.choice(boards))
                considered_move: chess.Move = random.choice(moves)
                for i in range(self.depth):
                    if i == 0:
                        my_move: chess.Move = considered_move
                    else:
                        my_move: chess.Move = random.choice(rbc_legal_moves(board))

                    # If performing a capture move, expect the enemy to attempt to recapture
                    is_capture = board.is_capture(my_move)

                    board.push(my_move)
                    if board.king(board.turn) is None:
                        results[considered_move].append(self.score_config.capture_king_score + score_material(board, board.turn))
                        break

                    if is_capture:
                        # NOTE: this does not take into account en-passant (in either direction)
                        weights = [self.recapture_weight if m.to_square == my_move.to_square else 1 for m in rbc_legal_moves(board)]
                    else:
                        weights = None

                    opponent_move = random.choices(rbc_legal_moves(board), weights=weights)[0]

                    score = -self.memo_calc_score_risk(board, opponent_move)[0]

                    board.push(opponent_move)

                    results[considered_move].append(score)
                if time_limit is not None and time() - start_time > time_limit:
                    print('Time limit for gamble sampling exceeded')
                    break

            except Exception as e:
                raise e

        # TODO: what if no samples?
        results = {move: np.nan_to_num(np.mean(scores), nan=-1000) for move, scores in results.items()}
        return results

    def memo_calc_score_risk(
            self,
            board: chess.Board,
            move: chess.Move = chess.Move.null(),
    ):
        """Memoized calculation of the score associated with one move on one board"""
        key = make_cache_key(board, simulate_move(board, move) or PASS)
        if key in self.score_cache:
            return self.score_cache[key], False

        score = calculate_score(
            board=board,
            move=move,
            engine=self.engine,
            score_config=self.score_config,
            is_op_turn=False,
        )
        self.score_cache[key] = score
        return score, True
