import random
from collections import defaultdict
from time import time
from typing import List, Tuple

import chess
import numpy as np
from tqdm import tqdm

from strangefish.strangefish_strategy import make_cache_key
from strangefish.utilities import fast_copy_board, rbc_legal_moves, simulate_move, PASS
from strangefish.utilities.rbc_move_score import ScoreConfig, calculate_score, score_material


class RiskTakerModule:
    def __init__(self, engine, score_cache, logger, score_config=ScoreConfig(), depth=1, samples=3000, recapture_weight=10, exploration_factor=200, rc_disable_pbar=False):
        self.exploration_factor = exploration_factor
        self.score_cache = score_cache
        self.engine = engine
        self.rc_disable_pbar = rc_disable_pbar
        self.recapture_weight = recapture_weight
        self.default_samples = samples
        self.depth = depth
        self.score_config = score_config
        self.logger = logger

    def get_high_risk_moves(
            self,
            boards: Tuple[chess.Board],
            moves: List[chess.Move],
            time_limit=None,
            samples=None,
            disable_pbar=False
    ):
        if samples is None:
            samples = self.default_samples
        results = {move: 0 for move in moves}
        num_samples = defaultdict(int)
        start_time = time()
        for total in tqdm(range(samples), desc="Zubat: Sampling for gambles", unit="Samples", disable=self.rc_disable_pbar or disable_pbar):
            try:
                board = fast_copy_board(random.choice(boards))
                considered_move: chess.Move = max(
                    moves,
                    key=lambda move: (
                        float("inf") if num_samples[move] == 0 else
                        results[move] + self.exploration_factor * np.sqrt(np.log(total) / num_samples[move])
                    )
                )
                num_samples[considered_move] += 1
                num = num_samples[considered_move]
                for i in range(self.depth):
                    if i == 0:
                        my_move: chess.Move = considered_move
                    else:
                        my_move: chess.Move = random.choice(rbc_legal_moves(board))

                    # If performing a capture move, expect the enemy to attempt to recapture
                    is_capture = board.is_capture(my_move)

                    board.push(my_move)
                    if board.king(board.turn) is None:
                        results[considered_move] = (
                            (num - 1) * results[considered_move] +
                            (self.score_config.capture_king_score + score_material(board, board.turn))
                        ) / num
                        break

                    if is_capture:
                        weights = [self.recapture_weight if m.to_square == my_move.to_square else 1 for m in rbc_legal_moves(board)]
                    else:
                        weights = None

                    opponent_move = random.choices(rbc_legal_moves(board), weights=weights)[0]

                    score = -self.memo_calc_score_risk(board, opponent_move)[0]

                    board.push(opponent_move)

                    results[considered_move] = ((num - 1) * results[considered_move] + score) / num

                    if board.king(board.turn) is None:
                        break
                if time_limit is not None and time() - start_time > time_limit:
                    self.logger.debug('Zubat: Time limit for gamble sampling exceeded')
                    break

            except Exception as e:
                raise e

        # results = {move: np.nan_to_num(np.mean(scores), nan=-1000) for move, scores in results.items()}
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
