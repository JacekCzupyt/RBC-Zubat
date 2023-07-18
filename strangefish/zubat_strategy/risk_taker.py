import random
from time import time
from typing import List, Tuple

import chess
import numpy as np
from tqdm import tqdm

from strangefish.utilities import fast_copy_board, rbc_legal_moves
from strangefish.utilities.rbc_move_score import ScoreConfig, calculate_score, score_material


def get_high_risk_moves(
        engine,
        boards: Tuple[chess.Board],
        moves: List[chess.Move],
        time_limit=None,
        depth=1,
        samples=5000,
        score_config: ScoreConfig = ScoreConfig(capture_king_score=100, checkmate_score=90, into_check_score=-100, remain_in_check_penalty=-20, op_into_check_score=-40),
        rc_disable_pbar=False,
):
    results = {move: [] for move in moves}
    start_time = time()
    for _ in tqdm(range(samples), desc="Sampling for gambles", unit="Samples", disable=rc_disable_pbar):
        try:
            board = fast_copy_board(random.choice(boards))
            move = random.choice(moves)
            for i in range(depth):
                if i == 0:
                    my_move = move
                else:
                    my_move = random.choice(rbc_legal_moves(board))
                # my_taken_move = simulate_move(board, my_move)
                board.push(my_move)
                if board.king(board.turn) is None:
                    results[move].append(score_config.capture_king_score + score_material(board, board.turn))
                    break

                opponent_move = random.choice(rbc_legal_moves(board))

                # print(board)
                # print(opponent_move)
                # print('')

                score = -calculate_score(engine, board, move=opponent_move, is_op_turn=False)
                # opponent_taken_move = simulate_move(board, opponent_move)
                board.push(opponent_move)

                results[move].append(score)
            if time_limit is not None and time() - start_time > time_limit:
                print('Time limit for gamble sampling exceeded')
                break

        except Exception as e:
            raise e

    # TODO: what if no samples?
    results = {move: np.nan_to_num(np.mean(scores), nan=-1000) for move, scores in results.items()}
    return results
