import logging
from typing import List, Optional

import chess
from reconchess import Square, GameHistory

from strangefish.strangefish_mht_core import StrangeFish, RC_DISABLE_PBAR


class UncertaintyPlaybackPlayer(StrangeFish):

    def __init__(self, log_to_file=True, game_id=None, rc_disable_pbar=True, stream_log_level=logging.CRITICAL,
                 max_pre_sense_boards = 100000, max_post_sense_boards = 10000):
        super().__init__(log_to_file, stream_log_level, game_id, rc_disable_pbar)

        self.max_post_sense_boards = max_post_sense_boards
        self.max_pre_sense_boards = max_pre_sense_boards
        self.sense_actions = None
        self.move_actions = None
        self.board_states_pre_sense = []
        self.board_states_post_sense = []

    def handle_game_start(self, color: chess.Color, board: chess.Board, opponent_name: str):
        super().handle_game_start(color, board, opponent_name)

    def sense_strategy(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        self.board_states_pre_sense.append(len(self.boards))
        if len(self.boards) > self.max_pre_sense_boards:
            raise Exception("Too many boards")
        return None

    def move_strategy(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.board_states_post_sense.append(len(self.boards))
        if len(self.boards) > self.max_post_sense_boards:
            raise Exception("Too many boards")
        return None


