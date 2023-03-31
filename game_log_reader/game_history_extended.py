import json

import chess
from reconchess import GameHistory, ChessJSONEncoder, ChessJSONDecoder, GameHistoryEncoder, GameHistoryDecoder


class GameHistoryExtended(GameHistory):
    def __init__(self):
        super().__init__()

        self._pre_sense_uncertainty = {chess.WHITE: None, chess.BLACK: None}
        self._post_sense_uncertainty = {chess.WHITE: None, chess.BLACK: None}

    def __str__(self):
        return json.dumps(self, cls=GameHistoryExtendedEncoder, indent=4)

    def save(self, filename):
        with open(filename, 'w') as fp:
            fp.write(str(self))

    @classmethod
    def from_file(cls, filename):
        with open(filename) as fp:
            return json.load(fp, cls=GameHistoryExtendedDecoder)

    def get_player_name(self, color: chess.Color):
        if color == chess.WHITE:
            return self.get_white_player_name()
        else:
            return self.get_black_player_name()



class GameHistoryExtendedEncoder(GameHistoryEncoder):
    def default(self, o):
        if isinstance(o, GameHistoryExtended):
            return {
                'type': 'GameHistoryExtended',
                'white_name': o._white_name,
                'black_name': o._black_name,
                'senses': o._senses,
                'sense_results': o._sense_results,
                'requested_moves': o._requested_moves,
                'taken_moves': o._taken_moves,
                'capture_squares': o._capture_squares,
                'fens_before_move': o._fens_before_move,
                'fens_after_move': o._fens_after_move,
                'winner_color': o._winner_color,
                'win_reason': o._win_reason,
                'pre_sense_uncertainty': o._pre_sense_uncertainty,
                'post_sense_uncertainty': o._post_sense_uncertainty,
            }
        return super().default(o)


class GameHistoryExtendedDecoder(GameHistoryDecoder):
    def _object_hook(self, obj):
        if 'type' in obj and (obj['type'] == 'GameHistory' or obj['type'] == 'GameHistoryExtended'):
            for key in ['senses', 'sense_results', 'requested_moves', 'taken_moves', 'capture_squares',
                        'fens_before_move', 'fens_after_move']:
                obj[key] = {True: obj[key]['true'], False: obj[key]['false']}
            history = GameHistoryExtended()
            history._white_name = obj['white_name']
            history._black_name = obj['black_name']
            history._senses = obj['senses']
            history._sense_results = obj['sense_results']
            history._requested_moves = obj['requested_moves']
            history._taken_moves = obj['taken_moves']
            history._capture_squares = obj['capture_squares']
            history._fens_before_move = obj['fens_before_move']
            history._fens_after_move = obj['fens_after_move']
            history._winner_color = obj['winner_color']
            history._win_reason = obj['win_reason']

            for color, sense_results in history._sense_results.items():
                for result in sense_results:
                    for i in range(len(result)):
                        result[i] = tuple(result[i])

            if obj['type'] == 'GameHistoryExtended':
                for key in ['pre_sense_uncertainty', 'post_sense_uncertainty']:
                    obj[key] = {True: obj[key]['true'], False: obj[key]['false']}
                history._pre_sense_uncertainty = obj['pre_sense_uncertainty']
                history._post_sense_uncertainty = obj['post_sense_uncertainty']

            return history

        return super()._object_hook(obj)