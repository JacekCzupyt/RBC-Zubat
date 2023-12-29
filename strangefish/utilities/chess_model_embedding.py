import chess
import numpy as np
from reconchess import Turn

from game_log_reader.game_history_extended import GameHistoryExtended


def int_to_square(squares):
    board = np.zeros(64)
    if squares is not None:
        board[squares] = 1
    return board.reshape((8, 8))


def parse_position(board, color):
    return [np.reshape([board.pieces_mask(piece_type, color) >> i & 1 for i in range(64)], (8, 8)) for piece_type in
            chess.PIECE_TYPES]


def chess_model_embedding(
        color,
        previous_requested_move,
        previous_move_piece_type,
        previous_move_result,
        last_move_capture,
        opponent_capture,
        sense_position,
        sense_result,
        owned_pieces,
        next_requested_move
):
    try:
        # 1 Color Layer
        color_layer = np.ones((1, 8, 8)) if color else np.zeros((1, 8, 8))
        # 1 Sense Location Layer
        sense_location = np.array([int_to_square(sense_position)])
        # 6 Sense Result Layers - one for each piece type
        sense_res = np.array([
            int_to_square([
                result[0] for result in sense_result if
                (result[1] is not None and result[1].piece_type == piece_type)
            ]) for piece_type in chess.PIECE_TYPES
        ])

        # 1 Layer for piece captured on previous turn
        opponent_capture_square = [int_to_square(opponent_capture)]

        # 6 Layers for current state of owned pieces
        owned_piece_positions = np.array([
            np.reshape([owned_pieces.pieces_mask(piece_type, color) >> i & 1 for i in range(64)],
                       (8, 8))
            for piece_type in chess.PIECE_TYPES
        ])

        # 6 Layers for last move origin - one for each piece type
        last_move_origin = np.zeros((6, 8, 8))

        # 1 Layer for last move requested destination
        last_move_requested_destination = np.zeros((1, 8, 8))

        # 3 Layers for requested under-promotions
        under_promotions = np.zeros((3, 8, 8))

        # 1 Layer for last move taken destination
        last_move_taken_destination = np.zeros((1, 8, 8))

        # 1 Layer for captured piece
        captured_square = np.zeros((1, 8, 8))

        if previous_requested_move is not None:
            if previous_requested_move is not None:
                # TODO
                last_move_origin[chess.PIECE_TYPES.index(previous_move_piece_type)] = int_to_square(
                    previous_requested_move.from_square)
                last_move_requested_destination = np.array([int_to_square(previous_requested_move.to_square)])

                if previous_requested_move.promotion not in [None, chess.QUEEN]:
                    under_promotions[
                        [chess.ROOK, chess.KNIGHT, chess.BISHOP].index(previous_requested_move.promotion)] = np.ones((8, 8))

            if last_move_capture is not None:
                captured_square = [int_to_square(last_move_capture)]
            if previous_move_result is not None:
                last_move_taken_destination = np.array([int_to_square(previous_move_result.to_square)])

        if next_requested_move is not None and next_requested_move != chess.Move.null():
            next_piece_type = owned_pieces.piece_at(next_requested_move.from_square).piece_type

            # 6 Layers for next move origin - one for each piece type
            next_move_origin = np.array([
                int_to_square(next_requested_move.from_square if next_piece_type == piece_type else None)
                for piece_type in chess.PIECE_TYPES
            ])

            # 1 Layer for last move requested destination
            next_move_requested_destination = np.array([int_to_square(next_requested_move.to_square)])

            # 3 Layers for attempted underpromotions
            next_under_promotions = np.zeros((3, 8, 8))
            if next_requested_move.promotion not in [None, chess.QUEEN]:
                under_promotions[[chess.ROOK, chess.KNIGHT, chess.BISHOP].index(next_requested_move.promotion)] = np.ones(
                    (8, 8))
        else:
            next_move_origin = np.zeros((6, 8, 8))
            next_move_requested_destination = np.zeros((1, 8, 8))
            next_under_promotions = np.zeros((3, 8, 8))

        return np.transpose(np.concatenate(
            (color_layer,  # 0
             sense_location,  # 1
             sense_res,  # 2-7
             opponent_capture_square,  # 8
             last_move_origin,  # 9-14
             last_move_requested_destination,  # 15
             last_move_taken_destination,  # 16
             captured_square,  # 17
             under_promotions,  # 18-20
             owned_piece_positions,  # 21-26
             next_move_origin,  # 27-32
             next_move_requested_destination,  # 33
             next_under_promotions,  # 34-36
             )
        ), (1, 2, 0))
    except Exception as e:
        raise Exception(owned_pieces, next_requested_move)


def map_move_from_history(history: GameHistoryExtended, turn: Turn):
    color = turn.color
    move = turn.turn_number

    # 1 Color Layer
    color_layer = np.ones((1, 8, 8)) if color else np.zeros((1, 8, 8))
    # 1 Sense Location Layer
    sense_location = np.array([int_to_square(history.sense(turn))])
    # 6 Sense Result Layers - one for each piece type
    sense_res = np.array([
        int_to_square([
            result[0] for result in history.sense_result(turn) if
            (result[1] is not None and result[1].piece_type == piece_type)
        ]) for piece_type in chess.PIECE_TYPES
    ])

    # 1 Layer for piece captured on previous turn
    opponent_capture_square = np.zeros((1, 8, 8)) if turn == Turn(chess.WHITE, 0) else \
        np.array([int_to_square(history.capture_square(turn.previous))])

    # 6 Layers for current state of owned pieces
    owned_piece_positions = np.array([
        np.reshape([history.truth_board_before_move(turn).pieces_mask(piece_type, color) >> i & 1 for i in range(64)],
                   (8, 8))
        for piece_type in chess.PIECE_TYPES
    ])

    # 6 Layers for last move origin - one for each piece type
    last_move_origin = np.zeros((6, 8, 8))

    # 1 Layer for last move requested destination
    last_move_requested_destination = np.zeros((1, 8, 8))

    # 3 Layers for requested under-promotions
    under_promotions = np.zeros((3, 8, 8))

    # 1 Layer for last move taken destination
    last_move_taken_destination = np.zeros((1, 8, 8))

    # 1 Layer for captured piece
    captured_square = np.zeros((1, 8, 8))

    if move > 0:
        prev_turn = turn.previous.previous
        prev_requested_move = history.requested_move(prev_turn)
        if prev_requested_move is not None:
            prev_piece_type = history.truth_board_before_move(prev_turn).piece_at(
                prev_requested_move.from_square).piece_type
            last_move_origin[chess.PIECE_TYPES.index(prev_piece_type)] = int_to_square(prev_requested_move.from_square)
            last_move_requested_destination = np.array([int_to_square(prev_requested_move.to_square)])

            if prev_requested_move.promotion not in [None, chess.QUEEN]:
                under_promotions[
                    [chess.ROOK, chess.KNIGHT, chess.BISHOP].index(prev_requested_move.promotion)] = np.ones((8, 8))

        if history.capture_square(prev_turn) is not None:
            captured_square = [int_to_square(history.capture_square(prev_turn))]
        taken_move = history.taken_move(prev_turn)
        if taken_move is not None:
            last_move_taken_destination = np.array([int_to_square(taken_move.to_square)])

    next_requested_move = history.requested_move(turn)
    if next_requested_move is not None and next_requested_move != chess.Move.null():
        next_piece_type = history.truth_board_before_move(turn).piece_at(next_requested_move.from_square).piece_type

        # 6 Layers for next move origin - one for each piece type
        next_move_origin = np.array([
            int_to_square(next_requested_move.from_square if next_piece_type == piece_type else None)
            for piece_type in chess.PIECE_TYPES
        ])

        # 1 Layer for last move requested destination
        next_move_requested_destination = np.array([int_to_square(next_requested_move.to_square)])

        # 3 Layers for attempted underpromotions
        next_under_promotions = np.zeros((3, 8, 8))
        if next_requested_move.promotion not in [None, chess.QUEEN]:
            under_promotions[[chess.ROOK, chess.KNIGHT, chess.BISHOP].index(next_requested_move.promotion)] = np.ones(
                (8, 8))
    else:
        next_move_origin = np.zeros((6, 8, 8))
        next_move_requested_destination = np.zeros((1, 8, 8))
        next_under_promotions = np.zeros((3, 8, 8))

    return np.concatenate(
        (color_layer,  # 0
         sense_location,  # 1
         sense_res,  # 2-7
         opponent_capture_square,  # 8
         last_move_origin,  # 9-14
         last_move_requested_destination,  # 15
         last_move_taken_destination,  # 16
         captured_square,  # 17
         under_promotions,  # 18-20
         owned_piece_positions,  # 21-26
         next_move_origin,  # 27-32
         next_move_requested_destination,  # 33
         next_under_promotions,  # 34-36
         )
    )


def map_game_from_history(history: GameHistoryExtended, color: chess.Color):
    if history._post_sense_uncertainty[not color] is None:
        return None

    measured_turns = len(history._post_sense_uncertainty[not color]) - int(not color)

    if measured_turns<=0:
        return None

    turns = [Turn(color, move) for move in range(measured_turns)]

    X = np.array([map_move_from_history(history, turn) for turn in turns])
    X = np.transpose(X, (0, 2, 3, 1))

    y = np.array(history._post_sense_uncertainty[not color][int(not color):])
    y = np.log(y)/12.0

    return X, y
