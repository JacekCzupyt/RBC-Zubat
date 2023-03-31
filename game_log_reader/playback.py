import chess
from reconchess import LocalGame, GameHistory, Player, Color


def playback(game_history: GameHistory, player: Player, color: Color):
    game = LocalGame()

    opponent_name = game_history.get_white_player_name()
    if color == chess.WHITE:
        opponent_name = game_history.get_black_player_name()
    player.handle_game_start(color, game.board.copy(), opponent_name)
    game.start()

    turn = game_history.first_turn()

    while not game.is_over() and turn < game_history.last_turn():
        opt_capture_square = game.opponent_move_results()
        if game.turn == color:
            player.handle_opponent_move_result(opt_capture_square is not None, opt_capture_square)

        sense_actions = game.sense_actions()
        move_actions = game.move_actions()

        sense = game_history.sense(turn)
        if game.turn == color:
            player.choose_sense(sense_actions, move_actions, game.get_seconds_left())
        sense_result = game.sense(sense)
        if game.turn == color:
            player.handle_sense_result(sense_result)

        move = game_history.requested_move(turn)
        if game.turn == color:
            player.choose_move(move_actions, game.get_seconds_left())
        requested_move, taken_move, opt_enemy_capture_square = game.move(move)
        if game.turn == color:
            player.handle_move_result(requested_move, taken_move,
                                      opt_enemy_capture_square is not None, opt_enemy_capture_square)

        game.end_turn()
        turn = turn.next

    game.end()
    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    player.handle_game_end(winner_color, win_reason, game_history)