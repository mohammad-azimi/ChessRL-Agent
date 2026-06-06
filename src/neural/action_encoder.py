import chess


ACTION_SPACE_SIZE = 64 * 64


def move_to_action(move: chess.Move) -> int:
    """
    Converts a chess move into an action index.

    Current version uses from_square and to_square only:
    action = from_square * 64 + to_square

    This gives 4096 possible actions.
    Promotions are simplified for now.
    """

    return move.from_square * 64 + move.to_square


def action_to_move(action: int, board: chess.Board) -> chess.Move | None:
    """
    Converts an action index back to a legal chess move.

    If multiple legal moves share the same from-to squares,
    queen promotion is preferred.
    """

    from_square = action // 64
    to_square = action % 64

    matching_moves = [
        move for move in board.legal_moves
        if move.from_square == from_square and move.to_square == to_square
    ]

    if not matching_moves:
        return None

    queen_promotions = [
        move for move in matching_moves
        if move.promotion == chess.QUEEN
    ]

    if queen_promotions:
        return queen_promotions[0]

    return matching_moves[0]


def get_legal_action_indices(board: chess.Board) -> list[int]:
    return [move_to_action(move) for move in board.legal_moves]