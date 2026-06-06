import chess


PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board) -> list[float]:
    """
    Encodes a chess board into a flat numeric representation.

    Representation:
    - 12 piece planes
    - 64 squares per plane
    - white pieces: planes 0-5
    - black pieces: planes 6-11
    - 1 extra value for side to move

    Total size: 12 * 64 + 1 = 769
    """

    encoded = [0.0] * (12 * 64)

    for square, piece in board.piece_map().items():
        piece_plane = PIECE_TO_PLANE[piece.piece_type]

        if piece.color == chess.BLACK:
            piece_plane += 6

        index = piece_plane * 64 + square
        encoded[index] = 1.0

    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
    encoded.append(side_to_move)

    return encoded