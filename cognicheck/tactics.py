"""Cheap, engine-free tactical primitives.

Everything here is written from the point of view of "what can a human
plausibly see": static exchange evaluation, hanging pieces, simple forks and
mates in one, and a vocabulary of *motifs* (knight jump, backward move, long
diagonal, ...) that we later use to describe a player's blind spots.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# Motif vocabulary used for blind-spot statistics. Order matters for display.
MOTIFS = [
    "knight",
    "bishop",
    "rook",
    "queen",
    "pawn",
    "king",
    "backward",
    "lateral",
    "long_range",
    "long_diagonal",
    "check",
    "fork",
    "discovered",
    "capture",
    "quiet",
]

MOTIF_LABEL = {
    "knight": "knight moves",
    "bishop": "bishop moves",
    "rook": "rook moves",
    "queen": "queen moves",
    "pawn": "pawn moves",
    "king": "king moves",
    "backward": "backward (retreating) moves",
    "lateral": "sideways moves along a rank",
    "long_range": "long-range moves (4+ squares)",
    "long_diagonal": "long diagonals (3+ squares)",
    "check": "checks",
    "fork": "forks / double attacks",
    "discovered": "discovered attacks",
    "capture": "captures",
    "quiet": "quiet (non-capturing) moves",
}


def piece_value(board: chess.Board, square: chess.Square) -> int:
    p = board.piece_at(square)
    return VALUE[p.piece_type] if p else 0


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return 1
    return piece_value(board, move.to_square)


def _see_square(board: chess.Board, square: chess.Square, depth: int = 0) -> int:
    """Best gain (>= 0) for the side to move from capturing on ``square``.

    The side to move may also decline to capture, hence the ``max(0, ...)``.
    Uses legal moves so pins are respected.
    """
    if depth > 12:
        return 0
    best_move = None
    best_val = 10**6
    for m in board.generate_legal_captures(to_mask=chess.BB_SQUARES[square]):
        attacker = board.piece_at(m.from_square)
        v = VALUE[attacker.piece_type]
        if m.promotion:
            v -= VALUE[m.promotion] - 1  # promoting captures are extra attractive
        if v < best_val:
            best_val, best_move = v, m
    if best_move is None:
        return 0
    gain = _captured_value(board, best_move)
    if best_move.promotion:
        gain += VALUE[best_move.promotion] - 1
    board.push(best_move)
    reply = _see_square(board, square, depth + 1)
    board.pop()
    return max(0, gain - reply)


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of ``move`` for the mover (in pawn units).

    For a capture, the material won after the full exchange sequence on the
    target square. For a quiet move, minus the material the moved piece loses
    if it can be captured there.
    """
    gain = _captured_value(board, move) if board.is_capture(move) else 0
    if move.promotion:
        gain += VALUE[move.promotion] - 1
    board.push(move)
    loss = _see_square(board, move.to_square)
    board.pop()
    return gain - loss


@dataclass
class Capture:
    move: chess.Move
    gain: int


def winning_captures(board: chess.Board, min_gain: int = 1) -> list[Capture]:
    """Captures for the side to move that win at least ``min_gain`` pawns."""
    out = []
    for m in board.generate_legal_captures():
        g = see(board, m)
        if g >= min_gain:
            out.append(Capture(m, g))
    out.sort(key=lambda c: -c.gain)
    return out


def threats_against_side_to_move(board: chess.Board, min_gain: int = 1) -> list[Capture]:
    """What the opponent could win if the side to move passed.

    Returned moves are opponent moves (valid on the null-moved board).
    """
    if board.is_check():
        return []
    board.push(chess.Move.null())
    try:
        return winning_captures(board, min_gain)
    finally:
        board.pop()


def mates_in_one(board: chess.Board) -> list[chess.Move]:
    out = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            out.append(m)
        board.pop()
    return out


def mate_threat(board: chess.Board) -> list[chess.Move]:
    """Opponent mates-in-one that exist if the side to move passed."""
    if board.is_check():
        return []
    board.push(chess.Move.null())
    try:
        return mates_in_one(board)
    finally:
        board.pop()


def max_opponent_gain(board: chess.Board) -> tuple[int, chess.Move | None]:
    """After a move was pushed: best SEE capture for the side to move now."""
    best, best_m = 0, None
    for m in board.generate_legal_captures():
        g = see(board, m)
        if g > best:
            best, best_m = g, m
    return best, best_m


def _relative_dy(color: chess.Color, move: chess.Move) -> int:
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    return dy if color == chess.WHITE else -dy


def _valuable_targets(board: chess.Board, square: chess.Square, color: chess.Color) -> list[chess.Square]:
    """Enemy pieces attacked from ``square`` that are worth hitting.

    A target counts if it is the king, more valuable than the attacker, or
    undefended.
    """
    attacker = board.piece_at(square)
    if attacker is None:
        return []
    av = VALUE[attacker.piece_type]
    out = []
    for t in board.attacks(square):
        p = board.piece_at(t)
        if p is None or p.color == color:
            continue
        if p.piece_type == chess.KING or VALUE[p.piece_type] > av or not board.is_attacked_by(not color, t):
            out.append(t)
    return out


def motifs(board: chess.Board, move: chess.Move) -> set[str]:
    """Describe ``move`` (legal on ``board``) with the motif vocabulary."""
    color = board.turn
    piece = board.piece_at(move.from_square)
    tags: set[str] = set()
    if piece is None:  # null move or bad input
        return tags
    tags.add(PIECE_NAME[piece.piece_type])
    dy = _relative_dy(color, move)
    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dist = chess.square_distance(move.from_square, move.to_square)
    if board.is_castling(move):
        tags.add("lateral")
    elif dy < 0:
        tags.add("backward")
    elif dy == 0:
        tags.add("lateral")
    if dist >= 4:
        tags.add("long_range")
    if abs(dx) == abs(dy) and dist >= 3 and piece.piece_type in (chess.BISHOP, chess.QUEEN):
        tags.add("long_diagonal")
    tags.add("capture" if board.is_capture(move) else "quiet")
    if board.gives_check(move):
        tags.add("check")

    before = {
        s: set(_valuable_targets(board, s, color))
        for s in board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color)
        if s != move.from_square
    }
    board.push(move)
    try:
        targets = _valuable_targets(board, move.to_square, color)
        if len(targets) >= 2:
            tags.add("fork")
        for s, old in before.items():
            if board.piece_at(s) is None or board.piece_at(s).color != color:
                continue
            new = set(_valuable_targets(board, s, color)) - old
            if new:
                tags.add("discovered")
                break
    finally:
        board.pop()
    return tags


def describe_move(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:  # pragma: no cover - defensive
        return move.uci()
