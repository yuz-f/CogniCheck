"""A move-prediction network built from scratch: board in, probability for every legal move out.

No hand-made features. The board is seen from the side to move (black positions are mirrored),
as 17 planes of 8x8:

  0-5    the mover's pawns, knights, bishops, rooks, queens, king
  6-11   the opponent's pieces, same order
  12-15  castling rights: mover king side, mover queen side, opponent king side, opponent queen side
  16     en-passant target square

Both ratings enter as two numbers that shift every channel of the first layer.

Moves use the AlphaZero layout: 73 move types for each of the 64 from-squares (index =
type * 64 + from). Types 0-55 are queen-like moves (8 directions x 7 distances; this includes
pawn moves and queen promotions), 56-63 knight jumps, 64-72 under-promotions (3 directions x
knight, bishop, rook). Illegal moves are masked out, so the network never has to learn the rules;
it only learns which legal move this kind of player chooses.
"""
from __future__ import annotations

import chess
import numpy as np

N_PLANES = 17
N_TYPES = 73
N_MOVES = N_TYPES * 64
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]  # (rank, file) steps
_KNIGHT = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_UNDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _view(board: chess.Board) -> chess.Board:
    """The board from the mover's side: white to move, mover's pieces white."""
    return board if board.turn == chess.WHITE else board.mirror()


def _flip(move: chess.Move, turn: bool) -> chess.Move:
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


def planes(board: chess.Board) -> np.ndarray:
    """(17, 64) bool array for the position."""
    b = _view(board)
    out = np.zeros((N_PLANES, 64), dtype=bool)
    for i, pt in enumerate(PIECES):
        for sq in b.pieces(pt, chess.WHITE):
            out[i, sq] = True
        for sq in b.pieces(pt, chess.BLACK):
            out[6 + i, sq] = True
    out[12, :] = b.has_kingside_castling_rights(chess.WHITE)
    out[13, :] = b.has_queenside_castling_rights(chess.WHITE)
    out[14, :] = b.has_kingside_castling_rights(chess.BLACK)
    out[15, :] = b.has_queenside_castling_rights(chess.BLACK)
    if b.ep_square is not None and b.has_legal_en_passant():
        out[16, b.ep_square] = True
    return out


def move_index(move: chess.Move, turn: bool) -> int:
    """Index of a move in the 4672-way output, from the mover's side."""
    m = _flip(move, turn)
    fr, fc = divmod(m.from_square, 8)
    tr, tc = divmod(m.to_square, 8)
    dr, dc = tr - fr, tc - fc
    if m.promotion is not None and m.promotion != chess.QUEEN:
        t = 64 + 3 * _UNDER.index(m.promotion) + (dc + 1)
        return t * 64 + m.from_square
    if (dr, dc) in _KNIGHT:
        return (56 + _KNIGHT.index((dr, dc))) * 64 + m.from_square
    dist = max(abs(dr), abs(dc))
    step = (dr // dist, dc // dist)
    return (_DIRS.index(step) * 7 + dist - 1) * 64 + m.from_square


def legal_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    return moves, np.array([move_index(m, board.turn) for m in moves], dtype=np.int16)


def mirror_lr_table() -> np.ndarray:
    """Move-index permutation for a left-right (a-file <-> h-file) mirror of the board.

    Valid as data augmentation only for positions without castling rights, where the mirrored
    position is equally legal and strategically equivalent."""
    perm = np.arange(N_MOVES)
    for t in range(N_TYPES):
        for sq in range(64):
            r, f = divmod(sq, 8)
            msq = r * 8 + (7 - f)
            if t < 56:
                d, dist = divmod(t, 7)
                dr, dc = _DIRS[d]
                mt = _DIRS.index((dr, -dc)) * 7 + dist
            elif t < 64:
                dr, dc = _KNIGHT[t - 56]
                mt = 56 + _KNIGHT.index((dr, -dc))
            else:
                piece, dc1 = divmod(t - 64, 3)  # dc1 = dc + 1
                mt = 64 + 3 * piece + (2 - dc1)
            perm[t * 64 + sq] = mt * 64 + msq
    return perm


def elo_input(elo_self: float, elo_oppo: float) -> np.ndarray:
    return np.array([(elo_self - 1500.0) / 500.0, (elo_oppo - 1500.0) / 500.0], dtype=np.float32)


# ---------------------------------------------------------------------------- network (torch)
def build(blocks: int = 6, channels: int = 64):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
            self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

        def forward(self, x):
            y = torch.relu(self.b1(self.c1(x)))
            return torch.relu(x + self.b2(self.c2(y)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = {"blocks": blocks, "channels": channels}
            self.stem = nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(channels)
            self.elo = nn.Linear(2, channels)
            self.body = nn.Sequential(*[Block(channels) for _ in range(blocks)])
            self.p1 = nn.Conv2d(channels, channels, 1)
            self.p2 = nn.Conv2d(channels, N_TYPES, 1)

        def forward(self, x, elo, mask):
            """x: (B,17,8,8) float, elo: (B,2), mask: (B,4672) bool of legal moves -> masked logits."""
            h = self.stem_bn(self.stem(x)) + self.elo(elo)[:, :, None, None]
            h = self.body(torch.relu(h))
            logits = self.p2(torch.relu(self.p1(h))).flatten(1)  # (B, 73*64), index = type*64 + square
            return logits.masked_fill(~mask, -1e9)

    return Net()
