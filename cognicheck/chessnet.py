"""A move-prediction network built from scratch: board in, probability for every legal move out.

No hand-made features. The board is seen from the side to move (black positions are mirrored),
as 17 planes of 8x8:

  0-5    the mover's pawns, knights, bishops, rooks, queens, king
  6-11   the opponent's pieces, same order
  12-15  castling rights: mover king side, mover queen side, opponent king side, opponent queen side
  16     en-passant target square

Both ratings enter as two numbers that shift every channel of the first layer.

Optionally (hist > 0, stage C) the network also sees what happened before: the from- and
to-squares of the last `hist` moves as extra planes, and N_CTX context numbers (both clocks, the
time each side spent on its previous move, whether the game has clocks, the move number); and it
predicts as a second output how long the player will think about this move. See context().

Optionally (memory=True, with players) the network also gets the player's personal memory: for
each legal move, how often this player chose it in exactly this position in their training games
(personal_memory()). A small learned function of those counts is added to the move's logit, so
"played Nf3 here 7 times out of 9" can raise Nf3 directly; it starts at zero.

Optionally (n_players > 0) the network also knows who is moving: each player has a learned vector
that shifts the first layer like the ratings, and a learned bias over the 4672 moves (a personal
taste for particular moves, e.g. "likes 1.e4"). Index n_players is a shared slot for anyone else.
Both start at zero, so a network trained without players is the starting point.

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
N_HIST = 8  # moves of history in stage C
FUSED_ATTENTION = False  # torch's fused kernel gives the same result but was slower on Apple MPS (scripts/net_speed.py)
N_CTX = 6
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


def to_square_table() -> np.ndarray:
    """to-square of every move index (-1 where the move would leave the board)."""
    t = np.full(N_MOVES, -1, np.int16)
    for typ in range(N_TYPES):
        for sq in range(64):
            r, f = divmod(sq, 8)
            if typ < 56:
                d, dist = divmod(typ, 7)
                dr, dc = _DIRS[d][0] * (dist + 1), _DIRS[d][1] * (dist + 1)
            elif typ < 64:
                dr, dc = _KNIGHT[typ - 56]
            else:
                dr, dc = 1, (typ - 64) % 3 - 1
            if 0 <= r + dr < 8 and 0 <= f + dc < 8:
                t[typ * 64 + sq] = (r + dr) * 8 + f + dc
    return t


def context(game, ply, clock, legal, off, target, n_hist: int = N_HIST):
    """Stage C inputs for rows that hold the positions of whole games, in order.

    game, ply: (N,) ids; rows of a game are consecutive with increasing ply (positions with a single
    legal move are absent, so their move counts as unknown). clock: (N, 2) mover's and opponent's
    clock before the move, as a fraction of the base time (NaN for games without clocks).
    Returns
      hist  (N, n_hist, 2) uint8: from- and to-square of the move j plies ago, in the current
            mover's frame (255 = unknown)
      ctx   (N, N_CTX) float32: own clock, opponent clock, own previous think, opponent's last
            think, has-clock flag, ply / 100
      think (N,) float32: time this move took, x10 of the base time, clipped to [0, 5] (NaN = unknown)
    """
    n = len(ply)
    key = game.astype(np.int64) * 256 + ply.astype(np.int64)
    if n > 1 and not np.all(np.diff(key) > 0):
        raise ValueError("rows must be whole games in ply order")
    mv = legal[off[:-1] + target].astype(np.int64)
    fr, to = (mv % 64).astype(np.int16), to_square_table()[mv]

    def row(j):  # row of ply - j in the same game, -1 if absent
        want = key - j
        pos = np.searchsorted(key, want).clip(0, max(n - 1, 0))
        return np.where(key[pos] == want, pos, -1)

    hist = np.full((n, n_hist, 2), 255, np.uint8)
    for j in range(1, n_hist + 1):
        r = row(j)
        ok = r >= 0
        f, t = fr[r[ok]], to[r[ok]]
        if j % 2:  # the opponent's move: their frame is mirrored relative to ours
            f, t = f ^ 56, t ^ 56
        hist[ok, j - 1, 0], hist[ok, j - 1, 1] = f, t
    c = clock.astype(np.float32)
    has = ~np.isnan(c).any(1)
    r1, r2, n1 = row(1), row(2), row(-1)
    own_think = np.where((r2 >= 0) & has, c[r2, 0] - c[:, 0], 0.0)
    opp_think = np.where((r1 >= 0) & has, c[r1, 0] - c[:, 1], 0.0)
    ctx = np.stack([np.where(has, c[:, 0], 0.0), np.where(has, c[:, 1], 0.0), np.nan_to_num(own_think).clip(-0.1, 1),
                    np.nan_to_num(opp_think).clip(-0.1, 1), has.astype(np.float32), ply / 100.0], 1).astype(np.float32)
    think = np.where((n1 >= 0) & has, (c[:, 0] - c[n1, 1]) * 10, np.nan).clip(0, 5).astype(np.float32)
    return hist, ctx, think


def personal_memory(tr: dict, te: dict, n_players: int):
    """Counts of the player's own past choices in the same position, per legal move.

    tr, te: dicts with planes, legal, off, target and mover (the player's index for their own
    moves, n_players for anyone else). The table is built from the player's own training moves
    only; a position is identified by its encoded planes (board, side to move, castling rights,
    en passant). Training rows leave their own move out (leave-one-out), so a position seen once
    in training looks exactly like a new position at test time.
    Returns two int16 arrays aligned with tr["legal"] and te["legal"]."""
    from collections import Counter, defaultdict

    table = defaultdict(Counter)
    mv_tr = tr["legal"][tr["off"][:-1] + tr["target"]]
    own_tr = np.nonzero(tr["mover"] < n_players)[0]
    for r in own_tr:
        table[(int(tr["mover"][r]), tr["planes"][r].tobytes())][int(mv_tr[r])] += 1

    def fill(d, rows, loo):
        mem = np.zeros(len(d["legal"]), np.int16)
        for r in rows:
            cnt = table.get((int(d["mover"][r]), d["planes"][r].tobytes()))
            if not cnt:
                continue
            a, b = int(d["off"][r]), int(d["off"][r + 1])
            vals = np.array([cnt.get(int(x), 0) for x in d["legal"][a:b]], np.int16)
            if loo:
                vals[d["target"][r]] -= 1
            mem[a:b] = vals
        return mem

    return fill(tr, own_tr, True), fill(te, np.nonzero(te["mover"] < n_players)[0], False)


def _memory_bias(torch, layer, mem):
    """Logit bonus per move from its memory count c (B, 4672) and the position's total n."""
    n = mem.sum(1, keepdim=True)
    has = (mem > 0).float()
    f = torch.stack([torch.log1p(mem), mem / n.clamp(min=1), has, has * torch.log1p(n)], -1)
    return layer(f).squeeze(-1)


def history_planes(hist: np.ndarray) -> np.ndarray:
    """(B, n_hist, 2) squares -> (B, 2 * n_hist, 8, 8) uint8 planes (from, to for each move)."""
    b = len(hist)
    h = hist.reshape(b, -1)
    out = np.zeros((b, h.shape[1], 64), np.uint8)
    bi, ki = np.nonzero(h != 255)
    out[bi, ki, h[bi, ki]] = 1
    return out.reshape(b, -1, 8, 8)


def elo_input(elo_self: float, elo_oppo: float) -> np.ndarray:
    return np.array([(elo_self - 1500.0) / 500.0, (elo_oppo - 1500.0) / 500.0], dtype=np.float32)


# ---------------------------------------------------------------------------- network (torch)
def build(blocks: int = 6, channels: int = 64, n_players: int = 0, hist: int = 0, arch: str = "resnet", heads: int = 8,
          stem: int = 0, memory: bool = False):
    """arch "resnet": `blocks` residual blocks of `channels` 3x3 convolutions.
    arch "transformer": the 64 squares are tokens of width `channels`; `blocks` pre-norm attention
    layers with `heads` heads, each with a learned bias for every (square, square) pair and head, so
    a layer can relate any two squares (a pin along a diagonal, a back-rank threat) in one step.
    With stem > 0 the transformer first runs that many small residual conv blocks (64 channels), so
    local patterns come cheaply and attention handles the long-range ones (a hybrid)."""
    import torch
    from torch import nn

    if arch == "transformer":
        net = _build_transformer(torch, nn, blocks, channels, n_players, hist, heads, stem, _block_class(torch, nn))
    else:
        net = _build_resnet(torch, nn, _block_class(torch, nn), blocks, channels, n_players, hist)
    if memory:
        net.pl_mem = nn.Linear(4, 1)  # "pl_" prefix: trained with the player parameters
        nn.init.zeros_(net.pl_mem.weight)
        nn.init.zeros_(net.pl_mem.bias)
    net.cfg["memory"] = memory
    return net


def _block_class(torch, nn):
    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
            self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

        def forward(self, x):
            y = torch.relu(self.b1(self.c1(x)))
            return torch.relu(x + self.b2(self.c2(y)))

    return Block


def _build_resnet(torch, nn, Block, blocks, channels, n_players, hist):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = {"blocks": blocks, "channels": channels, "n_players": n_players, "hist": hist}
            self.stem = nn.Conv2d(N_PLANES + 2 * hist, channels, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(channels)
            self.elo = nn.Linear(2, channels)
            self.body = nn.Sequential(*[Block(channels) for _ in range(blocks)])
            self.p1 = nn.Conv2d(channels, channels, 1)
            self.p2 = nn.Conv2d(channels, N_TYPES, 1)
            if hist:
                self.ctx = nn.Linear(N_CTX, channels)
                self.t1, self.t2 = nn.Linear(channels, 64), nn.Linear(64, 1)  # think-time head
            if n_players:
                self.pl_vec = nn.Embedding(n_players + 1, channels)
                self.pl_bias = nn.Embedding(n_players + 1, N_MOVES)
                nn.init.zeros_(self.pl_vec.weight)
                nn.init.zeros_(self.pl_bias.weight)

        def forward(self, x, elo, mask, player=None, ctx=None, aux=False, mem=None):
            """x: (B,17+2*hist,8,8) float, elo: (B,2), mask: (B,4672) bool of legal moves, player: (B,)
            long (only with n_players), ctx: (B,N_CTX) (only with hist) -> masked logits, and with
            aux=True also the predicted think time (B,)."""
            h = self.stem_bn(self.stem(x)) + self.elo(elo)[:, :, None, None]
            if hist:
                h = h + self.ctx(ctx)[:, :, None, None]
            if n_players and player is not None:
                h = h + self.pl_vec(player)[:, :, None, None]
            h = self.body(torch.relu(h))
            logits = self.p2(torch.relu(self.p1(h))).flatten(1)  # (B, 73*64), index = type*64 + square
            if n_players and player is not None:
                logits = logits + self.pl_bias(player)
            logits = logits.float()
            if mem is not None and hasattr(self, "pl_mem"):
                logits = logits + _memory_bias(torch, self.pl_mem, mem)
            logits = logits.masked_fill(~mask, -1e9)
            if aux:
                think = self.t2(torch.relu(self.t1(h.mean((2, 3))))).squeeze(1) if hist else None
                return logits, think
            return logits

    return Net()


def _build_transformer(torch, nn, layers, d, n_players, hist, heads, stem, Block):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.bias = nn.Parameter(torch.zeros(heads, 64, 64))  # learned square-to-square attention bias
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, h):
            b = h.shape[0]
            q, k, v = self.qkv(self.n1(h)).view(b, 64, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            bias = self.bias.to(q.dtype)
            if FUSED_ATTENTION:
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(b, -1, -1, -1))
            else:
                o = torch.softmax(q @ k.transpose(-1, -2) / (d // heads) ** 0.5 + bias, dim=-1) @ v
            h = h + self.out(o.transpose(1, 2).reshape(b, 64, d))
            return h + self.ff(self.n2(h))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = {"blocks": layers, "channels": d, "n_players": n_players, "hist": hist, "arch": "transformer", "heads": heads,
                        "stem": stem}
            n_in = N_PLANES + 2 * hist
            if stem:
                self.conv = nn.Sequential(nn.Conv2d(n_in, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                          *[Block(64) for _ in range(stem)])
                n_in += 64
            self.embed = nn.Linear(n_in, d)
            self.pos = nn.Parameter(torch.randn(64, d) * 0.02)
            self.elo = nn.Linear(2, d)
            self.body = nn.Sequential(*[Layer() for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, N_TYPES)  # per square: logits of its 73 move types
            if hist:
                self.ctx = nn.Linear(N_CTX, d)
                self.t1, self.t2 = nn.Linear(d, 64), nn.Linear(64, 1)
            if n_players:
                self.pl_vec = nn.Embedding(n_players + 1, d)
                self.pl_bias = nn.Embedding(n_players + 1, N_MOVES)
                nn.init.zeros_(self.pl_vec.weight)
                nn.init.zeros_(self.pl_bias.weight)

        def forward(self, x, elo, mask, player=None, ctx=None, aux=False, mem=None):
            if stem:
                x = torch.cat([x, self.conv(x)], 1)
            tok = x.flatten(2).transpose(1, 2)  # (B, 64 squares, planes), square = rank * 8 + file
            h = self.embed(tok) + self.pos + self.elo(elo)[:, None, :]
            if hist:
                h = h + self.ctx(ctx)[:, None, :]
            if n_players and player is not None:
                h = h + self.pl_vec(player)[:, None, :]
            h = self.norm(self.body(h))
            logits = self.head(h).transpose(1, 2).flatten(1)  # (B, 73*64), index = type*64 + square
            if n_players and player is not None:
                logits = logits + self.pl_bias(player)
            logits = logits.float()
            if mem is not None and hasattr(self, "pl_mem"):
                logits = logits + _memory_bias(torch, self.pl_mem, mem)
            logits = logits.masked_fill(~mask, -1e9)
            if aux:
                think = self.t2(torch.relu(self.t1(h.mean(1)))).squeeze(1) if hist else None
                return logits, think
            return logits

    return Net()
