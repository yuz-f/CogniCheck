import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # also works when run as a plain file
from cognicheck import chessnet  # noqa: E402


def _random_positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randrange(0, 120)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            out.append(b)
    return out


def test_move_indices_unique_and_in_range():
    for b in _random_positions(300):
        _, idx = chessnet.legal_indices(b)
        assert len(set(idx.tolist())) == len(idx)
        assert idx.min() >= 0 and idx.max() < chessnet.N_MOVES


def test_special_moves():
    b = chess.Board("r3k2r/1P6/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
    _, idx = chessnet.legal_indices(b)
    assert len(set(idx.tolist())) == len(idx)
    for uci in ("e1g1", "e1c1", "e5d6", "b7b8q", "b7b8n", "b7a8r"):
        assert chess.Move.from_uci(uci) in b.legal_moves
    # black's mirror: the same moves seen from black's side get the same indices
    bm = b.mirror()
    for uci in ("e1g1", "b7b8n"):
        m = chess.Move.from_uci(uci)
        mm = chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
        assert chessnet.move_index(m, chess.WHITE) == chessnet.move_index(mm, chess.BLACK)


def test_planes_mirror():
    b = chess.Board()
    b.push_san("e4")
    p_black = chessnet.planes(b)  # black to move, seen mirrored
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").mirror()
    assert np.array_equal(p_black[:12], chessnet.planes(b2)[:12])
    assert p_black[0].sum() == 8 and p_black[6].sum() == 8


def test_mirror_lr_table():
    perm = chessnet.mirror_lr_table()
    assert (perm[perm] == np.arange(chessnet.N_MOVES)).all()
    checked = 0
    for b in _random_positions(300, seed=1):
        if b.has_castling_rights(chess.WHITE) or b.has_castling_rights(chess.BLACK):
            continue
        bm = b.transform(chess.flip_horizontal)
        _, idx = chessnet.legal_indices(b)
        _, idxm = chessnet.legal_indices(bm)
        assert set(perm[idx].tolist()) == set(idxm.tolist())
        assert (chessnet.planes(b).reshape(17, 8, 8)[:, :, ::-1] == chessnet.planes(bm).reshape(17, 8, 8)).all()
        checked += 1
    assert checked > 50


if __name__ == "__main__":  # lets an editor's Run button run these tests
    import pytest

    root = Path(__file__).resolve().parent.parent
    sys.exit(pytest.main(["-q", str(Path(__file__)), "--rootdir", str(root)]))
