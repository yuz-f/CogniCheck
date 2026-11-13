"""Plan step 4, part 1: cache Maia-2's position summary for every decision.

Maia-2 ends in one linear layer (fc_1: 1024 -> 1880 move scores). For every decision we store
the 1024-number input of that layer (float16) and, for every legal move in the order of the
analysis rows, its index among Maia-2's 1880 moves. A small low-rank correction to that last
layer can then be trained on CPU, per player, without running the network again.

Self-check: the cached summary pushed through fc_1 must reproduce the stored maia_logp feature.

  python scripts/maia_embed.py               # cohort
  python scripts/maia_embed.py --lockbox     # the lockbox players (no evaluation, data only)
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.profile import _elo  # noqa: E402

BATCH = 64
MAX_PLIES = 200
_M = None


def _model():
    global _M
    if _M is None:
        import torch

        from cognicheck.neural import Neural

        torch.set_num_threads(1)
        n = Neural()
        core = n._model.module if hasattr(n._model, "module") else n._model
        core.eval()
        _M = (core, n._prepared)
    return _M


def maia_index(board: chess.Board, move: chess.Move, amd: dict) -> int:
    if board.turn == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
    return amd[move.uci()]


def summaries(core, boards, es, eo):
    import torch

    with torch.no_grad():
        b = torch.stack(boards)
        n = b.size(0)
        x = core.chess_cnn(b.view(n, core.cfg.input_channels, 8, 8))
        x = core.to_patch_embedding(x.view(n, x.size(1), 64)) + core.pos_embedding
        emb = torch.cat((core.elo_embedding(torch.tensor(es)), core.elo_embedding(torch.tensor(eo))), dim=1)
        x = core.last_ln(core.transformer(x, emb).mean(dim=1))
    return x.numpy()


def player(p, games_dir, analysis_dir):
    from maia2.inference import preprocessing

    out = analysis_dir / f"{p['id']}.emb.npz"
    if out.exists():
        return p["id"], "cached"
    core, (amd, ed, _) = _model()
    z = np.load(analysis_dir / f"{p['id']}.npz")
    starts = z["starts"]
    sizes = np.diff(np.append(starts, len(z["X"])))
    pgs = games_of(parse_pgn((games_dir / f"{p['id']}.pgn").read_text()), p["username"])
    E, MI, buf = [], [], []

    def flush():
        if buf:
            E.append(summaries(core, [b[0] for b in buf], [b[1] for b in buf], [b[2] for b in buf]).astype(np.float16))
            buf.clear()

    k = 0
    for pg in pgs:
        h = pg.game.headers
        es = _elo(h, "White" if pg.color else "Black") or p["rating"]
        eo = _elo(h, "Black" if pg.color else "White") or es
        board = pg.game.board()
        for ply, node in enumerate(pg.game.mainline()):
            if ply >= MAX_PLIES:
                break
            if board.turn == pg.color and board.legal_moves.count() > 1:
                moves = list(board.legal_moves)
                if k >= len(sizes) or len(moves) != sizes[k]:
                    raise RuntimeError(f"{p['id']}: decision {k} has {len(moves)} moves, analysis has {sizes[k] if k < len(sizes) else 'none'}")
                MI.extend(maia_index(board, m, amd) for m in moves)
                bi, e1, e2, _ = preprocessing(board.fen(), int(es), int(eo), ed, amd)
                buf.append((bi, e1, e2))
                if len(buf) == BATCH:
                    flush()
                k += 1
            board.push(node.move)
    flush()
    if k != len(starts):
        raise RuntimeError(f"{p['id']}: {k} decisions vs {len(starts)} in the analysis")
    E = np.concatenate(E)
    MI = np.array(MI, dtype=np.int16)
    # self-check against the stored Maia-2 feature (set from ply 10 on, clipped at -6)
    W = core.fc_1.weight.detach().numpy()
    bias = core.fc_1.bias.detach().numpy()
    X = z["X"]
    from cognicheck.features import IDX

    maia_col = np.load(analysis_dir / f"{p['id']}.maia.npy")[:, 0] if (analysis_dir / f"{p['id']}.maia.npy").exists() else X[:, IDX["maia_logp"]]
    errs = []
    for j in np.linspace(0, len(starts) - 1, 40).astype(int):
        s, e = starts[j], starts[j] + sizes[j]
        if not np.any(maia_col[s:e] != 0):
            continue
        logits = E[j].astype(np.float32) @ W[MI[s:e]].T + bias[MI[s:e]]
        lp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        # the stored cohort column is log p clipped at -6 and scaled by 1/3
        errs.append(np.abs(np.maximum(lp, -6.0) / 3.0 - maia_col[s:e]).max())
    np.savez(out, E=E, MI=MI)
    return p["id"], f"{len(E)} decisions, max log-prob mismatch {max(errs) if errs else float('nan'):.3f} on {len(errs)} checks"


if __name__ == "__main__":
    if "--lockbox" in sys.argv:
        sys.path.insert(0, "scripts")
        import lockbox

        players, gdir, adir = lockbox.players(), lockbox.GAMES, lockbox.ANALYSIS
    else:
        players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
        gdir, adir = cohort.GAMES, cohort.ANALYSIS
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    if nums:
        players = players[: int(nums[0])]
    t0 = time.time()
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(player, p, gdir, adir): p for p in players}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                pid, msg = f.result()
                print(f"  {i}/{len(players)} {pid}: {msg} ({time.time() - t0:.0f}s)", file=sys.stderr)
            except Exception as e:
                print(f"  FAILED {futs[f]['id']}: {e}", file=sys.stderr)
