"""Stage A data: encode the cohort's games for the from-scratch network (cognicheck/chessnet.py).

  train  every move of both sides in the cohort's training games
  test   each player's own decisions in their held-out games (the most recent 15%), exactly the
         positions scripts/evaluate.py scores, with phase and blunder labels for the breakdown

Leakage guard: a game is never used for training if it is held out for any cohort player or
appears in any lockbox player's games (matched by the game's Site URL), so a game shared by two
players cannot leak.

Writes data/real/net/cohort.npz.

  python scripts/net_data.py
"""
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np

sys.path.insert(0, ".")
from cognicheck import chessnet, cohort  # noqa: E402
from cognicheck.features import IDX, phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402

OUT = cohort.REAL / "net"
MAX_PLIES = 200
PHASES = {"opening": 0, "middlegame": 1, "endgame": 2}


def _elo(h, color):
    try:
        return float(h.get(color + "Elo", "1500"))
    except ValueError:
        return 1500.0


def encode_game(game, keep):
    """Encode positions of one game. keep(ply, board) -> bool selects positions."""
    rec = []
    h = game.headers
    board = game.board()
    for ply, node in enumerate(game.mainline()):
        if ply >= MAX_PLIES:
            break
        if board.legal_moves.count() > 1 and keep(ply, board):
            moves, idx = chessnet.legal_indices(board)
            tgt = moves.index(node.move)
            w, b = _elo(h, "White"), _elo(h, "Black")
            es, eo = (w, b) if board.turn == chess.WHITE else (b, w)
            rec.append((np.packbits(chessnet.planes(board)), chessnet.elo_input(es, eo), idx, tgt, ply, board.copy(stack=False)))
        board.push(node.move)
    return rec


def site(g):
    return g.headers.get("Site", "")


def player_data(p, exclude):
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    hold = cohort.holdout_games(p["id"])
    z = np.load(cohort.ANALYSIS / f"{p['id']}.npz")
    eng_loss = z["X"][:, IDX["eng_loss"]]
    starts, chosen = z["starts"], z["chosen"]
    train, test = [], []
    k = 0  # decision counter over all games, aligned with the analysis rows
    for gi, pg in enumerate(pgs):
        color = pg.color
        if gi in hold:
            for planes, elo, idx, tgt, ply, board in encode_game(pg.game, lambda ply, b: b.turn == color):
                test.append((planes, elo, idx, tgt, PHASES[phase(board)], bool(eng_loss[chosen[k]] >= 2), k))
                k += 1
        else:
            recs = encode_game(pg.game, lambda ply, b: True)
            k += sum(1 for r in recs if r[5].turn == color)
            if site(pg.game) not in exclude:
                train.extend((planes, elo, idx, tgt, site(pg.game)) for planes, elo, idx, tgt, ply, board in recs)
    if k != len(starts):
        raise RuntimeError(f"{p['id']}: {k} decisions vs {len(starts)} in the analysis")
    return p["id"], train, test


def pack(rows, extra):
    n = len(rows)
    planes = np.stack([r[0] for r in rows]) if n else np.zeros((0, 136), np.uint8)
    elo = np.stack([r[1] for r in rows]).astype(np.float16) if n else np.zeros((0, 2), np.float16)
    off = np.zeros(n + 1, np.int64)
    off[1:] = np.cumsum([len(r[2]) for r in rows])
    legal = np.concatenate([r[2] for r in rows]).astype(np.int16) if n else np.zeros(0, np.int16)
    tgt = np.array([r[3] for r in rows], np.int16)
    out = {"planes": planes, "elo": elo, "off": off, "legal": legal, "target": tgt}
    out.update(extra)
    return out


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    exclude = set()
    for p in players:
        pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
        exclude |= {site(pgs[i].game) for i in cohort.holdout_games(p["id"])}
    sys.path.insert(0, "scripts")
    import lockbox

    for p in lockbox.players():
        f = lockbox.GAMES / f"{p['id']}.pgn"
        if f.exists():
            exclude |= {site(g.game) for g in games_of(parse_pgn(f.read_text()), p["username"])}
    exclude.discard("")
    print(f"{len(exclude)} games excluded from training (held out or lockbox)", file=sys.stderr)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(player_data, players, [exclude] * len(players)))
    seen, train, test = set(), [], []
    for pi, (pid, tr, te) in enumerate(res):
        new_sites = {r[4] for r in tr} - seen
        train.extend(r for r in tr if r[4] in new_sites)  # each game once, even if two cohort players share it
        seen |= new_sites
        test.extend((*r, pi) for r in te)
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pack(train, {})
    te = pack(test, {"phase": np.array([r[4] for r in test], np.int8), "blunder": np.array([r[5] for r in test]),
                     "decision": np.array([r[6] for r in test], np.int32), "player": np.array([r[7] for r in test], np.int16)})
    np.savez(OUT / "cohort.npz", **{"train_" + k: v for k, v in tr.items()}, **{"test_" + k: v for k, v in te.items()},
             player_ids=np.array([r[0] for r in res]))
    print(f"train {len(train)} positions from {len(seen)} games; test {len(test)} decisions of {len(res)} players", file=sys.stderr)
