"""Stage A data: encode the cohort's games for the from-scratch network (cognicheck/chessnet.py).

  train  every move of both sides in the cohort's training games
  test   every move of the held-out games (the most recent 15% of each player's games); the rows
         with test_eval set are the player's own decisions, exactly the positions
         scripts/evaluate.py scores, with phase and blunder labels for the breakdown; the other
         rows (the opponent's moves) are there only as game history for stage C

Both splits store game, ply and clock per row (rows of a game are consecutive), from which
chessnet.context() derives the history and clock inputs the same way as for the Lichess shards.

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

sys.path.insert(0, "scripts")
from net_lichess import base_time  # noqa: E402

OUT = cohort.REAL / "net"
MAX_PLIES = 200
PHASES = {"opening": 0, "middlegame": 1, "endgame": 2}


def _elo(h, color):
    try:
        return float(h.get(color + "Elo", "1500"))
    except ValueError:
        return 1500.0


def encode_game(game):
    """Encode the positions of one game (those with more than one legal move)."""
    rec = []
    h = game.headers
    base = base_time(h.get("TimeControl", "180+0"))
    has_clk = any(n.clock() is not None for n in game.mainline())
    clocks = {chess.WHITE: base, chess.BLACK: base}
    board = game.board()
    prev, last = None, None
    for ply, node in enumerate(game.mainline()):
        if ply >= MAX_PLIES:
            break
        if board.legal_moves.count() > 1:
            moves, idx = chessnet.legal_indices(board)
            tgt = moves.index(node.move)
            w, b = _elo(h, "White"), _elo(h, "Black")
            es, eo = (w, b) if board.turn == chess.WHITE else (b, w)
            clk = (clocks[board.turn] / base, clocks[not board.turn] / base) if has_clk else (np.nan, np.nan)
            rec.append((np.packbits(chessnet.planes(board)), chessnet.elo_input(es, eo), idx, tgt, ply, board.copy(stack=False), clk,
                        np.packbits(chessnet.threat_planes(prev, board, last))))
        c = node.clock()
        if c is not None:
            clocks[board.turn] = c
        prev, last = board.copy(stack=False), node.move
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
        recs = encode_game(pg.game)
        if gi in hold:
            for planes, elo, idx, tgt, ply, board, clk, thr in recs:
                own = board.turn == color
                test.append((planes, elo, idx, tgt, PHASES[phase(board)], own and bool(eng_loss[chosen[k]] >= 2), k if own else -1,
                             own, ply, clk, gi, thr))
                k += own
        else:
            k += sum(1 for r in recs if r[5].turn == color)
            if site(pg.game) not in exclude:
                train.extend((planes, elo, idx, tgt, site(pg.game), board.turn, color, ply, clk, thr)
                             for planes, elo, idx, tgt, ply, board, clk, thr in recs)
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
    import lockbox

    for p in lockbox.players():
        f = lockbox.GAMES / f"{p['id']}.pgn"
        if f.exists():
            exclude |= {site(g.game) for g in games_of(parse_pgn(f.read_text()), p["username"])}
    exclude.discard("")
    print(f"{len(exclude)} games excluded from training (held out or lockbox)", file=sys.stderr)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(player_data, players, [exclude] * len(players)))
    # mover of each training position: the cohort player's index for their own moves, len(res) (a
    # shared "unknown" slot) for anyone else; a game shared by two cohort players is kept once,
    # with both players' moves labelled
    owner = {(r[4], r[6]): pi for pi, (pid, tr, te) in enumerate(res) for r in tr}
    seen, train, test, tr_player = set(), [], [], []
    for pi, (pid, tr, te) in enumerate(res):
        new_sites = {r[4] for r in tr} - seen
        kept = [r for r in tr if r[4] in new_sites]  # each game once, even if two cohort players share it
        train.extend(kept)
        tr_player.extend(owner.get((r[4], r[5]), len(res)) for r in kept)
        seen |= new_sites
        test.extend((*r, pi) for r in te)
    OUT.mkdir(parents=True, exist_ok=True)
    gid = {}
    tr_game = np.array([gid.setdefault(r[4], len(gid)) for r in train], np.int32)
    tgid = {}
    te_game = np.array([tgid.setdefault((r[12], r[10]), len(tgid)) for r in test], np.int32)  # (player, game): a game held out for two players appears twice
    n_pl = len(res)
    tr = pack(train, {"player": np.array(tr_player, np.int16), "game": tr_game, "ply": np.array([r[7] for r in train], np.int16),
                      "clock": np.array([r[8] for r in train], np.float16), "threat": np.stack([r[9] for r in train])})
    te = pack(test, {"phase": np.array([r[4] for r in test], np.int8), "blunder": np.array([r[5] for r in test]),
                     "decision": np.array([r[6] for r in test], np.int32), "eval": np.array([r[7] for r in test]),
                     "player": np.array([r[12] for r in test], np.int16),
                     "mover": np.array([r[12] if r[7] else n_pl for r in test], np.int16),
                     "game": te_game, "ply": np.array([r[8] for r in test], np.int16), "clock": np.array([r[9] for r in test], np.float16),
                     "threat": np.stack([r[11] for r in test])})
    np.savez(OUT / "cohort.npz", **{"train_" + k: v for k, v in tr.items()}, **{"test_" + k: v for k, v in te.items()},
             player_ids=np.array([r[0] for r in res]))
    print(f"train {len(train)} positions from {len(seen)} games; test {int(te['eval'].sum())} decisions of {len(res)} players "
          f"({len(test)} rows with the opponents' moves)", file=sys.stderr)
