"""Does the opening book help on real players? Evaluate the personal model with and
without the book on the cohort's held-out games, for several pseudo-counts and depths."""
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.profile import iter_decisions, position_key  # noqa: E402

ALPHAS = [1.0, 2.0, 4.0, 8.0]
PLIES = [20, 40, 400]


def one(p):
    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    tr, te = cohort._subset(ds, train_mask), cohort._subset(ds, ~train_mask)
    w = model.fit(tr, prior=cohort.load_population(p["rating"]), lam=0.01)
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    # books of different depth from the training games
    books = {}
    for plies in PLIES:
        book = defaultdict(lambda: defaultdict(int))
        for gi, pg in enumerate(pgs):
            if gi in hold:
                continue
            b = pg.game.board()
            for ply, mv in enumerate(pg.game.mainline_moves()):
                if ply >= plies:
                    break
                if b.turn == pg.color:
                    book[position_key(b)][mv.uci()] += 1
                b.push(mv)
        books[plies] = book
    # held-out decisions in the same order as te rows
    keys, ucis = [], []
    for gi, pg in enumerate(pgs):
        if gi not in hold:
            continue
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) > 1:
                keys.append(position_key(d.pa.board))
                ucis.append([m.uci() for m in d.pa.moves])
    assert len(keys) == len(te), (len(keys), len(te))
    ends = np.append(te.starts[1:], len(te.X))
    base = model.evaluate(te, w)["top1"]
    out = {"base": base, "n": len(te)}
    z = te.X @ w
    for plies, book in books.items():
        for a in ALPHAS:
            hits = 0
            for k, (s, e, c) in enumerate(zip(te.starts, ends, te.chosen)):
                pr = model.softmax(z[s:e])
                counts = book.get(keys[k], {})
                tot = sum(counts.values())
                if tot:
                    cvec = np.array([counts.get(u, 0) for u in ucis[k]], float)
                    pr = (cvec + a * pr) / (tot + a)
                hits += int(np.argmax(pr)) == c - s
            out[f"p{plies}_a{a}"] = hits / len(te)
    return p["username"], out


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for name, out in ex.map(one, players):
            res[name] = out
            print(name, {k: round(v, 3) for k, v in out.items() if k != "n"}, file=sys.stderr)
    keys = [k for k in next(iter(res.values())) if k != "n"]
    print("mean over players:")
    for k in keys:
        print(f"  {k:12s} {np.mean([r[k] for r in res.values()]):.4f}")
    json.dump(res, open("/tmp/claude-0/-home-user-CogniCheck/5f260670-b7ec-53c5-b964-37de6d06eaf3/scratchpad/book_eval.json", "w"))
