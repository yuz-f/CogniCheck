"""Does more data per player help? Data-scaling curve on 20 players.

Fetch up to 1000 recent blitz games per player, keep the 30 most recent as the frozen
test set (the same games as the cohort's held-out set), and train on the following
170 / 470 / 970 games. Everything else (features, Maia, book, head, priors) as in v5.

Stages: fetch, analyse, columns (maia + derived), fit. Results in scaling/results.json.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402

ROOT = cohort.REAL / "scaling"
GAMES, ANALYSIS = ROOT / "games", ROOT / "analysis"
N_PLAYERS, N_GAMES, TEST_GAMES = 20, 1000, 30
TRAIN_SIZES = [170, 470, 970]


def _patch():
    cohort.GAMES, cohort.ANALYSIS = GAMES, ANALYSIS


def players():
    ps = sorted((p for p in json.loads(cohort.PLAYERS.read_text()) if not p.get("added")), key=lambda p: p["rating"])
    step = max(1, len(ps) // N_PLAYERS)
    return ps[::step][:N_PLAYERS]


def fetch():
    GAMES.mkdir(parents=True, exist_ok=True)
    for p in players():
        out = GAMES / f"{p['id']}.pgn"
        if out.exists() and out.stat().st_size > 1000:
            continue
        url = f"{cohort.LICHESS}/api/games/user/{p['id']}?max={N_GAMES}&rated=true&perfType=blitz&clocks=true&evals=false&opening=false"
        text = cohort._get(url, "application/x-chess-pgn", timeout=900)
        out.write_text(text)
        print(f"  fetched {p['username']}: {text.count('[Event ')} games", file=sys.stderr)
        time.sleep(2)


def _analyse(pid, username, depth):
    _patch()
    return cohort._analyse_player(pid, username, depth)


def _derived(pid, username):
    _patch()
    return cohort._memory_player(pid, username)


def _maia(pid, username):
    _patch()
    return cohort._maia_player(pid, username)


def stage(fn, label, workers=4, **kw):
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, p["id"], p["username"], **kw): p for p in players()}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                f.result()
                print(f"  {label} {p['username']} ({i}/{len(futs)}, {time.time() - t0:.0f}s)", file=sys.stderr)
            except Exception as e:
                print(f"  FAILED {label} {p['username']}: {e}", file=sys.stderr)


def _fit_one(p):
    _patch()
    from cognicheck.ingest import games_of, parse_pgn

    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((ANALYSIS / f"{p['id']}.json").read_text())["games"]
    test = set(range(min(TEST_GAMES, n_games)))
    test_mask = np.isin(game, list(test))
    prior = cohort.load_population(p["rating"])
    pgs = games_of(parse_pgn((GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    out = {"username": p["username"], "rating": p["rating"], "games": n_games, "curve": {}}
    for n_train in TRAIN_SIZES:
        train_games = set(range(TEST_GAMES, min(TEST_GAMES + n_train, n_games)))
        if len(train_games) < n_train * 0.8:
            break
        train_mask = np.isin(game, list(train_games))
        ds_c, _ = cohort._load_ds(p["id"])  # fresh copy: habits are written in place
        cohort._fill_habits(ds_c, train_mask)
        tr, te = cohort._subset(ds_c, train_mask), cohort._subset(ds_c, test_mask)
        w = model.fit(tr, prior=prior, lam=0.01)
        head = cohort.fit_head(tr, game[train_mask], w)
        scores = {"linear_book": te.X @ w}
        ev = {"linear": model.evaluate(te, w)}
        if head is not None:
            z = te.X @ w + head.residual(te.X)
            ev["head"] = model.evaluate(te, None, scores=z)
            scores["full"] = z
        ev.update(_book(pgs, test, train_games, te, scores))
        ev["n_train_positions"] = len(tr)
        out["curve"][str(len(train_games))] = ev
        print(f"    {p['username']} train {len(train_games)} games: " + " ".join(f"{k} {v['top1']:.3f}" for k, v in ev.items() if isinstance(v, dict)), file=sys.stderr)
    return out


def _book(pgs, test, train_games, te, scores):
    """Book from the training games only (not every non-test game)."""
    from cognicheck.predictor import BOOK_ALPHA
    from cognicheck.profile import build_book, iter_decisions, position_key

    book = build_book([g for i, g in enumerate(pgs) if i in train_games])
    keys, ucis = [], []
    for gi, pg in enumerate(pgs):
        if gi not in test:
            continue
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) > 1:
                keys.append(position_key(d.pa.board))
                ucis.append([m.uci() for m in d.pa.moves])
    if len(keys) != len(te):
        return {}
    ends = np.append(te.starts[1:], len(te.X))
    out = {}
    for name, z in scores.items():
        hits = 0
        for k, (s, e, c) in enumerate(zip(te.starts, ends, te.chosen)):
            pr = model.softmax(z[s:e])
            counts = book.get(keys[k], {})
            tot = sum(counts.values())
            if tot:
                cvec = np.array([counts.get(u, 0) for u in ucis[k]], float)
                pr = (cvec + BOOK_ALPHA * pr) / (tot + BOOK_ALPHA)
            hits += int(np.argmax(pr)) == c - s
        out[name + "+book"] = {"top1": hits / len(te)}
    return out


def fit():
    import multiprocessing

    res = []
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        for r in ex.map(_fit_one, players()):
            res.append(r)
    (ROOT / "results.json").write_text(json.dumps(res, indent=1))
    print("\nData-scaling curve (top-1 on each player's 30 most recent games, mean over players):")
    for n in TRAIN_SIZES:
        rows = [r["curve"].get(str(n)) or r["curve"].get(max(r["curve"], key=int)) for r in res if r["curve"]]
        rows = [r for r in rows if r]
        keys = [k for k in ("linear", "linear_book+book", "head", "full+book") if all(k in r for r in rows)]
        print(f"  {n:4d} training games (n={len(rows)}): " + "  ".join(f"{k} {np.mean([r[k]['top1'] for r in rows]):.4f}" for k in keys))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("fetch", "all"):
        fetch()
    if st in ("analyse", "all"):
        stage(_analyse, "analysed", workers=4, depth=5)
    if st in ("columns", "all"):
        stage(_maia, "maia", workers=4)
        stage(_derived, "derived", workers=4)
    if st in ("fit", "all"):
        fit()
