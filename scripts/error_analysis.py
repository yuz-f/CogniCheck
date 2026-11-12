"""Where do the misses come from? Full model (linear + head + book) on the frozen,
time-ordered test games (each player's most recent 15 %), broken down by phase,
confidence, think time, number of legal moves, forcedness and rating."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import IDX, phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.predictor import BOOK_ALPHA  # noqa: E402
from cognicheck.profile import build_book, iter_decisions, position_key  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-CogniCheck/5f260670-b7ec-53c5-b964-37de6d06eaf3/scratchpad")
OUT = SCRATCH / "errors"
OUT.mkdir(exist_ok=True)
FIELDS = ["rating", "ply", "phase", "think", "tp", "n_moves", "hit", "top3", "p_chosen", "p_max", "entropy", "maia_top_hit", "eng_best_hit", "in_book", "forced", "capture", "recapture", "check_evasion", "loss"]
PH = {"opening": 0, "middlegame": 1, "endgame": 2}


def one(p):
    f = OUT / f"{p['id']}.npy"
    if f.exists():
        return p["username"], np.load(f)
    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    tr, te = cohort._subset(ds, train_mask), cohort._subset(ds, ~train_mask)
    prior = cohort.load_population(p["rating"])
    w = model.fit(tr, prior=prior, lam=0.01)
    head = cohort.fit_head(tr, game[train_mask], w)
    z = te.X @ w + (head.residual(te.X) if head is not None else 0.0)
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    book = build_book([g for i, g in enumerate(pgs) if i not in hold])
    ends = np.append(te.starts[1:], len(te.X))
    rows = []
    k = 0
    for gi, pg in enumerate(pgs):
        if gi not in hold:
            continue
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) <= 1:
                continue
            s, e, c = te.starts[k], ends[k], te.chosen[k]
            k += 1
            pr = model.softmax(z[s:e])
            counts = book.get(position_key(d.pa.board), {})
            tot = sum(counts.values())
            in_book = tot > 0
            if in_book:
                cvec = np.array([counts.get(m.uci(), 0) for m in d.pa.moves], float)
                pr = (cvec + BOOK_ALPHA * pr) / (tot + BOOK_ALPHA)
            ci = c - s
            order = np.argsort(-pr)
            X = te.X[s:e]
            b = d.pa.board
            think = np.nan
            forced = b.is_check() and len(d.pa.moves) <= 3 or (X[ci, IDX["recapture"]] > 0 and X[ci, IDX["capture_gain"]] > 0)
            rows.append([
                p["rating"], d.ply, PH[phase(b)], think, float(d.pa.time_pressure), len(d.pa.moves),
                float(order[0] == ci), float(ci in order[:3]), pr[ci], pr.max(), float(-(pr * np.log(pr + 1e-12)).sum()),
                float(X[ci, IDX["maia_top"]] > 0), float(X[ci, IDX["eng_loss"]] <= 0.1), float(in_book), float(forced),
                float(X[ci, IDX["capture"]]), float(X[ci, IDX["recapture"]]), float(b.is_check()), float(X[ci, IDX["eng_loss"]]),
            ])
    assert k == len(te)
    # think time from the stored quick/long flags is coarse; use ds.quick/long for the held-out rows
    R = np.array(rows, dtype=np.float64)
    q = getattr(ds, "quick", None)
    if q is not None:
        R[:, FIELDS.index("think")] = np.where(ds.quick[~train_mask], 0, np.where(ds.long[~train_mask], 2, 1))
    np.save(f, R)
    return p["username"], R


def bucket_report(R, col, edges, labels, title):
    v = R[:, FIELDS.index(col)]
    hit, t3 = R[:, FIELDS.index("hit")], R[:, FIELDS.index("top3")]
    print(f"\n{title}")
    for (lo, hi), lab in zip(zip(edges[:-1], edges[1:]), labels):
        m = (v >= lo) & (v < hi)
        if m.sum():
            print(f"  {lab:28s} {m.mean():5.1%} of moves   top1 {hit[m].mean():.3f}   top3 {t3[m].mean():.3f}")


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    allR = []
    with ProcessPoolExecutor(4) as ex:
        for name, R in ex.map(one, players):
            allR.append(R)
            print(f"  {name}: {len(R)} positions, top1 {R[:, FIELDS.index('hit')].mean():.3f}", file=sys.stderr)
    R = np.concatenate(allR)
    hit = R[:, FIELDS.index("hit")]
    print(f"{len(R)} held-out positions, {len(allR)} players; pooled top1 {hit.mean():.4f}, top3 {R[:, FIELDS.index('top3')].mean():.4f}")
    print(f"  P(played move) >= 0.3: {(R[:, FIELDS.index('p_chosen')] >= 0.3).mean():.3f};  >= 0.5: {(R[:, FIELDS.index('p_chosen')] >= 0.5).mean():.3f}")
    bucket_report(R, "phase", [0, 1, 2, 3], ["opening", "middlegame", "endgame"], "By phase")
    bucket_report(R, "ply", [0, 10, 20, 40, 60, 80, 1000], ["ply 0-9", "10-19", "20-39", "40-59", "60-79", "80+"], "By ply")
    bucket_report(R, "p_max", [0, 0.3, 0.5, 0.7, 0.9, 1.01], ["max prob < .3", ".3-.5", ".5-.7", ".7-.9", ">= .9"], "By model confidence (calibration)")
    bucket_report(R, "n_moves", [2, 10, 20, 30, 40, 200], ["2-9 legal", "10-19", "20-29", "30-39", "40+"], "By number of legal moves")
    bucket_report(R, "think", [0, 1, 2, 3], ["quick (<3s)", "normal", "long (>=15s)"], "By think time")
    bucket_report(R, "tp", [0, 1, 2], ["normal clock", "short of time"], "By time pressure")
    bucket_report(R, "forced", [0, 1, 2], ["free choice", "forced (recapture / few moves in check)"], "By forcedness")
    bucket_report(R, "in_book", [0, 1, 2], ["not in player's book", "in player's book"], "By opening book")
    bucket_report(R, "rating", [0, 1400, 1800, 2200, 4000], ["<1400", "1400-1800", "1800-2200", "2200+"], "By rating")
    bucket_report(R, "loss", [0, 0.1, 0.5, 2, 100], ["engine-best move", "small inaccuracy", "clear inaccuracy", "blunder"], "By objective quality of the move played")
    e = R[:, FIELDS.index("entropy")]
    print(f"\nmean entropy of the model distribution {e.mean():.2f} nats (= {np.exp(e.mean()):.1f} 'effective' candidate moves)")
    print(f"share of misses where the played move was the model's #2: {((R[:, FIELDS.index('top3')] == 1) & (hit == 0)).sum() / max(1, (hit == 0).sum()):.3f} (top-2 or 3)")
    maia = R[:, FIELDS.index("maia_top_hit")]
    print(f"Maia top move = played: {maia.mean():.3f}; model hits where Maia misses: {hit[maia == 0].mean():.3f}; model misses where Maia hits: {(hit[maia == 1] == 0).mean():.3f}")
