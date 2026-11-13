"""Plan step 2: a learned gate between Maia-2 and the personal model (a two-expert mixture).

    log p(m) = sum_k z_k * (a_k * log p_personal(m) + b_k * log p_maia(m))   (renormalised)

z describes the situation: phase, time pressure, complexity, each expert's uncertainty,
whether the position is in the player's book, and whether Maia-2 is available (it is not
used before ply 10). The gate weights a, b are learned once over all players.

No leakage: for each player the gate is fitted on the most recent 15% of their *training*
games, with the personal model refitted on the older rest. The test set (the most recent
15% of games, as in scripts/evaluate.py) is scored with the personal model fitted on all
training games and the gate frozen.

  python scripts/gate_experiment.py
"""
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import IDX, phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.predictor import BOOK_ALPHA  # noqa: E402
from cognicheck.profile import build_book, iter_decisions, position_key  # noqa: E402

GATE_FRAC = 0.15
NO_THINK = "--no-think" in sys.argv  # drop the think time of the move being predicted (unknown before it is played)
CLOCK = "--clock" in sys.argv  # plan step 3: clock and session context (scripts/clock_columns.py)
Z_NAMES = ["const", "middlegame", "endgame", "time_pressure", "load", "maia_entropy", "own_entropy", "in_book", "maia_on"]
CLOCK_NAMES = ["own_clock", "opp_clock", "clock_lead", "prev_think", "opp_think", "session_game", "after_loss", "move_number"]
MOVE_FEATURES = ["eng_loss", "eng_best", "capture", "check", "habit", "hangs", "castle", "king_move", "pawn_move", "same_piece"]
if CLOCK:
    Z_NAMES = Z_NAMES + CLOCK_NAMES
F_IDX = [IDX[f] for f in MOVE_FEATURES]


def _entropy(p):
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def _expert_probs(p, ds, game, pgs, fit_games, eval_games, clock=None):
    """Personal model fitted on fit_games, both experts' log-probs on eval_games."""
    fit_mask = np.isin(game, list(fit_games))
    cohort._fill_habits(ds, fit_mask)
    tr = cohort._subset(ds, fit_mask)
    w = model.fit(tr, prior=cohort.load_population(p["rating"]), lam=0.01)
    head = cohort.fit_head(tr, game[fit_mask], w)
    book = build_book([g for i, g in enumerate(pgs) if i in fit_games])
    ev_mask = np.isin(game, list(eval_games))
    te = cohort._subset(ds, ev_mask)
    ck = clock[ev_mask] if clock is not None else None
    tp = ds.tp[ev_mask] if getattr(ds, "tp", None) is not None else np.zeros(len(te), bool)
    ends = np.append(te.starts[1:], len(te.X))
    rows, k = [], 0
    for gi, pg in enumerate(pgs):
        if gi not in eval_games:
            continue
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) <= 1:
                continue
            s, e, c = te.starts[k], ends[k], te.chosen[k]
            X = te.X[s:e]
            po = model.softmax(X @ w + (head.residual(X) if head is not None else 0.0))
            counts = book.get(position_key(d.pa.board), {})
            tot = sum(counts.values())
            if tot:
                cvec = np.array([counts.get(m.uci(), 0) for m in d.pa.moves], float)
                po = (cvec + BOOK_ALPHA * po) / (tot + BOOK_ALPHA)
            lm = X[:, IDX["maia_logp"]]
            maia_on = bool(np.any(lm != 0))
            pm = model.softmax(lm) if maia_on else np.full(len(X), 1.0 / len(X))
            ph = phase(d.pa.board)
            z = np.array([1.0, ph == "middlegame", ph == "endgame", float(tp[k]), X[0, IDX["load"]] / 10.0,
                          _entropy(pm) if maia_on else 0.0, _entropy(po), float(tot > 0), float(maia_on)])
            if CLOCK:
                cv = ck[k].astype(float).copy()
                cv[5] /= 10.0  # session_game
                z = np.concatenate([z, cv])
            rows.append({"lpo": np.log(np.maximum(po, 1e-6)), "lpm": np.log(np.maximum(pm, 1e-6)), "z": z,
                         "F": X[:, F_IDX].copy(), "ck": (z[-len(CLOCK_NAMES):] if CLOCK else None), "c": int(c - s), "phase": ph, "blunder": bool(X[c - s, IDX["eng_loss"]] >= 2)})
            k += 1
    if k != len(te):
        raise RuntimeError(f"{p['id']}: {k} decisions vs {len(te)} rows")
    return rows


def player(p):
    ds, game = cohort._load_ds(p["id"])
    if NO_THINK and getattr(ds, "quick", None) is not None:
        ds.quick = np.zeros_like(ds.quick)
        ds.long = np.zeros_like(ds.long)
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    hold = cohort.holdout_games(p["id"])
    train_games = sorted(set(game.tolist()) - hold)
    n_gate = max(1, int(round(len(train_games) * GATE_FRAC)))
    gate_games = set(train_games[:n_gate])  # newest training games (PGN is newest first)
    inner = set(train_games[n_gate:])
    clock = None
    if CLOCK:
        clock = np.load(cohort.ANALYSIS / f"{p['id']}.clock.npz")["C"]
    gate_rows = _expert_probs(p, ds, game, pgs, inner, gate_games, clock)
    test_rows = _expert_probs(p, ds, game, pgs, set(train_games), hold, clock)
    return {"id": p["id"], "gate": gate_rows, "test": test_rows}


def design(rows):
    pos = []
    for r in rows:
        parts = [np.outer(r["lpo"], r["z"]), np.outer(r["lpm"], r["z"])]
        if CLOCK:  # move features x clock context
            parts.append((r["F"][:, :, None] * r["ck"][None, None, :]).reshape(len(r["F"]), -1))
        Xb = np.concatenate(parts, axis=1)
        pos.append((Xb, r["c"]))
    return model.Dataset.from_positions(pos)


def blend(r, theta):
    K = len(Z_NAMES)
    a, b = theta[:K] @ r["z"], theta[K:2 * K] @ r["z"]
    s = a * r["lpo"] + b * r["lpm"]
    if CLOCK:
        s = s + (r["F"][:, :, None] * r["ck"][None, None, :]).reshape(len(r["F"]), -1) @ theta[2 * K:]
    return model.softmax(s), a, b


def score(rows, f):
    hit, ll = [], []
    for r in rows:
        p = f(r)
        hit.append(int(np.argmax(p)) == r["c"])
        ll.append(-np.log(max(p[r["c"]], 1e-9)))
    return np.array(hit, float), np.array(ll)


def ci(x, rng):
    bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    if nums:
        players = players[: int(nums[0])]
    out = []
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(player, players), 1):
            out.append(r)
            print(f"  {i}/{len(players)} {r['id']}", file=sys.stderr)
    K = len(Z_NAMES)
    gds = design([row for r in out for row in r["gate"]])
    prior = np.zeros(2 * K + (len(MOVE_FEATURES) * len(CLOCK_NAMES) if CLOCK else 0))
    prior[0] = 1.0
    theta = model.fit(gds, prior=prior, lam=1e-4)
    print("\ngate weights (personal | maia):")
    for j, n in enumerate(Z_NAMES):
        print(f"  {n:14s} {theta[j]:+.3f} | {theta[K + j]:+.3f}")
    rng = np.random.default_rng(0)
    per = {"personal": [], "maia": [], "gated": []}
    ph_hits = {m: {"opening": [], "middlegame": [], "endgame": [], "blunder": []} for m in per}
    fns = {"personal": lambda r: np.exp(r["lpo"]) / np.exp(r["lpo"]).sum(),
           "maia": lambda r: np.exp(r["lpm"]) / np.exp(r["lpm"]).sum(),
           "gated": lambda r: blend(r, theta)[0]}
    for r in out:
        for m, f in fns.items():
            h, ll = score(r["test"], f)
            per[m].append((h.mean(), ll.mean()))
            for name in ("opening", "middlegame", "endgame"):
                msk = np.array([t["phase"] == name for t in r["test"]])
                if msk.any():
                    ph_hits[m][name].append(h[msk].mean())
            bl = np.array([t["blunder"] for t in r["test"]])
            if bl.any():
                ph_hits[m]["blunder"].append(h[bl].mean())
    print(f"\n{len(out)} players, test = most recent 15% of games. Mean over players.")
    print(f"{'model':10s} {'top-1':>8s} {'log-loss':>9s} {'open':>7s} {'middle':>7s} {'end':>7s} {'blunder':>8s}")
    for m in per:
        a = np.array(per[m])
        print(f"{m:10s} {a[:, 0].mean():8.4f} {a[:, 1].mean():9.4f} " + " ".join(f"{np.mean(ph_hits[m][n]):7.3f}" for n in ("opening", "middlegame", "endgame")) + f" {np.mean(ph_hits[m]['blunder']):8.3f}")
    d1 = np.array(per["gated"])[:, 0] - np.array(per["personal"])[:, 0]
    d2 = np.array(per["gated"])[:, 1] - np.array(per["personal"])[:, 1]
    lo, hi = ci(d1, rng)
    lo2, hi2 = ci(d2, rng)
    print(f"\ngated minus personal: top-1 {d1.mean() * 100:+.2f} pts [{lo * 100:+.2f}, {hi * 100:+.2f}], better for {(d1 > 0).sum()}/{len(d1)}; "
          f"log-loss {d2.mean():+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    (cohort.REAL / ("gate_experiment" + ("_clock" if CLOCK else "") + ("_nothink" if NO_THINK else "") + ".json")).write_text(json.dumps({
        "theta": [float(x) for x in theta], "z_names": Z_NAMES,
        "top1_diff": [float(d1.mean()), float(lo), float(hi)], "logloss_diff": [float(d2.mean()), float(lo2), float(hi2)],
        "rows": [{"id": r["id"], **{m: per[m][i] for m in per}} for i, r in enumerate(out)]}, indent=1))
