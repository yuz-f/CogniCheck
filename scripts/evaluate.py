"""One evaluation harness for every accuracy claim.

For each player, the most recent `test_games` are the test set and all older games are
training data. Models compared on exactly the same positions:

  maia            Maia-2 alone at the players' ratings, on every ply (no gating)
  population      rating-band prior + population head, no personal information
  personal_linear the player's linear model alone
  personal_full   linear + fine-tuned head + the player's opening book (the shipped model)

Metrics: top-1, top-3, log-loss, calibration error of the top guess, top-1 by phase and
on blunders. Summaries are means over players with 95% bootstrap intervals (players
resampled), and paired differences personal_full minus each baseline.

  python scripts/evaluate.py            # cohort validation set (82 players)
  python scripts/lockbox.py evaluate    # the sealed final test, once
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import IDX, phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.predictor import BOOK_ALPHA  # noqa: E402
from cognicheck.profile import _elo, build_book, iter_decisions, position_key  # noqa: E402

MODELS = ["maia", "population", "personal_linear", "personal_full"]
PHASES = ["opening", "middlegame", "endgame"]
_NEURAL = None


def _neural():
    global _NEURAL
    if _NEURAL is None:
        from cognicheck.neural import Neural

        _NEURAL = Neural()
    return _NEURAL


def _metrics(probs: list[np.ndarray], chosen: list[int], ph: list[str], blunder: np.ndarray) -> dict:
    hit = np.array([int(np.argmax(p)) == c for p, c in zip(probs, chosen)], float)
    top3 = np.array([c in np.argsort(-p)[:3] for p, c in zip(probs, chosen)], float)
    pc = np.array([max(p[c], 1e-9) for p, c in zip(probs, chosen)])
    conf = np.array([p.max() for p in probs])
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(hit[bins == b].mean() - conf[bins == b].mean()) for b in range(10) if (bins == b).any())
    out = {"n": len(hit), "top1": hit.mean(), "top3": top3.mean(), "logloss": float(-np.log(pc).mean()), "ece": float(ece)}
    for name in PHASES:
        m = np.array([x == name for x in ph])
        out["top1_" + name] = float(hit[m].mean()) if m.any() else None
    out["top1_blunder"] = float(hit[blunder].mean()) if blunder.any() else None
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def evaluate_player(p: dict, test_games: int | None = None) -> dict:
    """Uses cohort.GAMES / cohort.ANALYSIS as currently set (callers may point them elsewhere)."""
    ds, game = cohort._load_ds(p["id"])
    if test_games is None:
        hold = cohort.holdout_games(p["id"])
    else:
        hold = set(range(test_games))
    train = ~np.isin(game, list(hold))
    cohort._fill_habits(ds, train)
    tr, te = cohort._subset(ds, train), cohort._subset(ds, ~train)
    band = cohort.load_population(p["rating"])
    w = model.fit(tr, prior=band, lam=0.01)
    head = cohort.fit_head(tr, game[train], w)
    from cognicheck import head as head_mod

    pop_head = head_mod.HeadParams.load(cohort.HEAD_REAL) if cohort.HEAD_REAL.exists() else None
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    book = build_book([g for i, g in enumerate(pgs) if i not in hold])
    neural = _neural()
    ends = np.append(te.starts[1:], len(te.X))
    probs = {m: [] for m in MODELS}
    chosen, ph = [], []
    k = 0
    for gi, pg in enumerate(pgs):
        if gi not in hold:
            continue
        h = pg.game.headers
        es = _elo(h, "White" if pg.color else "Black") or p["rating"]
        eo = _elo(h, "Black" if pg.color else "White") or es
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) <= 1:
                continue
            s, e, c = te.starts[k], ends[k], te.chosen[k]
            k += 1
            X = te.X[s:e]
            mp = neural.move_probs(d.pa.board, es, eo)
            pm = np.array([mp.get(m.uci(), 0.0) for m in d.pa.moves]) + 1e-6
            probs["maia"].append(pm / pm.sum())
            zp = X @ band + (pop_head.residual(X) if pop_head is not None else 0.0)
            probs["population"].append(model.softmax(zp))
            probs["personal_linear"].append(model.softmax(X @ w))
            pf = model.softmax(X @ w + (head.residual(X) if head is not None else 0.0))
            counts = book.get(position_key(d.pa.board), {})
            tot = sum(counts.values())
            if tot:
                cvec = np.array([counts.get(m.uci(), 0) for m in d.pa.moves], float)
                pf = (cvec + BOOK_ALPHA * pf) / (tot + BOOK_ALPHA)
            probs["personal_full"].append(pf)
            chosen.append(int(c - s))
            ph.append(phase(d.pa.board))
    if k != len(te):
        raise RuntimeError(f"{p['id']}: {k} test decisions vs {len(te)} stored rows")
    blunder = te.X[te.chosen, IDX["eng_loss"]] >= 2
    return {"id": p["id"], "rating": p["rating"], "models": {m: _metrics(probs[m], chosen, ph, blunder) for m in MODELS}}


def _ci(x: np.ndarray, rng) -> tuple[float, float]:
    bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarise(rows: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    out = {"players": len(rows), "models": {}, "personal_full_minus": {}}
    keys = ["top1", "top3", "logloss", "ece", "top1_opening", "top1_middlegame", "top1_endgame", "top1_blunder"]
    for m in MODELS:
        out["models"][m] = {}
        for key in keys:
            v = np.array([r["models"][m][key] for r in rows if r["models"][m][key] is not None])
            lo, hi = _ci(v, rng)
            out["models"][m][key] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for base in ("maia", "population", "personal_linear"):
        out["personal_full_minus"][base] = {}
        for key in ("top1", "logloss"):
            d = np.array([r["models"]["personal_full"][key] - r["models"][base][key] for r in rows])
            lo, hi = _ci(d, rng)
            better = int((d > 0).sum()) if key == "top1" else int((d < 0).sum())
            out["personal_full_minus"][base][key] = {"mean": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "players_better": better}
    return out


def print_summary(s: dict) -> None:
    print(f"\n{s['players']} players. Mean over players, 95% bootstrap interval in brackets.\n")
    print(f"{'model':16s} {'top-1':>22s} {'top-3':>8s} {'log-loss':>9s} {'calib.err':>9s} {'open':>7s} {'middle':>7s} {'end':>7s} {'blunder':>8s}")
    for m, r in s["models"].items():
        t = r["top1"]
        print(f"{m:16s} {t['mean']:.4f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] {r['top3']['mean']:8.4f} {r['logloss']['mean']:9.4f} {r['ece']['mean']:9.4f} {r['top1_opening']['mean']:7.3f} {r['top1_middlegame']['mean']:7.3f} {r['top1_endgame']['mean']:7.3f} {r['top1_blunder']['mean']:8.3f}")
    print("\npersonal_full minus baseline (top-1 points; log-loss, negative is better):")
    for b, r in s["personal_full_minus"].items():
        t, ll = r["top1"], r["logloss"]
        print(f"  vs {b:16s} top-1 {t['mean'] * 100:+.2f} [{t['ci95'][0] * 100:+.2f}, {t['ci95'][1] * 100:+.2f}] better for {t['players_better']}/{s['players']}   log-loss {ll['mean']:+.4f} [{ll['ci95'][0]:+.4f}, {ll['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    import multiprocessing

    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    rows = []
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(evaluate_player, players), 1):
            rows.append(r)
            print(f"  {i}/{len(players)} {r['id']}: full {r['models']['personal_full']['top1']:.3f} maia {r['models']['maia']['top1']:.3f}", file=sys.stderr)
    s = summarise(rows)
    out = cohort.REAL / "evaluation_validation.json"
    out.write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
    print_summary(s)
