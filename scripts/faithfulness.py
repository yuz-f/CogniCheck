"""Are the explanations faithful? A deletion test on held-out positions.

For each held-out position, take the model's top predicted move and the reason the
explanation ranks first for it. Neutralise that reason (set those features of the move
to the model's own expectation over the legal moves, which is exactly the quantity the
reason is measured against) and recompute the move's probability. A faithful
explanation means the probability drops, and drops more than when a randomly chosen
non-top reason is neutralised instead.

Also reports how much of each prediction rests on the non-linear "pattern" term, i.e.
how much of the model is *not* covered by readable reasons.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort  # noqa: E402
from cognicheck.features import FEATURE_NAMES  # noqa: E402
from cognicheck.predictor import _GROUP_OF, Predictor  # noqa: E402
from cognicheck.profile import Profile  # noqa: E402

MAX_POSITIONS = 400  # per player


def group_cols(g: str) -> list[int]:
    return [j for j, f in enumerate(FEATURE_NAMES) if _GROUP_OF.get(f, f) == g]


def neutralise(pred: Predictor, X: np.ndarray, i: int, g: str) -> float:
    """Probability of move i after replacing its group-g features by the expectation."""
    probs = pred.model_distribution(SimpleNamespace(X=X))
    if g == "pattern":
        z = X @ pred.w
        r = pred.head.residual(X)
        z = z + r
        z[i] += float(probs @ r) - r[i]
    else:
        X2 = X.copy()
        cols = group_cols(g)
        X2[i, cols] = probs @ X[:, cols]
        z = X2 @ pred.w + (pred.head.residual(X2) if pred.head is not None else 0.0)
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(p[i])


def one(p: dict, rng) -> dict:
    prof = Profile.load(cohort.PROFILES_REAL / f"{p['id']}.json")
    pred = Predictor(prof)
    ds, game = cohort._load_ds(p["id"])
    cohort._fill_habits(ds, cohort.train_mask(p["id"], game))
    te = cohort._subset(ds, ~cohort.train_mask(p["id"], game))
    ends = np.append(te.starts[1:], len(te.X))
    idx = rng.permutation(len(te))[:MAX_POSITIONS]
    drop_top, drop_rand, lowered, pattern_share, top_is_pattern = [], [], 0, [], 0
    for k in idx:
        X = te.X[te.starts[k]:ends[k]]
        pa = SimpleNamespace(X=X)
        probs = pred.model_distribution(pa)
        i = int(np.argmax(probs))
        reasons = pred.reasons(pa, i, k=50)
        pos = [r for r in reasons if r["effect"] > 0]
        if not pos:
            continue
        effects = np.array([abs(r["effect"]) for r in reasons])
        pattern_share.append(sum(abs(r["effect"]) for r in reasons if r["feature"] == "pattern") / effects.sum())
        top = pos[0]
        top_is_pattern += top["feature"] == "pattern"
        p_top = neutralise(pred, X, i, top["feature"])
        drop_top.append(probs[i] - p_top)
        lowered += p_top < probs[i]
        others = [r for r in pos[1:]]
        if others:
            drop_rand.append(probs[i] - neutralise(pred, X, i, others[rng.integers(len(others))]["feature"]))
    n = len(drop_top)
    return {
        "id": p["id"], "positions": n,
        "top_reason_lowers_probability": lowered / max(1, n),
        "mean_drop_top": float(np.mean(drop_top)) if n else None,
        "mean_drop_other": float(np.mean(drop_rand)) if drop_rand else None,
        "pattern_share": float(np.mean(pattern_share)) if pattern_share else None,
        "top_reason_is_pattern": top_is_pattern / max(1, n),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.PROFILES_REAL / f"{p['id']}.json").exists()]
    rows = [one(p, rng) for p in players]
    for r in rows:
        print(f"  {r['id']:22s} n={r['positions']:4d} lowered {r['top_reason_lowers_probability']:.3f} drop top {r['mean_drop_top']:.3f} other {r['mean_drop_other']:.3f} pattern share {r['pattern_share']:.3f}", file=sys.stderr)
    agg = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ("top_reason_lowers_probability", "mean_drop_top", "mean_drop_other", "pattern_share", "top_reason_is_pattern")}
    agg["players"] = len(rows)
    agg["positions"] = int(sum(r["positions"] for r in rows))
    (cohort.REAL / "faithfulness.json").write_text(json.dumps({"summary": agg, "rows": rows}, indent=1))
    print(json.dumps(agg, indent=1))
