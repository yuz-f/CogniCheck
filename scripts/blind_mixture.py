"""Two-mode mixture: with probability pi the player is 'blind' to the tactical content
and chooses as if the engine / hanging-material terms were absent.

P(m) = (1 - pi) softmax(w.x) + pi softmax(w_blind.x),  w_blind = w with danger terms zeroed
pi = sigmoid(a + b*quick + c*time_pressure + d*load + e*long)

w comes from the usual Newton fit; the five pi parameters are fit per player by maximum
likelihood on the training games. Reported: top-1 (all / blunders), log-loss, vs. linear.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import ENGINE_FEATURES, IDX  # noqa: E402

DANGER = sorted(ENGINE_FEATURES | {"hangs", "tp_hangs", "load_hangs", "hangs_moved", "hangs_other", "threat_hangs", "greedy_gain", "ignores_threat"} | {f for f in IDX if f.startswith("allows_")})
DCOLS = [IDX[f] for f in DANGER if f in IDX]


def covariates(ds):
    quick = getattr(ds, "quick", None)
    long_ = getattr(ds, "long", None)
    tp = getattr(ds, "tp", None)
    n = len(ds)
    load = ds.X[ds.starts, IDX["load"]]
    return np.c_[np.ones(n), quick if quick is not None else np.zeros(n), tp if tp is not None else np.zeros(n), load, long_ if long_ is not None else np.zeros(n)].astype(float)


def seg_softmax(z, starts):
    _, p = model._seg_logsumexp(z, starts)
    return p


def mixture_probs(ds, w, theta, C):
    pi = 1 / (1 + np.exp(-(C @ theta)))
    sizes = np.diff(np.append(ds.starts, len(ds.X)))
    p_full = seg_softmax(ds.X @ w, ds.starts)
    wb = w.copy()
    wb[DCOLS] = 0
    p_blind = seg_softmax(ds.X @ wb, ds.starts)
    pir = np.repeat(pi, sizes)
    return (1 - pir) * p_full + pir * p_blind


def fit_theta(ds, w, C):
    def nll(theta):
        p = mixture_probs(ds, w, theta, C)
        return -np.log(p[ds.chosen] + 1e-12).mean() + 1e-3 * (theta[1:] ** 2).sum()

    th0 = np.array([-2.0, 0, 0, 0, 0])
    return minimize(nll, th0, method="L-BFGS-B").x


def evaluate(ds, p):
    ends = np.append(ds.starts[1:], len(ds.X))
    hits = np.array([int(np.argmax(p[s:e])) == c - s for s, e, c in zip(ds.starts, ends, ds.chosen)])
    loss = ds.X[ds.chosen, IDX["eng_loss"]]
    return {"top1": hits.mean(), "top1_blunder": hits[loss >= 2].mean() if (loss >= 2).any() else np.nan, "top1_good": hits[loss <= 0.1].mean(), "logloss": -np.log(p[ds.chosen] + 1e-12).mean()}


def subset_keep(ds, mask):
    sub = cohort._subset(ds, mask)
    for k in ("quick", "long", "tp"):
        v = getattr(ds, k, None)
        setattr(sub, k, v[mask] if v is not None else None)
    return sub


def one(p):
    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    tr, te = subset_keep(ds, train_mask), subset_keep(ds, ~train_mask)
    w = model.fit(tr, prior=cohort.load_population(p["rating"]), lam=0.01)
    Ctr, Cte = covariates(tr), covariates(te)
    theta = fit_theta(tr, w, Ctr)
    base = evaluate(te, seg_softmax(te.X @ w, te.starts))
    mix = evaluate(te, mixture_probs(te, w, theta, Cte))
    pi = 1 / (1 + np.exp(-(Cte @ theta)))
    return p["username"], p["rating"], base, mix, theta, pi.mean()


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists() or (cohort.ANALYSIS.parent / "analysis_v5" / f"{p['id']}.npz").exists()]
    # the stored matrices may be in the v5 backup while a re-analysis runs
    if not (cohort.ANALYSIS / f"{players[0]['id']}.npz").exists():
        cohort.ANALYSIS = cohort.ANALYSIS.parent / "analysis_v5"
        import shutil
        for p in players:  # ctx / maia columns live next to the main file
            for ext in (".ctx.npz", ".maia.npy", ".json"):
                src = cohort.REAL / "analysis" / f"{p['id']}{ext}"
                dst = cohort.ANALYSIS / f"{p['id']}{ext}"
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
    B, M, TH, PI = [], [], [], []
    with ProcessPoolExecutor(2) as ex:
        for name, rating, base, mix, theta, pim in ex.map(one, players):
            B.append(base); M.append(mix); TH.append(theta); PI.append(pim)
            print(f"  {name:20s} {rating}  top1 {base['top1']:.3f}->{mix['top1']:.3f}  blunders {base['top1_blunder']:.3f}->{mix['top1_blunder']:.3f}  logloss {base['logloss']:.3f}->{mix['logloss']:.3f}  mean pi {pim:.2f}  theta {np.round(theta, 2)}", file=sys.stderr)
    for k in ("top1", "top1_blunder", "top1_good", "logloss"):
        print(f"{k:14s} linear {np.nanmean([b[k] for b in B]):.4f}  mixture {np.nanmean([m[k] for m in M]):.4f}")
    TH = np.array(TH)
    print("mean theta [const, quick, time pressure, load, long]:", np.round(TH.mean(0), 2), " mean pi", np.mean(PI).round(3))
    print("players where mixture improves logloss:", sum(m["logloss"] < b["logloss"] for b, m in zip(B, M)), "/", len(B))
