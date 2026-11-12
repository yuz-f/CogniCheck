"""Which feature families carry the held-out gain? Zero a family, refit, evaluate."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import FEATURE_NAMES, IDX  # noqa: E402

FAMILIES = {
    "full": [],
    "no habit": ["habit"],
    "no allows_* (learned blind spots)": [f for f in FEATURE_NAMES if f.startswith("allows_")],
    "no time pressure": [f for f in FEATURE_NAMES if f.startswith("tp_")],
    "no engine bands/rank": ["eng_loss_small", "eng_loss_mid", "eng_rank"],
    "no engine at all": [f for f in FEATURE_NAMES if f.startswith("eng_") or f == "tp_eng_loss"],
    "no Maia-2 prior": ["maia_logp", "maia_top"],
    "no positional": ["mobility_gain", "passed_pawn", "opens_file", "attacks_weak_pawn", "weakens_shield", "isolates_pawn"],
    "no think time": ["quick_habit", "quick_eng_loss", "quick_capture", "long_eng_loss", "long_quiet"],
    "no attention": ["attn_opp", "attn_own", "reacts_harmless"],
    "no plan persistence": ["same_plan"],
    "no working memory": ["allows_simple", "allows_complex", "load_eng_loss", "load_hangs", "focus_recent", "focus_stale"],
    "no traps (greedy / threat / hangs split)": ["greedy_gain", "creates_threat", "threat_hangs", "hangs_moved", "hangs_other"],
    "no game state (behind / ahead / tilt)": ["behind_eng_loss", "ahead_eng_loss", "tilt_eng_loss"],
    "no rating-band prior": [],  # handled below: evaluate against the global prior instead
}
players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
prior = cohort.load_population()
data = []
for p in players:
    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    data.append((cohort._subset(ds, train_mask), cohort._subset(ds, ~train_mask)))
band_priors = [cohort.load_population(p["rating"]) for p in players]
for name, feats in FAMILIES.items():
    cols = [IDX[f] for f in feats]
    pers, pop = [], []
    for (tr, te), bprior in zip(data, band_priors):
        prior = cohort.load_population() if name == "no rating-band prior" else bprior
        trX, teX = tr.X.copy(), te.X.copy()
        trX[:, cols] = 0; teX[:, cols] = 0
        tr2, te2 = model.Dataset(trX, tr.starts, tr.chosen), model.Dataset(teX, te.starts, te.chosen)
        pr = prior.copy(); pr[cols] = 0
        pers.append(model.evaluate(te2, model.fit(tr2, prior=pr, lam=0.01)))
        pop.append(model.evaluate(te2, pr))
    t1 = np.mean([r["top1"] for r in pers]); ll = np.mean([r["logloss"] for r in pers])
    pt1 = np.mean([r["top1"] for r in pop]); wins = sum(a["top1"] > b["top1"] for a, b in zip(pers, pop))
    print(f"{name:36s} personal top1 {t1:.4f} logloss {ll:.4f} | population top1 {pt1:.4f} | personal wins top1 {wins}/{len(pop)}")
