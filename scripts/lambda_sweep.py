"""Pick the regularisation strength: mean held-out log-loss / top-1 over the cohort."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402

players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
prior = cohort.load_population()
lams = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
res = {l: [] for l in lams}
pop = []
for p in players:
    ds, game = cohort._load_ds(p["id"])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    tr, te = cohort._subset(ds, train_mask), cohort._subset(ds, ~train_mask)
    pop.append(model.evaluate(te, prior))
    for l in lams:
        res[l].append(model.evaluate(te, model.fit(tr, prior=prior, lam=l)))
print(f"population        top1 {np.mean([r['top1'] for r in pop]):.4f}  logloss {np.mean([r['logloss'] for r in pop]):.4f}")
for l in lams:
    t1 = np.mean([r["top1"] for r in res[l]]); ll = np.mean([r["logloss"] for r in res[l]])
    wins = sum(a["logloss"] < b["logloss"] for a, b in zip(res[l], pop))
    print(f"lam {l:<6} top1 {t1:.4f}  logloss {ll:.4f}  beats population (logloss) {wins}/{len(pop)}")
