"""Accuracy experiments on a few players: multi-Elo Maia columns and a non-linear head.

head variants (all evaluated on the same held-out games as the cohort fit):
  linear            the current conditional logit (prior-shrunk Newton fit)
  linear+elo3       same, plus Maia log-probabilities at rating-400 / +400
  mlp               linear + small MLP residual, trained per player from scratch
  mlp(pop)          MLP pre-trained on other players, then fine-tuned per player
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import IDX  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.neural import Neural  # noqa: E402
from cognicheck.profile import iter_decisions  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-CogniCheck/5f260670-b7ec-53c5-b964-37de6d06eaf3/scratchpad")
torch.set_num_threads(2)
torch.manual_seed(0)


def elo3_columns(p, neural):
    """Maia log-prob at rating-400 and rating+400, aligned with stored rows (cached)."""
    out = SCRATCH / f"{p['id']}.elo3.npy"
    if out.exists():
        return np.load(out)
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    cols = []
    for gi, pg in enumerate(pgs):
        for d in iter_decisions(pg, gi, None):
            if len(d.pa.moves) <= 1:
                continue
            b = d.pa.board
            if b.ply() < 10:
                cols.append(np.zeros((len(d.pa.moves), 2), np.float32))
                continue
            lo, _ = neural.features(b, d.pa.moves, max(800, p["rating"] - 400), max(800, p["rating"] - 400))
            hi, _ = neural.features(b, d.pa.moves, p["rating"] + 400, p["rating"] + 400)
            cols.append(np.stack([lo, hi], 1) / 3.0)
    M = np.concatenate(cols)
    np.save(out, M)
    return M


def split(p, extra=None):
    ds, game = cohort._load_ds(p["id"])
    if extra is not None:
        assert len(extra) == len(ds.X)
        ds.X = np.hstack([ds.X, extra])
    n_games = json.loads((cohort.ANALYSIS / f"{p['id']}.json").read_text())["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * 0.15)))])
    train_mask = np.array([g not in hold for g in game])
    cohort._fill_habits(ds, train_mask)
    # validation = last 15% of the training games (for early stopping)
    train_games = sorted(set(game[train_mask]))
    val = set(train_games[-max(1, len(train_games) // 7):])
    val_mask = np.array([g in val for g in game]) & train_mask
    fit_mask = train_mask & ~val_mask
    return cohort._subset(ds, fit_mask), cohort._subset(ds, val_mask), cohort._subset(ds, ~train_mask), cohort._subset(ds, train_mask)


class Head(torch.nn.Module):
    def __init__(self, d, hidden=32, w0=None):
        super().__init__()
        self.lin = torch.nn.Linear(d, 1, bias=False)
        if w0 is not None:
            self.lin.weight.data = torch.tensor(w0, dtype=torch.float32)[None]
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.lin(x).squeeze(-1) + self.mlp(x).squeeze(-1)


def seg_nll(z, starts, chosen, sizes):
    seg = torch.repeat_interleave(torch.arange(len(starts)), sizes)
    lse = torch.zeros(len(starts)).scatter_reduce(0, seg, z, reduce="amax", include_self=False)
    lse = lse + torch.log(torch.zeros(len(starts)).scatter_add(0, seg, torch.exp(z - lse[seg])))
    return (lse - z[chosen]).mean()


def to_t(ds):
    sizes = torch.tensor(np.diff(np.append(ds.starts, len(ds.X))))
    return torch.tensor(ds.X, dtype=torch.float32), torch.tensor(ds.starts), torch.tensor(ds.chosen), sizes


def top1(head, ds, mu, sd):
    X, starts, chosen, sizes = to_t(ds)
    with torch.no_grad():
        z = head((X - mu) / sd).numpy()
    ends = np.append(ds.starts[1:], len(ds.X))
    hits = [int(np.argmax(z[s:e])) == c - s for s, e, c in zip(ds.starts, ends, ds.chosen)]
    ll = seg_nll(torch.tensor(z), starts, chosen, sizes).item()
    return float(np.mean(hits)), ll


def train(head, fit, val, mu, sd, epochs=60, lr=1e-3, wd=1e-3, lr_lin=None):
    Xf, sf, cf, zf = to_t(fit)
    Xv, sv, cv, zv = to_t(val)
    Xf, Xv = (Xf - mu) / sd, (Xv - mu) / sd
    params = [{"params": head.mlp.parameters(), "lr": lr}, {"params": head.lin.parameters(), "lr": lr_lin or lr}]
    opt = torch.optim.Adam(params, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        head.train()
        # minibatch over positions
        perm = torch.randperm(len(sf))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            rows = torch.cat([torch.arange(int(sf[j]), int(sf[j]) + int(zf[j])) for j in idx])
            sizes = zf[idx]
            starts = torch.cat([torch.tensor([0]), torch.cumsum(sizes, 0)[:-1]])
            chosen = starts + (cf[idx] - sf[idx])
            loss = seg_nll(head(Xf[rows]), starts, chosen, sizes)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = seg_nll(head(Xv), sv, cv, zv).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    head.load_state_dict(best_state)
    return head


def main(ids):
    players = {p["id"]: p for p in json.loads(cohort.PLAYERS.read_text())}
    neural = Neural()
    # population pre-training set: 16 other players (fit part only)
    others = [p for p in players.values() if p["id"] not in ids][:16]
    pool = []
    for p in others:
        f, v, t, _ = split(p)
        pool.append(f)
    pool_ds = cohort._pool(pool)
    mu = torch.tensor(pool_ds.X.mean(0), dtype=torch.float32)
    sd = torch.tensor(pool_ds.X.std(0) + 1e-3, dtype=torch.float32)
    pop_prior = cohort.load_population()
    t0 = time.time()
    pop_head = Head(pool_ds.X.shape[1], w0=(pop_prior * sd.numpy()))  # linear part in standardized units
    # use a slice of the pool as validation for early stopping
    n = len(pool_ds)
    cut = int(n * 0.9)
    fit_pool = model.Dataset(pool_ds.X[: pool_ds.starts[cut]], pool_ds.starts[:cut], pool_ds.chosen[:cut])
    val_pool = model.Dataset(pool_ds.X[pool_ds.starts[cut]:], pool_ds.starts[cut:] - pool_ds.starts[cut], pool_ds.chosen[cut:] - pool_ds.starts[cut])
    pop_head = train(pop_head, fit_pool, val_pool, mu, sd, epochs=20, lr=1e-3, wd=1e-4)
    print(f"population MLP trained on {len(pool_ds)} positions in {time.time() - t0:.0f}s; pool val top1 {top1(pop_head, val_pool, mu, sd)}", file=sys.stderr)

    for pid in ids:
        p = players[pid]
        res = {}
        band = cohort.load_population(p["rating"])
        fit, val, test, train_all = split(p)
        w = model.fit(train_all, prior=band, lam=0.01)
        res["linear"] = model.evaluate(test, w)["top1"], model.evaluate(test, w)["logloss"]
        # multi-Elo Maia
        extra = elo3_columns(p, neural)
        fit3, val3, test3, train3 = split(p, extra)
        band3 = np.append(band, [0.0, 0.0])
        w3 = model.fit(train3, prior=band3, lam=0.01)
        e3 = model.evaluate(test3, w3)
        res["linear+elo3"] = e3["top1"], e3["logloss"]
        print(f"  elo3 weights for {p['username']}: maia_logp {w3[IDX['maia_logp']]:+.2f} lo {w3[-2]:+.2f} hi {w3[-1]:+.2f}", file=sys.stderr)
        # MLP from scratch (linear part initialised at the band prior)
        h = Head(train_all.X.shape[1], w0=(band * sd.numpy()))
        h = train(h, fit, val, mu, sd, epochs=60, lr=1e-3, wd=1e-3)
        res["mlp"] = top1(h, test, mu, sd)
        # MLP pre-trained on the population, fine-tuned
        h2 = Head(train_all.X.shape[1])
        h2.load_state_dict(pop_head.state_dict())
        h2 = train(h2, fit, val, mu, sd, epochs=40, lr=3e-4, wd=1e-4)
        res["mlp(pop)"] = top1(h2, test, mu, sd)
        res["mlp(pop) no finetune"] = top1(pop_head, test, mu, sd)
        print(f"\n{p['username']} ({p['rating']}), {len(test)} held-out positions")
        for k, (t1, ll) in res.items():
            print(f"  {k:24s} top1 {t1:.4f}  logloss {ll:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pikingkong", "rodri80", "shafar16"])
