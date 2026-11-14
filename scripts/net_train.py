"""Train and evaluate the from-scratch network (cognicheck/chessnet.py).

  python scripts/net_train.py --data data/real/net/cohort.npz --blocks 6 --channels 64 --epochs 8
  python scripts/net_train.py --train-shards data/real/net/lichess --epochs 2 --device mps   # stage B on a Mac

--device auto picks Apple's GPU (mps) or CUDA when present, else the CPU. With --train-shards the
network trains on the Lichess shards (scripts/net_lichess.py) and is still tested on the cohort.

Validation for early stopping: 2% of the training games (never the test games). The test set is
the cohort's held-out decisions, the same positions as scripts/evaluate.py, so the numbers are
directly comparable: means over players of top-1, top-3, log-loss, top-1 by phase and on blunders.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from cognicheck import chessnet  # noqa: E402

PHASES = ["opening", "middlegame", "endgame"]


class Split:
    def __init__(self, z, prefix, idx=None):
        self.planes = z[prefix + "planes"]
        self.elo = z[prefix + "elo"].astype(np.float32)
        self.off = z[prefix + "off"]
        self.legal = z[prefix + "legal"]  # int16; converted per batch to save memory (20M positions: 1.3 GB instead of 5.4 GB)
        self.target = z[prefix + "target"]
        self.idx = np.arange(len(self.target)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def batch(self, j):
        """Tensors for positions j (indices into the full split)."""
        x = np.unpackbits(self.planes[j], axis=1)[:, : chessnet.N_PLANES * 64].reshape(-1, chessnet.N_PLANES, 8, 8)
        mask = np.zeros((len(j), chessnet.N_MOVES), dtype=bool)
        tgt = np.empty(len(j), np.int64)
        for r, k in enumerate(j):
            lm = self.legal[self.off[k]:self.off[k + 1]].astype(np.int64)
            mask[r, lm] = True
            tgt[r] = lm[self.target[k]]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.elo[j]), torch.from_numpy(mask), torch.from_numpy(tgt)


def load_shards(folder):
    """Concatenate Lichess shards into one dict with the train_ prefix of a cohort file."""
    files = sorted(Path(folder).glob("shard_*.npz"))
    parts = [np.load(f) for f in files]
    d = {"train_planes": np.concatenate([p["planes"] for p in parts]), "train_elo": np.concatenate([p["elo"] for p in parts]),
         "train_legal": np.concatenate([p["legal"] for p in parts]), "train_target": np.concatenate([p["target"] for p in parts])}
    offs, base = [], 0
    for p in parts:
        offs.append(p["off"][:-1] + base)
        base += int(p["off"][-1])
    d["train_off"] = np.concatenate(offs + [np.array([base])])
    print(f"{len(files)} shards, {len(d['train_target']):,} positions", file=sys.stderr)
    return d


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = torch.device("cpu")


def run_eval(net, split, bs=2048):
    """Per-position probability of the move played, hit flags (top-1, top-3) and max probability."""
    net.eval()
    p_true, hit1, hit3 = [], [], []
    with torch.no_grad():
        for i in range(0, len(split), bs):
            j = split.idx[i:i + bs]
            x, e, m, t = (v.to(DEV) for v in split.batch(j))
            logp = torch.log_softmax(net(x, e, m), dim=1)
            p_true.append(logp.gather(1, t[:, None]).squeeze(1).exp().cpu().numpy())
            top = logp.topk(3, dim=1).indices
            hit1.append((top[:, 0] == t).cpu().numpy())
            hit3.append((top == t[:, None]).any(1).cpu().numpy())
    net.train()
    return np.concatenate(p_true), np.concatenate(hit1).astype(float), np.concatenate(hit3).astype(float)


def cohort_metrics(z, p_true, h1, h3):
    player, ph, bl = z["test_player"], z["test_phase"], z["test_blunder"]
    rows = []
    for pi in np.unique(player):
        m = player == pi
        r = {"top1": h1[m].mean(), "top3": h3[m].mean(), "logloss": float(-np.log(np.maximum(p_true[m], 1e-9)).mean())}
        for k, name in enumerate(PHASES):
            mm = m & (ph == k)
            r[name] = h1[mm].mean() if mm.any() else np.nan
        mb = m & bl
        r["blunder"] = h1[mb].mean() if mb.any() else np.nan
        rows.append(r)
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/net/cohort.npz")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training positions (smoke tests)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--train-shards", default="", help="folder of Lichess shards to train on instead of the cohort games")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    DEV = pick_device(a.device)
    z = dict(np.load(a.data))
    if a.train_shards:
        z.update(load_shards(a.train_shards))
    n = len(z["train_target"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    if a.limit:
        perm = perm[: a.limit]
    n_val = max(1000, len(perm) // 50)
    train, val = Split(z, "train_", perm[n_val:]), Split(z, "train_", perm[:n_val])
    test = Split(z, "test_")
    net = chessnet.build(a.blocks, a.channels).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    steps = a.epochs * (len(train) // a.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    name = a.tag or f"b{a.blocks}c{a.channels}"
    out = Path(a.data).parent / f"net_{name}.pt"
    print(f"{n_par:,} parameters, {len(train):,} training positions, {steps:,} steps, device {DEV}", file=sys.stderr)
    best, t0, step = 1e9, time.time(), 0
    for ep in range(a.epochs):
        order = rng.permutation(train.idx)
        for i in range(0, len(order) - a.batch + 1, a.batch):
            x, e, m, t = (v.to(DEV) for v in train.batch(order[i:i + a.batch]))
            loss = torch.nn.functional.cross_entropy(net(x, e, m), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                el = time.time() - t0
                print(f"    step {step}/{steps} loss {loss.item():.3f} ({el:.0f}s, {step * a.batch / el:,.0f} positions/s)", file=sys.stderr)
        pv, hv, _ = run_eval(net, val)
        vl = float(-np.log(np.maximum(pv, 1e-9)).mean())
        print(f"  epoch {ep + 1}: val log-loss {vl:.4f} top-1 {hv.mean():.4f} ({time.time() - t0:.0f}s)", file=sys.stderr)
        if vl < best:
            best = vl
            torch.save({"cfg": net.cfg, "state": net.state_dict()}, out)
    ck = torch.load(out, map_location=DEV)
    net.load_state_dict(ck["state"])
    pt, h1, h3 = run_eval(net, test)
    summ, rows = cohort_metrics(z, pt, h1, h3)
    print(f"\n{name}: {n_par:,} parameters. Cohort test decisions, mean over players:")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in summ.items()))
    res = {"name": name, "parameters": n_par, "train_positions": len(train), "epochs": a.epochs, "val_logloss": best,
           "test": summ, "rows": rows}
    (Path(a.data).parent / f"result_{name}.json").write_text(json.dumps(res, indent=1))
