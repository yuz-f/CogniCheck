"""Train and evaluate the from-scratch network (cognicheck/chessnet.py).

  python scripts/net_train.py --data data/real/net/cohort.npz --blocks 6 --channels 64 --epochs 8
  python scripts/net_train.py --train-shards data/real/net/lichess --epochs 2 --device mps   # stage B on a Mac
  python scripts/net_train.py --init data/real/net/net_B_b6c64.pt --epochs 3 --lr 5e-4 --tag D_cohort   # fine-tune on the cohort
  python scripts/net_train.py --init data/real/net/net_B_b10c128.pt --players --epochs 3 --lr 5e-4 --tag P_b10c128   # ... knowing who moves
  python scripts/net_train.py --train-shards data/real/net/lichess --context --blocks 10 --channels 128 --mirror --epochs 2 --device mps --tag C_b10c128   # stage C
  python scripts/net_train.py --train-shards data/real/net/lichess --context --arch transformer --blocks 8 --channels 256 --lr 1e-3 --mirror --epochs 2 --device mps --tag T_t8d256

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
        self.player = z.get(prefix + "mover", z.get(prefix + "player"))  # mover's cohort index (n_players = anyone else), or None
        self.unknown = 0
        self.hist = self.ctx = self.think = None  # stage C inputs (chessnet.context), set by add_context()
        self.idx = np.arange(len(self.target)) if idx is None else idx
        self.mirror = None  # set to chessnet.mirror_lr_table() to augment
        self.rng = np.random.default_rng(1)

    def add_context(self, z, prefix):
        self.hist, self.ctx, self.think = chessnet.context(z[prefix + "game"], z[prefix + "ply"], z[prefix + "clock"],
                                                           self.legal, self.off, self.target)

    def __len__(self):
        return len(self.idx)

    def batch(self, j):
        """Tensors for positions j (indices into the full split)."""
        x = np.unpackbits(self.planes[j], axis=1)[:, : chessnet.N_PLANES * 64].reshape(-1, chessnet.N_PLANES, 8, 8)
        if self.hist is not None:
            x = np.concatenate([x, chessnet.history_planes(self.hist[j])], axis=1)
        mask = np.zeros((len(j), chessnet.N_MOVES), dtype=bool)
        tgt = np.empty(len(j), np.int64)
        for r, k in enumerate(j):
            lm = self.legal[self.off[k]:self.off[k + 1]].astype(np.int64)
            mask[r, lm] = True
            tgt[r] = lm[self.target[k]]
        if self.mirror is not None:  # left-right mirror half of the positions without castling rights
            flip = ~x[:, 12:16].any(axis=(1, 2, 3)) & (self.rng.random(len(j)) < 0.5)
            if self.player is not None:  # a mirrored board would blur a player's personal move bias
                flip &= self.player[j] == self.unknown
            if flip.any():
                x[flip] = x[flip][:, :, :, ::-1]
                mask[flip] = mask[flip][:, self.mirror]  # the table is its own inverse
                tgt[flip] = self.mirror[tgt[flip]]
        pl = self.player[j].astype(np.int64) if self.player is not None else np.full(len(j), self.unknown, np.int64)
        ctx = self.ctx[j] if self.ctx is not None else np.zeros((len(j), chessnet.N_CTX), np.float32)
        think = self.think[j] if self.think is not None else np.full(len(j), np.nan, np.float32)
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.elo[j]), torch.from_numpy(mask), torch.from_numpy(tgt),
                torch.from_numpy(pl), torch.from_numpy(ctx), torch.from_numpy(think))


def load_shards(folder):
    """Concatenate Lichess shards into one dict with the train_ prefix of a cohort file."""
    files = sorted(Path(folder).glob("shard_*.npz"))
    parts = [np.load(f) for f in files]
    d = {"train_planes": np.concatenate([p["planes"] for p in parts]), "train_elo": np.concatenate([p["elo"] for p in parts]),
         "train_legal": np.concatenate([p["legal"] for p in parts]), "train_target": np.concatenate([p["target"] for p in parts]),
         "train_game": np.concatenate([p["game"] for p in parts]), "train_ply": np.concatenate([p["ply"] for p in parts]),
         "train_clock": np.concatenate([p["clock"] for p in parts])}
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
    p_true, hit1, hit3, th_p, th_t = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(split), bs):
            j = split.idx[i:i + bs]
            x, e, m, t, pl, c, th = (v.to(DEV) for v in split.batch(j))
            logits, tp = net(x, e, m, pl, c, aux=True)
            if tp is not None:
                th_p.append(tp.cpu().numpy())
                th_t.append(th.cpu().numpy())
            logp = torch.log_softmax(logits, dim=1)
            p_true.append(logp.gather(1, t[:, None]).squeeze(1).exp().cpu().numpy())
            top = logp.topk(3, dim=1).indices
            hit1.append((top[:, 0] == t).cpu().numpy())
            hit3.append((top == t[:, None]).any(1).cpu().numpy())
    net.train()
    if th_p:
        tp, tt = np.concatenate(th_p), np.concatenate(th_t)
        ok = np.isfinite(tt)
        if ok.sum() > 2:
            run_eval.think_r = float(np.corrcoef(tp[ok], tt[ok])[0, 1])
    return np.concatenate(p_true), np.concatenate(hit1).astype(float), np.concatenate(hit3).astype(float)


run_eval.think_r = None


def cohort_metrics(z, p_true, h1, h3, idx):
    player, ph, bl = z["test_player"][idx], z["test_phase"][idx], z["test_blunder"][idx]
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
    ap.add_argument("--init", default="", help="start from a saved network (fine-tuning); its size overrides --blocks/--channels")
    ap.add_argument("--players", action="store_true", help="give the network the mover's identity (cohort data only)")
    ap.add_argument("--player-lr", type=float, default=10.0, help="learning-rate multiplier for the new player parameters")
    ap.add_argument("--arch", default="resnet", help="resnet or transformer (then --blocks = layers, --channels = width)")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--stem", type=int, default=0, help="transformer: conv blocks before the attention layers (hybrid)")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay (0.01 suits the transformer)")
    ap.add_argument("--context", action="store_true", help="stage C: last moves and clocks as input, think time as second output")
    ap.add_argument("--think-weight", type=float, default=0.1, help="weight of the think-time loss (with --context)")
    ap.add_argument("--mirror", action="store_true", help="left-right mirror augmentation for positions without castling rights")
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
    if "test_eval" in z:  # newer cohort files keep the opponents' moves of held-out games as history rows
        test.idx = np.nonzero(z["test_eval"])[0]
    if a.init and not a.context:
        a.context = bool(torch.load(a.init, map_location="cpu")["cfg"].get("hist", 0))
    if a.context:
        train.add_context(z, "train_")
        val.hist, val.ctx, val.think = train.hist, train.ctx, train.think
        test.add_context(z, "test_")
        print(f"stage C context: think time known for {np.isfinite(train.think).mean():.0%} of training positions", file=sys.stderr)
    n_pl = len(z["player_ids"]) if a.players else 0
    for sp in (train, val, test):
        sp.unknown = n_pl
        if not a.players:
            sp.player = None
    if a.players and train.player is None:
        sys.exit("--players needs a cohort file with train_player (rebuild it with scripts/net_data.py)")
    if a.mirror:
        train.mirror = chessnet.mirror_lr_table()
    if a.init:
        ck0 = torch.load(a.init, map_location="cpu")
        c0 = ck0["cfg"]
        net = chessnet.build(c0["blocks"], c0["channels"], n_pl or c0.get("n_players", 0), c0.get("hist", 0), c0.get("arch", "resnet"), c0.get("heads", 8), c0.get("stem", 0))
        missing, _ = net.load_state_dict(ck0["state"], strict=False)
        if any(not k.startswith("pl_") for k in missing):
            sys.exit(f"{a.init} does not fit: missing {missing}")
        net = net.to(DEV)
        print(f"starting from {a.init}" + (f", new player parameters for {n_pl} players" if missing else ""), file=sys.stderr)
    else:
        net = chessnet.build(a.blocks, a.channels, n_pl, chessnet.N_HIST if a.context else 0, a.arch, a.heads, a.stem).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    steps = a.epochs * (len(train) // a.batch)
    pl_par = [p for k, p in net.named_parameters() if k.startswith("pl_")]
    rest = [p for k, p in net.named_parameters() if not k.startswith("pl_")]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": pl_par, "lr": a.lr * a.player_lr, "weight_decay": 0.0}] if pl_par else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    name = a.tag or (f"t{a.blocks}d{a.channels}" if a.arch == "transformer" else f"b{a.blocks}c{a.channels}")
    out = Path(a.data).parent / f"net_{name}.pt"
    print(f"{n_par:,} parameters, {len(train):,} training positions, {steps:,} steps, device {DEV}", file=sys.stderr)
    best, t0, step = 1e9, time.time(), 0
    for ep in range(a.epochs):
        order = rng.permutation(train.idx)
        for i in range(0, len(order) - a.batch + 1, a.batch):
            x, e, m, t, pl, c, th = (v.to(DEV) for v in train.batch(order[i:i + a.batch]))
            logits, th_pred = net(x, e, m, pl, c, aux=True)
            loss = torch.nn.functional.cross_entropy(logits, t)
            if th_pred is not None and a.think_weight > 0:
                ok = torch.isfinite(th)
                if ok.any():
                    loss = loss + a.think_weight * torch.nn.functional.smooth_l1_loss(th_pred[ok], th[ok])
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
    summ, rows = cohort_metrics(z, pt, h1, h3, test.idx)
    if run_eval.think_r is not None:
        summ["think_r"] = run_eval.think_r
    print(f"\n{name}: {n_par:,} parameters. Cohort test decisions, mean over players:")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in summ.items()))
    res = {"name": name, "parameters": n_par, "train_positions": len(train), "epochs": a.epochs, "val_logloss": best,
           "test": summ, "rows": rows}
    (Path(a.data).parent / f"result_{name}.json").write_text(json.dumps(res, indent=1))
