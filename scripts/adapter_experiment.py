"""Plan step 4: a personal low-rank correction to Maia-2's last layer.

Maia-2 scores move m in a position as W_m . e + b_m, where e is its 1024-number position
summary (cached by scripts/maia_embed.py). We add a low-rank correction

    delta_m = U_m . (V e)          U: 1880 x r, V: r x 1024

on top of the player's linear model. It is pre-trained on the training games of all cohort
players, then fine-tuned per player with an L2 pull back to the population values and early
stopping on the player's oldest training games. This is the new information the gate could
not add: how this player's choices differ from Maia-2's, learned from Maia-2's own reading of
the position.

Compared on the same test positions as scripts/evaluate.py (the most recent 15% of games):
  current   linear + non-linear head + book   (the shipped model)
  pop       linear + population correction + book
  personal  linear + fine-tuned correction + book
  both      linear + head + fine-tuned correction + book

"beyond Maia" = top-1 on decisions from ply 10 on where the player did not play Maia-2's top
move. Maia-2 alone scores 0 there, so any accuracy on them is personal.

  python scripts/adapter_experiment.py [n_players]
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort, model  # noqa: E402
from cognicheck.features import IDX, phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.predictor import BOOK_ALPHA  # noqa: E402
from cognicheck.profile import build_book, position_key  # noqa: E402

RANK = int(sys.argv[sys.argv.index("--rank") + 1]) if "--rank" in sys.argv else 32
POP_EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 6
N_MOVES = 1880
MAX_PLIES = 200


def prep(p):
    """Per player: linear model and head fitted on the training games, scores for every row."""
    ds, game = cohort._load_ds(p["id"])
    train = cohort.train_mask(p["id"], game)
    cohort._fill_habits(ds, train)
    tr = cohort._subset(ds, train)
    w = model.fit(tr, prior=cohort.load_population(p["rating"]), lam=0.01)
    head = cohort.fit_head(tr, game[train], w)
    X = ds.X
    sizes = np.diff(np.append(ds.starts, len(X)))
    lin = (X @ w).astype(np.float32)
    hres = np.zeros(len(X), np.float32)
    if head is not None:
        te_rows = np.repeat(~train, sizes)
        hres[te_rows] = head.residual(X[te_rows])
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    hold = cohort.holdout_games(p["id"])
    book = build_book([g for i, g in enumerate(pgs) if i not in hold])
    # book counts and phase for the test decisions, by plain board iteration (same order as the analysis)
    bookc, phases, k = {}, {}, 0
    for gi, pg in enumerate(pgs):
        board = pg.game.board()
        for ply, node in enumerate(pg.game.mainline()):
            if ply >= MAX_PLIES:
                break
            if board.turn == pg.color and board.legal_moves.count() > 1:
                if gi in hold:
                    counts = book.get(position_key(board), {})
                    if counts:
                        bookc[k] = np.array([counts.get(m.uci(), 0) for m in board.legal_moves], float)
                    phases[k] = phase(board)
                k += 1
            board.push(node.move)
    if k != len(ds.starts):
        raise RuntimeError(f"{p['id']}: {k} decisions vs {len(ds.starts)}")
    c_off = ds.chosen - ds.starts
    blunder = X[ds.chosen, IDX["eng_loss"]] >= 2
    mtop = X[:, IDX["maia_top"]]
    maia_on = np.add.reduceat(np.abs(X[:, IDX["maia_logp"]]), ds.starts) > 0
    beyond = maia_on & (mtop[ds.chosen] == 0)
    games_train = sorted(set(game[train].tolist()))
    val_games = set(games_train[-max(1, len(games_train) // 7):])  # oldest training games, as fit_head
    return {"id": p["id"], "starts": ds.starts, "sizes": sizes, "c": c_off, "game": game, "train": train,
            "ft_val": np.isin(game, list(val_games)) & train, "lin": lin, "hres": hres, "book": bookc,
            "phase": phases, "blunder": blunder, "beyond": beyond}


# ------------------------------------------------------------------ torch part
def _batch(D, idx, torch):
    """Padded tensors for positions idx of the pooled data D."""
    st, sz = D["starts"][idx], D["sizes"][idx]
    L = int(sz.max())
    ar = np.arange(L)
    mask = ar[None, :] < sz[:, None]
    rows = np.where(mask, st[:, None] + ar[None, :], 0)
    return (torch.from_numpy(D["E"][D["epos"][idx]].astype(np.float32)),
            torch.from_numpy(D["MI"][rows].astype(np.int64)),
            torch.from_numpy(np.where(mask, D["base"][rows], -1e9).astype(np.float32)),
            torch.from_numpy(D["c"][idx].astype(np.int64)))


def _nll(params, batch, torch):
    U, V = params
    E, MI, base, c = batch
    h = E @ V.T  # B x r
    delta = (U[MI] * h[:, None, :]).sum(-1)  # B x L
    z = base + delta
    return torch.nn.functional.cross_entropy(z, c, reduction="sum")


def train_adapter(D, fit_idx, val_idx, init=None, l2=0.0, lr=1e-3, epochs=6, patience=2, batch=512, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    if init is None:
        U = (torch.randn(N_MOVES, RANK, generator=g) * 0.01).requires_grad_()
        V = (torch.randn(RANK, 1024, generator=g) * 0.01).requires_grad_()
    else:
        U, V = (t.clone().requires_grad_() for t in init)
    ref = None if init is None else [t.clone() for t in init]
    opt = torch.optim.Adam([U, V], lr=lr)
    rng = np.random.default_rng(seed)

    def val_loss():
        with torch.no_grad():
            tot = sum(float(_nll((U, V), _batch(D, val_idx[i:i + 2048], torch), torch)) for i in range(0, len(val_idx), 2048))
        return tot / max(1, len(val_idx))

    best, best_p, bad = val_loss(), (U.detach().clone(), V.detach().clone()), 0
    for _ in range(epochs):
        perm = rng.permutation(fit_idx)
        for i in range(0, len(perm), batch):
            opt.zero_grad()
            loss = _nll((U, V), _batch(D, perm[i:i + batch], torch), torch) / len(perm[i:i + batch])
            if ref is not None and l2 > 0:
                loss = loss + l2 * (((U - ref[0]) ** 2).sum() + ((V - ref[1]) ** 2).sum())
            loss.backward()
            opt.step()
        v = val_loss()
        print(f"    epoch val NLL {v:.4f}", file=sys.stderr)
        if v < best - 1e-5:
            best, best_p, bad = v, (U.detach().clone(), V.detach().clone()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_p, best


def delta_rows(params, D, idx, torch):
    """Correction per row for positions idx, as a flat array aligned with their rows."""
    U, V = params
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            E, MI, _, _ = _batch(D, j, torch)
            d = (U[MI] * (E @ V.T)[:, None, :]).sum(-1).numpy()
            for r, s in zip(d, D["sizes"][j]):
                out.append(r[:s])
    return out


def pooled(rows):
    """Concatenate per-player data into one indexable set of positions."""
    D = {"starts": [], "sizes": [], "c": [], "base": [], "MI": [], "E": [], "epos": []}
    row_off = pos_off = 0
    owner = []
    for pi, r in enumerate(rows):
        emb = np.load(cohort.ANALYSIS / f"{r['id']}.emb.npz")
        D["starts"].append(r["starts"] + row_off)
        D["sizes"].append(r["sizes"])
        D["c"].append(r["c"])
        D["base"].append(r["lin"])
        D["MI"].append(emb["MI"])
        D["E"].append(emb["E"])
        D["epos"].append(np.arange(len(r["starts"])) + pos_off)
        owner.append(np.full(len(r["starts"]), pi))
        row_off += len(r["lin"])
        pos_off += len(r["starts"])
    D = {k: np.concatenate(v) for k, v in D.items()}
    return D, np.concatenate(owner)


def metrics(r, probs):
    test = np.flatnonzero(~r["train"])
    hit = np.array([int(np.argmax(probs[k])) == r["c"][k] for k in test], float)
    ll = np.array([-np.log(max(probs[k][r["c"][k]], 1e-9)) for k in test])
    out = {"top1": hit.mean(), "logloss": ll.mean()}
    for ph in ("opening", "middlegame", "endgame"):
        m = np.array([r["phase"][k] == ph for k in test])
        out[ph] = hit[m].mean() if m.any() else np.nan
    for name in ("blunder", "beyond"):
        m = r[name][test]
        out[name] = hit[m].mean() if m.any() else np.nan
    return out


def with_book(r, k, z):
    p = model.softmax(z)
    cv = r["book"].get(k)
    if cv is not None:
        p = (cv + BOOK_ALPHA * p) / (cv.sum() + BOOK_ALPHA)
    return p


def ci(x, rng):
    bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.emb.npz").exists()]
    nums = [a for i, a in enumerate(sys.argv[1:], 1) if a.isdigit() and sys.argv[i - 1] not in ("--rank", "--epochs")]
    if nums:
        players = players[: int(nums[0])]
    t0 = time.time()
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        rows = list(ex.map(prep, players))
    print(f"prepared {len(rows)} players in {time.time() - t0:.0f}s", file=sys.stderr)
    D, owner = pooled(rows)
    pos_train = np.concatenate([r["train"] for r in rows])
    rng = np.random.default_rng(0)
    tr_idx = np.flatnonzero(pos_train)
    rng.shuffle(tr_idx)
    n_val = max(1, len(tr_idx) // 20)
    t0 = time.time()
    pop, pop_val = train_adapter(D, tr_idx[n_val:], tr_idx[:n_val], epochs=POP_EPOCHS, patience=3)
    print(f"population correction trained on {len(tr_idx) - n_val} positions in {time.time() - t0:.0f}s, val NLL {pop_val:.4f}", file=sys.stderr)
    torch.save({"U": pop[0], "V": pop[1]}, cohort.REAL / "adapter_population.pt")

    res = {"current": [], "pop": [], "personal": [], "both": []}
    for pi, r in enumerate(rows):
        idx = np.flatnonzero(owner == pi)
        tr_mask = r["train"] & ~r["ft_val"]
        fit_idx, val_idx = idx[tr_mask], idx[r["ft_val"]]
        ft, _ = train_adapter(D, fit_idx, val_idx, init=pop, l2=1e-3, lr=3e-4, epochs=30, patience=3, batch=256, seed=pi)
        test_local = np.flatnonzero(~r["train"])
        d_pop = delta_rows(pop, D, idx[test_local], torch)
        d_ft = delta_rows(ft, D, idx[test_local], torch)
        probs = {m: {} for m in res}
        for j, k in enumerate(test_local):
            s, e = r["starts"][k], r["starts"][k] + r["sizes"][k]
            lin, hr = r["lin"][s:e].astype(np.float64), r["hres"][s:e]
            probs["current"][k] = with_book(r, k, lin + hr)
            probs["pop"][k] = with_book(r, k, lin + d_pop[j])
            probs["personal"][k] = with_book(r, k, lin + d_ft[j])
            probs["both"][k] = with_book(r, k, lin + hr + d_ft[j])
        for m in res:
            res[m].append(metrics(r, probs[m]))
        print(f"  {pi + 1}/{len(rows)} {r['id']}: current {res['current'][-1]['top1']:.3f} pop {res['pop'][-1]['top1']:.3f} personal {res['personal'][-1]['top1']:.3f}", file=sys.stderr)

    keys = ["top1", "logloss", "opening", "middlegame", "endgame", "blunder", "beyond"]
    print(f"\n{len(rows)} players, test = most recent 15% of games. Mean over players.")
    print(f"{'model':10s} " + " ".join(f"{k:>10s}" for k in keys))
    for m in res:
        print(f"{m:10s} " + " ".join(f"{np.nanmean([x[k] for x in res[m]]):10.4f}" for k in keys))
    summary = {}
    for m in ("pop", "personal", "both"):
        for k in ("top1", "logloss", "middlegame", "endgame", "beyond"):
            d = np.array([a[k] - b[k] for a, b in zip(res[m], res["current"])])
            d = d[~np.isnan(d)]
            lo, hi = ci(d, rng)
            summary[f"{m}_minus_current_{k}"] = [float(d.mean()), float(lo), float(hi)]
            print(f"{m} minus current, {k:10s}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (cohort.REAL / f"adapter_experiment_r{RANK}_e{POP_EPOCHS}.json").write_text(json.dumps({"summary": summary, "rows": [
        {"id": r["id"], **{m: {k: float(res[m][i][k]) for k in keys} for m in res}} for i, r in enumerate(rows)]}, indent=1))
