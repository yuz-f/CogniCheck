"""Non-linear residual head on top of the interpretable linear model.

score(move) = w · x  +  g(x)

``w`` is the linear personal model (fit by Newton in ``model.py``; it carries the
readable traits and their standard errors).  ``g`` is a small MLP (d → 32 → 32 → 1,
tanh) that captures interactions the linear terms miss.  It is pre-trained on the
population and fine-tuned per player with early stopping.  Training needs torch;
prediction only needs numpy, so a profile with a head works on any machine.

The residual keeps the linear part fixed, so the profile's traits do not change
when the head is added; in explanations it appears as one "pattern" term.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

HIDDEN = 32


def available() -> bool:
    if os.environ.get("COGNICHECK_NO_HEAD"):
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class HeadParams:
    """Plain-array MLP for numpy inference and JSON storage."""

    mu: np.ndarray
    sd: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (W, b) per layer

    def residual(self, X: np.ndarray) -> np.ndarray:
        h = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h[:, 0]

    def to_json(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "layers": [[W.tolist(), b.tolist()] for W, b in self.layers]}

    @classmethod
    def from_json(cls, d: dict) -> "HeadParams":
        return cls(np.array(d["mu"]), np.array(d["sd"]), [(np.array(W), np.array(b)) for W, b in d["layers"]])

    def save(self, path) -> None:
        np.savez_compressed(path, mu=self.mu, sd=self.sd, **{f"W{i}": W for i, (W, _) in enumerate(self.layers)}, **{f"b{i}": b for i, (_, b) in enumerate(self.layers)})

    @classmethod
    def load(cls, path) -> "HeadParams":
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith("W"))
        return cls(z["mu"], z["sd"], [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)])


# ------------------------------------------------------------------ training (torch)
def _torch_mlp(d: int, init: HeadParams | None):
    import torch

    mlp = torch.nn.Sequential(torch.nn.Linear(d, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, 1))
    if init is not None:
        lin = [m for m in mlp if isinstance(m, torch.nn.Linear)]
        for m, (W, b) in zip(lin, init.layers):
            m.weight.data = torch.tensor(W, dtype=torch.float32)
            m.bias.data = torch.tensor(b, dtype=torch.float32)
    else:
        torch.nn.init.zeros_(mlp[-1].weight)
        torch.nn.init.zeros_(mlp[-1].bias)
    return mlp


def _params(mlp, mu, sd) -> HeadParams:
    import torch

    layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in mlp if isinstance(m, torch.nn.Linear)]
    return HeadParams(np.asarray(mu, np.float64), np.asarray(sd, np.float64), layers)


def _seg_nll(z, starts, chosen, sizes):
    import torch

    seg = torch.repeat_interleave(torch.arange(len(starts)), sizes)
    m = torch.full((len(starts),), -1e30).scatter_reduce(0, seg, z, reduce="amax")
    lse = m + torch.log(torch.zeros(len(starts)).scatter_add(0, seg, torch.exp(z - m[seg])))
    return (lse - z[chosen]).mean()


def train(ds_fit, ds_val, w_lin: np.ndarray, init: HeadParams | None, mu=None, sd=None, epochs: int = 40, lr: float = 3e-4, wd: float = 1e-4, patience: int = 6, seed: int = 0) -> HeadParams:
    """Fit the residual g with the linear scores ``X @ w_lin`` held fixed.

    ``ds_fit`` / ``ds_val`` are ``model.Dataset``; validation is used for early
    stopping. ``mu``/``sd`` standardise the inputs (taken from ``init`` when given).
    """
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if init is not None:
        mu, sd = init.mu, init.sd
    mu_t, sd_t = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)

    def tens(ds):
        X = (torch.tensor(ds.X, dtype=torch.float32) - mu_t) / sd_t
        lin = torch.tensor(ds.X @ w_lin, dtype=torch.float32)
        sizes = torch.tensor(np.diff(np.append(ds.starts, len(ds.X))))
        return X, lin, torch.tensor(ds.starts), torch.tensor(ds.chosen), sizes

    Xf, lf, sf, cf, zf = tens(ds_fit)
    Xv, lv, sv, cv, zv = tens(ds_val)
    mlp = _torch_mlp(Xf.shape[1], init)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = float("inf"), {k: t.clone() for k, t in mlp.state_dict().items()}, 0
    with torch.no_grad():
        best = _seg_nll(lv + mlp(Xv).squeeze(-1), sv, cv, zv).item()  # do not accept a worse head than the start
    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(len(sf))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            rows = torch.cat([torch.arange(int(sf[j]), int(sf[j]) + int(zf[j])) for j in idx])
            sizes = zf[idx]
            starts = torch.cat([torch.tensor([0]), torch.cumsum(sizes, 0)[:-1]])
            chosen = starts + (cf[idx] - sf[idx])
            loss = _seg_nll(lf[rows] + mlp(Xf[rows]).squeeze(-1), starts, chosen, sizes)
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            v = _seg_nll(lv + mlp(Xv).squeeze(-1), sv, cv, zv).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in mlp.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    mlp.load_state_dict(best_state)
    return _params(mlp, mu, sd)


def standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(0), X.std(0) + 1e-3
