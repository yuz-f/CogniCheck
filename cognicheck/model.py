"""Conditional-logit ("which of these legal moves?") model in plain numpy.

P(move | position) = softmax_i(w . x_i) over the legal moves of a position.

A player's weights are fit with an L2 penalty pulling them towards a prior
(the population model): with few games the player looks like "an average
player"; with many games their quirks dominate. The deviation w - w_prior is
exactly what makes the player *specific*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray  # (total_moves, d)
    starts: np.ndarray  # (n_positions,) start offset of each position in X
    chosen: np.ndarray  # (n_positions,) global row index of the move played

    @classmethod
    def from_positions(cls, positions: list[tuple[np.ndarray, int]]) -> "Dataset":
        positions = [(x, c) for x, c in positions if len(x) > 1]
        if not positions:
            return cls(np.zeros((0, 0), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64))
        sizes = np.array([len(x) for x, _ in positions])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        X = np.concatenate([x for x, _ in positions]).astype(np.float64)
        chosen = starts + np.array([c for _, c in positions])
        return cls(X, starts.astype(np.int64), chosen.astype(np.int64))

    def __len__(self) -> int:
        return len(self.starts)


def _seg_logsumexp(z: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg_max = np.maximum.reduceat(z, starts)
    sizes = np.diff(np.append(starts, len(z)))
    zz = z - np.repeat(seg_max, sizes)
    e = np.exp(zz)
    s = np.add.reduceat(e, starts)
    lse = np.log(s) + seg_max
    probs = e / np.repeat(s, sizes)
    return lse, probs


def nll_and_grad(w: np.ndarray, ds: Dataset, prior: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    z = ds.X @ w
    lse, probs = _seg_logsumexp(z, ds.starts)
    n = len(ds)
    nll = float((lse - z[ds.chosen]).sum()) / n
    grad = ds.X.T @ probs
    grad -= ds.X[ds.chosen].sum(axis=0)
    grad /= n
    diff = w - prior
    nll += 0.5 * lam * float(diff @ diff)
    grad += lam * diff
    return nll, grad


def _hessian(w: np.ndarray, ds: Dataset, lam: float) -> np.ndarray:
    z = ds.X @ w
    _, probs = _seg_logsumexp(z, ds.starts)
    PX = probs[:, None] * ds.X
    M = np.add.reduceat(PX, ds.starts, axis=0)  # per-position expected feature vector
    H = ds.X.T @ PX - M.T @ M
    H /= len(ds)
    H += lam * np.eye(len(w))
    return H


def fit(ds: Dataset, prior: np.ndarray | None = None, lam: float = 0.01, iters: int = 50, tol: float = 1e-7) -> np.ndarray:
    """Damped Newton on the penalised mean negative log-likelihood (convex)."""
    d = ds.X.shape[1] if ds.X.ndim == 2 and ds.X.size else (0 if prior is None else len(prior))
    prior = np.zeros(d) if prior is None else np.asarray(prior, dtype=np.float64)
    w = prior.copy()
    if len(ds) == 0:
        return w
    lam = max(lam, 1e-6)
    f, g = nll_and_grad(w, ds, prior, lam)
    for _ in range(iters):
        H = _hessian(w, ds, lam)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = g
        t = 1.0
        while t > 1e-4:
            w_new = w - t * step
            f_new, g_new = nll_and_grad(w_new, ds, prior, lam)
            if f_new <= f - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        else:
            break
        converged = f - f_new < tol
        w, f, g = w_new, f_new, g_new
        if converged:
            break
    return w


def standard_errors(w: np.ndarray, ds: Dataset, lam: float) -> np.ndarray:
    """Approximate standard errors of the weights (inverse observed information)."""
    if len(ds) == 0:
        return np.full(len(w), np.inf)
    H = _hessian(w, ds, max(lam, 1e-6)) * len(ds)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def evaluate(ds: Dataset, w: np.ndarray | None, scores: np.ndarray | None = None) -> dict:
    """Top-1/top-3 accuracy and mean log-loss on a dataset (``scores`` overrides ``X @ w``)."""
    if len(ds) == 0:
        return {"positions": 0, "top1": 0.0, "top3": 0.0, "logloss": 0.0}
    z = ds.X @ w if scores is None else np.asarray(scores, dtype=np.float64)
    lse, _ = _seg_logsumexp(z, ds.starts)
    ends = np.append(ds.starts[1:], len(z))
    top1 = top3 = 0
    for s, e, c in zip(ds.starts, ends, ds.chosen):
        seg = z[s:e]
        greater = int((seg > z[c]).sum())
        ties = int((seg == z[c]).sum()) - 1  # ties are broken at random
        top1 += (greater == 0) / (1 + ties)
        top3 += min(1.0, max(0.0, (3 - greater) / (1 + ties)))
    n = len(ds)
    return {
        "positions": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "logloss": float((lse - z[ds.chosen]).mean()),
    }
