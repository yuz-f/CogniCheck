"""Working-memory ("capacity") statistics of a player.

Three counted measures that complement the model weights:

* **tactic complexity** – how often the player walks into a tactic, split by how many
  pieces one must hold in mind to see it (simple: 2, complex: 4+);
* **overload** – blunder rate as a function of how many pieces are in contact;
* **focus span** – how much the player's choice concentrates on the pieces they moved
  most recently, against the share of such moves among the legal ones.

Everything is computed from the feature matrix, so the same code serves the cohort
(stored arrays) and a freshly built profile.
"""
from __future__ import annotations

import numpy as np

from .features import IDX

LOAD_BUCKETS = [("quiet", 0.0, 0.3), ("busy", 0.3, 0.6), ("crowded", 0.6, 1.01)]
MIN_EVENTS = 15


def _rate(num: float, den: float) -> float | None:
    return round(float(num) / float(den), 3) if den >= MIN_EVENTS else None


def capacity_stats(X: np.ndarray, starts: np.ndarray, chosen: np.ndarray, load: np.ndarray | None, tp: np.ndarray | None = None) -> dict:
    """``load`` is one value per position (pieces in contact / MAX_CONTACTS); ``tp`` a
    per-position time-pressure flag. Returns {} when there is too little data."""
    if len(starts) == 0:
        return {}
    ends = np.append(starts[1:], len(X))
    sizes = ends - starts
    out: dict = {}

    # 1. Tactic complexity: P(chosen | candidate allows a simple / complex tactic).
    simple, complex_ = X[:, IDX["allows_simple"]] > 0, X[:, IDX["allows_complex"]] > 0
    chosen_mask = np.zeros(len(X), bool)
    chosen_mask[chosen] = True
    cx = {
        "simple": {"available": int(simple.sum()), "walked_in": int((simple & chosen_mask).sum())},
        "complex": {"available": int(complex_.sum()), "walked_in": int((complex_ & chosen_mask).sum())},
    }
    rs = _rate(cx["simple"]["walked_in"], cx["simple"]["available"])
    rc = _rate(cx["complex"]["walked_in"], cx["complex"]["available"])
    cx["simple"]["rate"], cx["complex"]["rate"] = rs, rc
    # Per-candidate rates are deflated by position size; compare them as a ratio.
    cx["complex_over_simple"] = round(rc / rs, 2) if rs and rc is not None else None
    out["complexity"] = cx

    # 2. Overload: blunder rate (chosen move loses >= 2 pawns) by load bucket.
    if load is not None and len(load) == len(starts) and X[:, IDX["eng_loss"]].any():
        loss = X[chosen, IDX["eng_loss"]]
        buckets = []
        for name, lo, hi in LOAD_BUCKETS:
            m = (load >= lo) & (load < hi)
            row = {"bucket": name, "decisions": int(m.sum()), "share": round(float(m.mean()), 3), "blunder_rate": _rate((loss[m] >= 2).sum(), m.sum()), "avg_loss": round(float(loss[m].mean()), 3) if m.any() else None}
            if tp is not None and len(tp) == len(starts):
                mt = m & tp.astype(bool)
                row["blunder_rate_short_of_time"] = _rate((loss[mt] >= 2).sum(), mt.sum())
            buckets.append(row)
        rates = [b["blunder_rate"] for b in buckets]
        out["overload"] = {
            "buckets": buckets,
            # Ratio of crowded to quiet blunder rate: > 1 = the player degrades when there is more to track.
            "crowded_over_quiet": round(rates[2] / rates[0], 2) if rates[0] and rates[2] is not None else None,
            "mean_load": round(float(load.mean()), 3),
        }

    # 3. Focus span: share of chosen moves on recently moved pieces vs. share of such candidates.
    for key, label in (("focus_recent", "recent"), ("focus_stale", "stale")):
        col = X[:, IDX[key]] > 0
        n_pos = int(np.add.reduceat(col.astype(np.int64), starts).astype(bool).sum())  # positions where it applies
        avail_share = float((np.add.reduceat(col.astype(np.float64), starts) / sizes).mean())
        chosen_share = float(col[chosen].mean())
        out[label] = {
            "positions": n_pos,
            "chosen_share": round(chosen_share, 3),
            "candidate_share": round(avail_share, 3),
            "index": round(chosen_share / avail_share, 2) if avail_share > 0 else None,
        }
    return out


def capacity_summary(stats: dict) -> list[str]:
    """One-line readings for the profile page / CLI."""
    lines = []
    cx = stats.get("complexity") or {}
    r = cx.get("complex_over_simple")
    if r is not None:
        lines.append(f"walks into complex tactics (4+ pieces) {r:.2f}× as readily as simple ones")
    ov = stats.get("overload") or {}
    r = ov.get("crowded_over_quiet")
    if r is not None:
        lines.append(f"blunders {r:.2f}× as often in crowded positions as in quiet ones")
    rc = (stats.get("recent") or {}).get("index")
    if rc is not None:
        lines.append(f"picks one of the 3 most recently moved pieces {rc:.2f}× as often as chance")
    st = (stats.get("stale") or {}).get("index")
    if st is not None:
        lines.append(f"returns to a piece untouched for 8+ moves {st:.2f}× as often as chance")
    return lines
