"""Maia-2 human-move prior (optional).

Maia-2 (Tang et al., 2024) is a network trained on millions of Lichess games
to predict the move a human of a given rating would play. We do not predict
with it directly: its probability for each legal move becomes one feature of
the interpretable personal model ("what a typical player of this strength
would do here"), and the personal weights say how far this player deviates.

Weights (~100 MB) are downloaded on first use by the ``maia2`` package into
``data/maia2_models``. Everything degrades gracefully when torch/maia2 are
not installed: the feature is simply zero.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "maia2_models"
LOGP_FLOOR = -6.0  # log(1/400): clip very small probabilities


def available() -> bool:
    if os.environ.get("COGNICHECK_NO_NEURAL"):
        return False
    try:
        import maia2  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


class Neural:
    """Thread-safe Maia-2 wrapper with a position cache."""

    def __init__(self, kind: str = "blitz"):
        import torch
        from maia2 import inference, model

        torch.set_num_threads(1)  # we parallelise over players, not inside the net
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model = model.from_pretrained(type=kind, device="cpu", save_root=str(MODEL_DIR))
        self._prepared = inference.prepare()
        self._inference = inference.inference_each
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, int, int], dict[str, float]] = {}

    def move_probs(self, board: chess.Board, elo_self: int, elo_oppo: int) -> dict[str, float]:
        """P(move | position, ratings) for every legal move, keyed by UCI."""
        fen = board.fen()
        key = (fen, int(elo_self), int(elo_oppo))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:
            probs, _win = self._inference(self._model, self._prepared, fen, int(elo_self), int(elo_oppo))
        # maia2 writes castling as king-to-rook-square for Chess960 style; normalise to standard UCI.
        out: dict[str, float] = {}
        for uci, p in probs.items():
            try:
                mv = board.parse_uci(uci)
                out[mv.uci()] = float(p)
            except ValueError:
                out[uci] = float(p)
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = out
        return out

    def features(self, board: chess.Board, moves: list[chess.Move], elo_self: int, elo_oppo: int) -> tuple[np.ndarray, np.ndarray]:
        """(log-probability per move clipped at LOGP_FLOOR, indicator of Maia's top move)."""
        probs = self.move_probs(board, elo_self, elo_oppo)
        p = np.array([probs.get(m.uci(), 0.0) for m in moves], dtype=np.float32)
        logp = np.log(np.clip(p, np.exp(LOGP_FLOOR), 1.0))
        top = np.zeros(len(moves), dtype=np.float32)
        if len(p):
            top[int(np.argmax(p))] = 1.0
        return logp, top


_shared: Neural | None = None
_shared_lock = threading.Lock()


def shared_neural() -> Neural | None:
    """Process-wide Maia-2 instance, or None when unavailable."""
    global _shared
    if not available():
        return None
    with _shared_lock:
        if _shared is None:
            try:
                _shared = Neural()
            except Exception:
                return None
        return _shared
