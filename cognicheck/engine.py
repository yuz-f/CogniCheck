"""Optional Stockfish integration.

The engine is *not* used to predict the move directly. It provides one
signal among many ("how much does this move lose objectively?") whose weight
is learned per player: a strong player gets a large weight, a beginner a
small one. Everything works without an engine; predictions are just less
sharp.
"""
from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass

import chess
import chess.engine

MATE_CP = 3000


def find_stockfish() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.exists(env):
        return env
    for cand in ("stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"):
        p = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if p:
            return p
    return None


@dataclass
class MoveEval:
    cp: int  # from the mover's point of view
    reply: chess.Move | None  # engine's best answer for the opponent


class Engine:
    def __init__(self, path: str | None = None, depth: int = 6, threads: int = 1):
        self.path = path or find_stockfish()
        if not self.path:
            raise RuntimeError("Stockfish not found (set STOCKFISH_PATH)")
        self.depth = depth
        self._lock = threading.Lock()
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self._engine.configure({"Threads": threads, "Hash": 32})
        self._cache: dict[tuple[str, int], dict[str, MoveEval]] = {}

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def evaluate_all(self, board: chess.Board, depth: int | None = None) -> dict[str, MoveEval]:
        """Evaluate every legal move. Keys are UCI strings."""
        depth = depth or self.depth
        key = (board.fen(), depth)
        if key in self._cache:
            return self._cache[key]
        n = board.legal_moves.count()
        out: dict[str, MoveEval] = {}
        if n == 0:
            return out
        with self._lock:
            infos = self._engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            score = info["score"].pov(board.turn).score(mate_score=MATE_CP)
            out[pv[0].uci()] = MoveEval(int(score), pv[1] if len(pv) > 1 else None)
        # Very rarely multipv misses a move; give it the worst score seen.
        if len(out) < n and out:
            worst = min(e.cp for e in out.values())
            for m in board.legal_moves:
                out.setdefault(m.uci(), MoveEval(worst, None))
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[key] = out
        return out

    def top_moves(self, board: chess.Board, k: int, nodes: int = 15000) -> list[tuple[chess.Move, int, chess.Move | None]]:
        n = min(k, board.legal_moves.count())
        if n == 0:
            return []
        with self._lock:
            infos = self._engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=n)
        if isinstance(infos, dict):
            infos = [infos]
        out = []
        for info in infos:
            pv = info.get("pv") or []
            if pv:
                out.append((pv[0], int(info["score"].pov(board.turn).score(mate_score=MATE_CP)), pv[1] if len(pv) > 1 else None))
        return out

    def best_score(self, board: chess.Board, root_moves: list[chess.Move] | None = None, nodes: int = 15000) -> int:
        """Score for the side to move, optionally restricted to ``root_moves``."""
        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_CP
            return 0
        with self._lock:
            info = self._engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=root_moves or None)
        return int(info["score"].pov(board.turn).score(mate_score=MATE_CP))


_shared: Engine | None = None
_shared_lock = threading.Lock()


def shared_engine() -> Engine | None:
    """Process-wide engine, or None when Stockfish is unavailable/disabled."""
    global _shared
    if os.environ.get("COGNICHECK_NO_ENGINE"):
        return None
    with _shared_lock:
        if _shared is None:
            try:
                _shared = Engine()
            except Exception:
                return None
        return _shared
