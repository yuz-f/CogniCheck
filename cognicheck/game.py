"""The prediction game: guess a real player's next move, race the model.

A session replays one of the player's *held-out* games (never seen during
training). At each of the player's moves you guess what they played; the
model makes its own guess; then the real move is revealed together with the
model's reading of the position from the player's perspective.
"""
from __future__ import annotations

import io
import secrets
import threading
import time
from dataclasses import dataclass, field

import chess
import chess.pgn

from .predictor import Predictor

POINTS_EXACT = 10


@dataclass
class Round:
    ply: int
    fen: str
    actual: str
    actual_san: str
    guess: str | None
    guess_san: str | None
    model_top: str
    model_top_san: str
    p_actual: float
    p_guess: float | None
    user_points: int
    model_points: int


@dataclass
class GameSession:
    id: str
    predictor: Predictor
    game: chess.pgn.Game
    color: chess.Color
    moves: list[chess.Move]
    ply: int = 0
    user_score: int = 0
    model_score: int = 0
    rounds: list[Round] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    _pending: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------------------------------------------------------------- state
    def board(self) -> chess.Board:
        b = self.game.board()
        for m in self.moves[: self.ply]:
            b.push(m)
        return b

    def _skip_to_player(self) -> None:
        b = self.board()
        while self.ply < len(self.moves) and b.turn != self.color:
            b.push(self.moves[self.ply])
            self.ply += 1

    @property
    def finished(self) -> bool:
        return self.ply >= len(self.moves)

    def state(self) -> dict:
        b = self.board()
        san_moves = []
        tmp = self.game.board()
        for m in self.moves[: self.ply]:
            san_moves.append(tmp.san(m))
            tmp.push(m)
        h = self.game.headers
        return {
            "id": self.id,
            "player": self.predictor.profile.name,
            "player_color": "white" if self.color == chess.WHITE else "black",
            "white": h.get("White", "?"),
            "black": h.get("Black", "?"),
            "result": h.get("Result", "*") if self.finished else None,
            "fen": b.fen(),
            "last_move": self.moves[self.ply - 1].uci() if self.ply else None,
            "history": san_moves,
            "ply": self.ply,
            "total_plies": len(self.moves),
            "finished": self.finished,
            "user_score": self.user_score,
            "model_score": self.model_score,
            "rounds_played": len(self.rounds),
            "user_hits": sum(r.user_points > 0 for r in self.rounds),
            "model_hits": sum(r.model_points > 0 for r in self.rounds),
            "legal": [m.uci() for m in b.legal_moves] if not self.finished else [],
        }

    # ---------------------------------------------------------------- play
    def prediction(self) -> dict:
        if self._pending is None or self._pending["ply"] != self.ply:
            self._pending = {"ply": self.ply, "pred": self.predictor.predict(self.board())}
        return self._pending["pred"]

    def hint(self) -> dict:
        """The perspective panel without the move list (costs points)."""
        pred = self.prediction()
        p = pred["perspective"]
        return {"notes": p["notes"], "plans": p["plans"], "attention": p["attention"], "threats": p["threats"], "chances": p["chances"]}

    def guess(self, uci: str | None) -> dict:
        if self.finished:
            raise ValueError("game is over")
        b = self.board()
        guess_move = None
        if uci:
            guess_move = chess.Move.from_uci(uci)
            if guess_move not in b.legal_moves:
                # Allow auto-queen style input like e7e8 without promotion piece
                q = chess.Move(guess_move.from_square, guess_move.to_square, promotion=chess.QUEEN)
                if q in b.legal_moves:
                    guess_move = q
                else:
                    raise ValueError(f"illegal move {uci}")
        pred = self.prediction()
        actual = self.moves[self.ply]
        top = pred["moves"][0]["uci"] if pred["moves"] else None
        probs = pred["all"]
        user_pts = POINTS_EXACT if guess_move == actual else 0
        model_pts = POINTS_EXACT if top == actual.uci() else 0
        self.user_score += user_pts
        self.model_score += model_pts
        rnd = Round(
            ply=self.ply,
            fen=b.fen(),
            actual=actual.uci(),
            actual_san=b.san(actual),
            guess=guess_move.uci() if guess_move else None,
            guess_san=b.san(guess_move) if guess_move else None,
            model_top=top,
            model_top_san=pred["moves"][0]["san"] if pred["moves"] else None,
            p_actual=probs.get(actual.uci(), 0.0),
            p_guess=probs.get(guess_move.uci(), 0.0) if guess_move else None,
            user_points=user_pts,
            model_points=model_pts,
        )
        self.rounds.append(rnd)
        # Did the player actually react to the threats the model flagged?
        persp = pred["perspective"]
        b2 = b.copy()
        b2.push(actual)
        reality = []
        for t in persp["threats"]:
            mv = chess.Move.from_uci(t["uci"])
            still = mv in b2.legal_moves
            reality.append({"threat": t["move"], "p_notice": t["p_notice"], "player_reacted": not still})
        for c in persp["chances"]:
            reality.append({"chance": c["move"], "p_find": c["p_find"], "player_found": c["uci"] == actual.uci()})

        self.ply += 1
        self._skip_to_player()
        return {"round": rnd.__dict__, "prediction": pred, "reality": reality, "state": self.state()}


class SessionStore:
    def __init__(self, max_sessions: int = 500):
        self._s: dict[str, GameSession] = {}
        self._lock = threading.Lock()
        self.max = max_sessions

    def create(self, predictor: Predictor, pgn: str) -> GameSession:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            raise ValueError("could not parse game")
        name = predictor.profile.name.lower()
        if game.headers.get("White", "").lower() == name:
            color = chess.WHITE
        elif game.headers.get("Black", "").lower() == name:
            color = chess.BLACK
        else:
            raise ValueError("player does not appear in this game")
        sess = GameSession(secrets.token_urlsafe(8), predictor, game, color, list(game.mainline_moves()))
        sess._skip_to_player()
        with self._lock:
            if len(self._s) >= self.max:
                oldest = min(self._s.values(), key=lambda s: s.created)
                self._s.pop(oldest.id, None)
            self._s[sess.id] = sess
        return sess

    def get(self, sid: str) -> GameSession:
        with self._lock:
            if sid not in self._s:
                raise KeyError(sid)
            return self._s[sid]
