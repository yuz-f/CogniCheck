"""Follow a game while it is being played and predict each move *before* it happens.

Sources produce a common event format:

    {"type": "start", "fen": <full FEN>, "white": name, "black": name, "last_move": uci|None,
     "clock": {"white": s, "black": s}|None, "game_id": str}
    {"type": "move", "uci": "e2e4", "clock": {...}|None}
    {"type": "end", "result": "1-0"|"0-1"|"1/2-1/2"|"*", "status": str}

* ``lichess_events``   - Lichess public game stream (no login needed).
* ``simulated_events`` - replays a PGN with a delay; same format, for demos/tests.

A ``LiveSession`` consumes events in a background thread. Whenever the
profiled player is to move it computes the prediction immediately and
timestamps it, so every prediction is registered before the real move
arrives. Browsers subscribe to a queue of JSON updates (served as SSE).
"""
from __future__ import annotations

import io
import json
import queue
import re
import secrets
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

import chess
import chess.pgn

from .ingest import USER_AGENT
from .predictor import Predictor

LICHESS = "https://lichess.org"
GAME_ID_RE = re.compile(r"^(?:https?://)?(?:www\.)?lichess\.org/([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?:/.*)?$")


# ---------------------------------------------------------------- sources
def _open(url: str, accept: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return urllib.request.urlopen(req, timeout=timeout)


def resolve_lichess_game(ref: str) -> str:
    """Game URL, 8-char game id, or username (-> their current game)."""
    ref = ref.strip()
    m = GAME_ID_RE.match(ref)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{8}", ref) and not ref.isalpha():
        return ref
    user = urllib.parse.quote(ref.rsplit("/", 1)[-1])
    with _open(f"{LICHESS}/api/user/{user}/current-game?moves=false&clocks=false", "application/json") as r:
        data = json.loads(r.read().decode())
    if not data.get("id"):
        raise ValueError(f"{ref} is not playing right now")
    return data["id"]


def _name(players: dict, side: str) -> str:
    p = (players or {}).get(side) or {}
    user = p.get("user") or {}
    return user.get("name") or user.get("id") or p.get("name") or ("AI" if p.get("aiLevel") else "?")


def parse_lichess_stream(lines: Iterator[str], game_id: str = "") -> Iterator[dict]:
    """Translate Lichess ``/api/stream/game/{id}`` NDJSON into our events.

    The first line describes the game (full FEN); later lines carry the board
    part of the FEN plus ``lm`` (last move) and clocks ``wc``/``bc``.
    """
    board: chess.Board | None = None
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        if board is None:
            if msg.get("variant", {}).get("key", "standard") not in ("standard", "fromPosition"):
                raise ValueError("only standard chess is supported")
            fen = msg.get("fen") or chess.STARTING_FEN
            board = chess.Board(fen) if len(fen.split()) >= 4 else _board_from_placement(fen, msg)
            yield {
                "type": "start",
                "game_id": msg.get("id", game_id),
                "fen": board.fen(),
                "white": _name(msg.get("players"), "white"),
                "black": _name(msg.get("players"), "black"),
                "last_move": msg.get("lastMove"),
                "clock": _clock(msg),
                "speed": msg.get("speed"),
            }
            status = (msg.get("status") or {}).get("name") if isinstance(msg.get("status"), dict) else msg.get("status")
            if status and status not in ("started", "created"):
                yield {"type": "end", "result": _result(msg), "status": status}
                return
            continue
        if "lm" in msg:
            mv = _parse_move(board, msg["lm"])
            if mv is None:  # out of sync: trust the server's position
                board = _board_from_placement(msg.get("fen", ""), {"turns": None}, fallback=board)
                yield {"type": "resync", "fen": board.fen(), "clock": _clock(msg)}
                continue
            board.push(mv)
            yield {"type": "move", "uci": mv.uci(), "clock": _clock(msg)}
        elif "status" in msg or "winner" in msg:
            st = msg.get("status")
            yield {"type": "end", "result": _result(msg), "status": st.get("name") if isinstance(st, dict) else st}
            return
    yield {"type": "end", "result": "*", "status": "stream closed"}


def _parse_move(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        return board.parse_uci(uci)  # normalises king-takes-rook castling to e1g1
    except ValueError:
        pass
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return None
    # Lichess may encode castling as king-takes-rook (e1h1).
    for cand in board.legal_moves:
        if board.is_castling(cand) and cand.from_square == mv.from_square and board.piece_at(mv.to_square) and board.piece_at(mv.to_square).piece_type == chess.ROOK:
            if (chess.square_file(mv.to_square) > chess.square_file(mv.from_square)) == (chess.square_file(cand.to_square) > chess.square_file(cand.from_square)):
                return cand
    return None


def _board_from_placement(placement: str, msg: dict, fallback: chess.Board | None = None) -> chess.Board:
    b = chess.Board()
    try:
        b.set_board_fen(placement.split()[0])
    except (ValueError, IndexError):
        return fallback or b
    turns = msg.get("turns")
    if turns is not None:
        b.turn = chess.WHITE if turns % 2 == 0 else chess.BLACK
    elif fallback is not None:
        b.turn = not fallback.turn
    b.castling_rights = b.clean_castling_rights() if fallback is None else fallback.castling_rights & b.clean_castling_rights()
    return b


def _clock(msg: dict) -> dict | None:
    if "wc" in msg or "bc" in msg:
        return {"white": msg.get("wc"), "black": msg.get("bc")}
    c = msg.get("clock")
    if isinstance(c, dict) and "white" in c:
        return {"white": c.get("white"), "black": c.get("black")}
    return None


def _result(msg: dict) -> str:
    w = msg.get("winner")
    if w == "white":
        return "1-0"
    if w == "black":
        return "0-1"
    st = msg.get("status")
    st = st.get("name") if isinstance(st, dict) else st
    return "1/2-1/2" if st in ("draw", "stalemate") else "*"


def lichess_events(ref: str) -> Iterator[dict]:
    game_id = resolve_lichess_game(ref)
    resp = _open(f"{LICHESS}/api/stream/game/{game_id}", "application/x-ndjson", timeout=600)
    lines = (ln.decode("utf-8", "replace") for ln in resp)
    yield from parse_lichess_stream(lines, game_id)


def simulated_events(pgn: str, delay: float = 3.0, sleep: Callable[[float], None] = time.sleep) -> Iterator[dict]:
    """Replay a finished game as if it were live (same event format)."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("could not parse game")
    h = game.headers
    yield {"type": "start", "game_id": "sim-" + secrets.token_hex(3), "fen": game.board().fen(), "white": h.get("White", "?"), "black": h.get("Black", "?"), "last_move": None, "clock": None, "speed": "simulated"}
    for mv in game.mainline_moves():
        sleep(delay)
        yield {"type": "move", "uci": mv.uci(), "clock": None}
    yield {"type": "end", "result": h.get("Result", "*"), "status": "finished"}


# ---------------------------------------------------------------- session
@dataclass
class LivePrediction:
    ply: int
    fen: str
    made_at: float
    prediction: dict
    guess: str | None = None
    guess_at: float | None = None
    actual: str | None = None
    actual_san: str | None = None
    p_actual: float | None = None
    model_hit: bool | None = None
    user_hit: bool | None = None
    updates: int = 0  # re-predictions made while the player was thinking


@dataclass
class LiveSession:
    id: str
    source: Iterator[dict]
    predictor_for: Callable[[str, str], tuple[Predictor | None, chess.Color | None]]
    status: str = "connecting"
    board: chess.Board = field(default_factory=chess.Board)
    white: str = "?"
    black: str = "?"
    player: str | None = None
    color: chess.Color | None = None
    clock: dict | None = None
    history: list[str] = field(default_factory=list)
    predictions: list[LivePrediction] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    _subs: list[queue.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _predictor: Predictor | None = None

    # ------------------------------------------------------------ pub/sub
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._subs.append(q)
            q.put(self.snapshot())
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self) -> None:
        snap = self.snapshot()
        for q in list(self._subs):
            try:
                q.put_nowait(snap)
            except queue.Full:
                pass

    # -------------------------------------------------------------- state
    @property
    def pending(self) -> LivePrediction | None:
        if self.predictions and self.predictions[-1].actual is None:
            return self.predictions[-1]
        return None

    def snapshot(self) -> dict:
        done = [p for p in self.predictions if p.actual is not None]
        pend = self.pending
        last = done[-1] if done else None
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "white": self.white,
            "black": self.black,
            "player": self.player,
            "player_color": None if self.color is None else ("white" if self.color == chess.WHITE else "black"),
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "legal": [m.uci() for m in self.board.legal_moves] if pend else [],
            "last_move": self.board.move_stack[-1].uci() if self.board.move_stack else None,
            "history": self.history,
            "clock": self.clock,
            "result": self.result,
            "pending": None if pend is None else {"ply": pend.ply, "prediction": pend.prediction, "guess": pend.guess, "made_at": pend.made_at, "updates": pend.updates, "thinking_for": round(time.time() - pend.made_at, 1)},
            "last": None if last is None else {k: getattr(last, k) for k in ("ply", "actual", "actual_san", "guess", "p_actual", "model_hit", "user_hit")} | {"model_top": last.prediction["moves"][0]["san"] if last.prediction["moves"] else None, "perspective": last.prediction.get("perspective")},
            "score": {
                "predicted": len(done),
                "model_hits": sum(bool(p.model_hit) for p in done),
                "model_top3": sum(p.actual in [m["uci"] for m in p.prediction["moves"][:3]] for p in done),
                "user_hits": sum(bool(p.user_hit) for p in done),
                "user_guesses": sum(p.guess is not None for p in done),
            },
        }

    def guess(self, uci: str) -> dict:
        with self._lock:
            pend = self.pending
            if pend is None:
                raise ValueError("not this player's turn")
            mv = chess.Move.from_uci(uci)
            if mv not in self.board.legal_moves:
                mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
                if mv not in self.board.legal_moves:
                    raise ValueError("illegal move")
            pend.guess, pend.guess_at = mv.uci(), time.time()
        self._publish()
        return self.snapshot()

    def log(self) -> list[dict]:
        return [p.__dict__ | {"prediction": {"moves": p.prediction["moves"][:3]}} for p in self.predictions]

    # --------------------------------------------------------------- loop
    def _time_left(self) -> float | None:
        if not self.clock:
            return None
        v = self.clock.get("white" if self.color == chess.WHITE else "black")
        return float(v) if v is not None else None

    def _maybe_predict(self) -> None:
        if self._predictor is None or self.board.is_game_over() or self.board.turn != self.color:
            return
        pred = self._predictor.predict(self.board.copy(), think_time=0.0, time_left=self._time_left())
        lp = LivePrediction(ply=len(self.board.move_stack), fen=self.board.fen(), made_at=time.time(), prediction=pred)
        self.predictions.append(lp)
        # Re-predict once the player has clearly been thinking for a while: habits
        # weigh less and calculation more. Each update is timestamped too.
        threading.Thread(target=self._rethink, args=(lp,), daemon=True).start()

    def _rethink(self, lp: "LivePrediction") -> None:
        from .features import LONG_SECONDS, QUICK_SECONDS

        for wait in (QUICK_SECONDS + 0.5, LONG_SECONDS + 0.5):
            while time.time() - lp.made_at < wait:
                time.sleep(0.25)
                if lp.actual is not None or self.status != "live":
                    return
            with self._lock:
                if lp.actual is not None or self.pending is not lp:
                    return
                board = self.board.copy()
            pred = self._predictor.predict(board, think_time=time.time() - lp.made_at, time_left=self._time_left())
            with self._lock:
                if lp.actual is None:
                    lp.prediction = pred
                    lp.updates = getattr(lp, "updates", 0) + 1
            self._publish()

    def _apply_move(self, uci: str, clock: dict | None) -> None:
        mv = chess.Move.from_uci(uci)
        with self._lock:
            pend = self.pending
            if pend is not None and self.board.turn == self.color:
                pend.actual = mv.uci()
                pend.actual_san = self.board.san(mv)
                pend.p_actual = pend.prediction["all"].get(mv.uci(), 0.0)
                pend.model_hit = bool(pend.prediction["moves"]) and pend.prediction["moves"][0]["uci"] == mv.uci()
                pend.user_hit = pend.guess == mv.uci() if pend.guess else False
            self.history.append(self.board.san(mv))
            self.board.push(mv)
            self.clock = clock or self.clock

    def run(self) -> None:
        try:
            for ev in self.source:
                t = ev["type"]
                if t == "start":
                    with self._lock:
                        self.board = chess.Board(ev["fen"])
                        self.white, self.black, self.clock = ev["white"], ev["black"], ev.get("clock")
                        self.status = "matching player"
                    self._publish()
                    self._predictor, self.color = self.predictor_for(self.white, self.black)
                    if self._predictor is None:
                        self.status = "error"
                        self.error = f"No profile for {self.white} or {self.black}. Build one first (Players tab)."
                        self._publish()
                        return
                    self.player = self._predictor.profile.name
                    self.status = "live"
                    self._maybe_predict()
                elif t == "move":
                    self._apply_move(ev["uci"], ev.get("clock"))
                    self._maybe_predict()
                elif t == "resync":
                    with self._lock:
                        self.board = chess.Board(ev["fen"])
                        self.predictions = [p for p in self.predictions if p.actual is not None]
                    self._maybe_predict()
                elif t == "end":
                    with self._lock:
                        self.result = ev.get("result")
                        self.status = "finished"
                        if self.pending is not None:
                            self.predictions.pop()
                self._publish()
        except Exception as e:  # network errors, bad streams ...
            self.status, self.error = "error", str(e)
            self._publish()
        finally:
            if self.status not in ("finished", "error"):
                self.status = "finished"
                self._publish()

    def start(self) -> "LiveSession":
        threading.Thread(target=self.run, daemon=True).start()
        return self
