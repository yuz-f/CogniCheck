"""FastAPI app: REST API + the single-page game in ``web/``."""
from __future__ import annotations

import io
import secrets
import threading
import traceback
from pathlib import Path

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cli import PROFILES, ROOT, build_from_pgn_text
from .engine import shared_engine
from .game import SessionStore
from .live import LiveSession, lichess_events, simulated_events
from .predictor import Predictor
from .profile import Profile

WEB = ROOT / "web"

app = FastAPI(title="CogniCheck", version="0.1.0")
sessions = SessionStore()
_profiles: dict[str, Profile] = {}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def profiles() -> dict[str, Profile]:
    with _lock:
        if PROFILES.exists():
            for p in PROFILES.glob("*.json"):
                pid = p.stem
                mtime = p.stat().st_mtime
                cur = _profiles.get(pid)
                if cur is None or getattr(cur, "_mtime", 0) < mtime:
                    prof = Profile.load(p)
                    prof._mtime = mtime  # type: ignore[attr-defined]
                    _profiles[pid] = prof
        return dict(_profiles)


def get_profile(pid: str) -> Profile:
    prof = profiles().get(pid)
    if prof is None:
        raise HTTPException(404, f"unknown player {pid}")
    return prof


def predictor_for(pid: str) -> Predictor:
    return Predictor(get_profile(pid), shared_engine())


# ------------------------------------------------------------------ models
class BuildRequest(BaseModel):
    pgn: str | None = None
    player: str | None = None
    lichess: str | None = None
    chesscom: str | None = None
    max_games: int = 150


class PredictRequest(BaseModel):
    player_id: str
    fen: str | None = None
    moves: list[str] = []
    top_k: int = 5


class GameRequest(BaseModel):
    player_id: str
    game_index: int | None = None
    pgn: str | None = None


class GuessRequest(BaseModel):
    uci: str | None = None


class LiveRequest(BaseModel):
    source: str | None = None  # Lichess game URL / id / username
    player_id: str | None = None  # whose profile to use (default: match by name)
    color: str | None = None  # "white"/"black" if the profile name is not in the game
    simulate_game: int | None = None  # replay a held-out game of player_id instead
    delay: float = 4.0


class PositionRequest(BaseModel):
    fen: str | None = None
    moves: list[str] = []


# --------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "engine": shared_engine() is not None, "players": len(profiles())}


@app.get("/api/players")
def list_players():
    out = []
    for p in profiles().values():
        ev = p.evaluation or {}
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "games": p.stats.get("games"),
                "decisions": p.stats.get("decisions"),
                "heldout_games": len(p.heldout),
                "top1": (ev.get("personal_with_book") or {}).get("top1"),
                "engine": p.engine,
            }
        )
    return sorted(out, key=lambda r: r["name"].lower())


@app.get("/api/players/{pid}")
def player(pid: str):
    return get_profile(pid).summary()


@app.get("/api/players/{pid}/games")
def player_games(pid: str):
    prof = get_profile(pid)
    out = []
    for i, pgn in enumerate(prof.heldout):
        g = chess.pgn.read_game(io.StringIO(pgn))
        if g is None:
            continue
        h = g.headers
        out.append({"index": i, "white": h.get("White"), "black": h.get("Black"), "result": h.get("Result"), "date": h.get("Date"), "plies": sum(1 for _ in g.mainline_moves())})
    return out


def _run_build(job_id: str, req: BuildRequest):
    job = _jobs[job_id]
    try:
        from .ingest import fetch_chesscom, fetch_lichess

        if req.lichess:
            job["status"] = "downloading from Lichess"
            text, player = fetch_lichess(req.lichess, req.max_games), req.lichess
        elif req.chesscom:
            job["status"] = "downloading from Chess.com"
            text, player = fetch_chesscom(req.chesscom, req.max_games), req.chesscom
        else:
            text, player = req.pgn or "", req.player

        def progress(i, n):
            job["status"] = f"analysing game {i}/{n}"
            job["progress"] = i / n

        prof = build_from_pgn_text(text, player, shared_engine(), req.max_games, progress)
        job.update(status="done", progress=1.0, player_id=prof.id)
    except Exception as e:  # report to the client
        traceback.print_exc()
        job.update(status="error", error=str(e))


@app.post("/api/players")
def build_player(req: BuildRequest):
    if not (req.pgn or req.lichess or req.chesscom):
        raise HTTPException(400, "provide pgn, lichess or chesscom")
    job_id = secrets.token_urlsafe(6)
    _jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0}
    threading.Thread(target=_run_build, args=(job_id, req), daemon=True).start()
    return _jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.post("/api/predict")
def predict(req: PredictRequest):
    pred = predictor_for(req.player_id)
    try:
        board = chess.Board(req.fen) if req.fen else chess.Board()
        for u in req.moves:
            board.push_uci(u)
    except ValueError as e:
        raise HTTPException(400, f"bad position: {e}")
    return pred.predict(board, top_k=req.top_k)


@app.post("/api/position")
def position(req: PositionRequest):
    """Apply moves to a position; returns FEN, SAN history and legal moves."""
    try:
        board = chess.Board(req.fen) if req.fen else chess.Board()
        start = board.copy()
        sans = []
        for u in req.moves:
            m = chess.Move.from_uci(u)
            if m not in board.legal_moves:
                m = chess.Move(m.from_square, m.to_square, promotion=chess.QUEEN)
            sans.append(board.san(m))
            board.push(m)
    except ValueError as e:
        raise HTTPException(400, f"bad position: {e}")
    return {
        "fen": board.fen(),
        "start_fen": start.fen(),
        "history": sans,
        "legal": [m.uci() for m in board.legal_moves],
        "turn": "white" if board.turn else "black",
        "game_over": board.is_game_over(),
        "last_move": board.move_stack[-1].uci() if board.move_stack else None,
    }


@app.post("/api/game")
def new_game(req: GameRequest):
    prof = get_profile(req.player_id)
    if req.pgn:
        pgn = req.pgn
    else:
        if not prof.heldout:
            raise HTTPException(400, "this player has no held-out games")
        idx = req.game_index if req.game_index is not None else secrets.randbelow(len(prof.heldout))
        if not 0 <= idx < len(prof.heldout):
            raise HTTPException(400, "bad game index")
        pgn = prof.heldout[idx]
    try:
        sess = sessions.create(Predictor(prof, shared_engine()), pgn)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return sess.state()


def _session(sid: str):
    try:
        return sessions.get(sid)
    except KeyError:
        raise HTTPException(404, "unknown game session")


@app.get("/api/game/{sid}")
def game_state(sid: str):
    return _session(sid).state()


@app.post("/api/game/{sid}/guess")
def game_guess(sid: str, req: GuessRequest):
    s = _session(sid)
    with s.lock:
        try:
            return s.guess(req.uci)
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.post("/api/game/{sid}/hint")
def game_hint(sid: str):
    s = _session(sid)
    with s.lock:
        s.user_score -= 2
        return {"hint": s.hint(), "state": s.state()}


# -------------------------------------------------------------------- live
_live: dict[str, LiveSession] = {}


def _live_matcher(player_id: str | None, color: str | None):
    def match(white: str, black: str):
        profs = profiles()
        if player_id:
            prof = profs.get(player_id)
            if prof is None:
                return None, None
            names = {white.lower(): chess.WHITE, black.lower(): chess.BLACK}
            if prof.name.lower() in names:
                c = names[prof.name.lower()]
            elif color in ("white", "black"):
                c = chess.WHITE if color == "white" else chess.BLACK
            else:
                return None, None
            return Predictor(prof, shared_engine()), c
        for c, name in ((chess.WHITE, white), (chess.BLACK, black)):
            for prof in profs.values():
                if prof.name.lower() == name.lower():
                    return Predictor(prof, shared_engine()), c
        return None, None

    return match


@app.post("/api/live")
def live_start(req: LiveRequest):
    if req.simulate_game is not None:
        prof = get_profile(req.player_id or "")
        if not 0 <= req.simulate_game < len(prof.heldout):
            raise HTTPException(400, "bad game index")
        src = simulated_events(prof.heldout[req.simulate_game], delay=max(0.5, req.delay))
    elif req.source:
        src = lichess_events(req.source)
    else:
        raise HTTPException(400, "give a Lichess game/username or simulate_game")
    sid = secrets.token_urlsafe(8)
    sess = LiveSession(sid, src, _live_matcher(req.player_id, req.color))
    if len(_live) > 100:
        _live.pop(min(_live, key=lambda k: _live[k].created))
    _live[sid] = sess.start()
    return {"id": sid}


def _live_session(sid: str) -> LiveSession:
    if sid not in _live:
        raise HTTPException(404, "unknown live session")
    return _live[sid]


@app.get("/api/live/{sid}")
def live_state(sid: str):
    return _live_session(sid).snapshot()


@app.get("/api/live/{sid}/log")
def live_log(sid: str):
    return _live_session(sid).log()


@app.post("/api/live/{sid}/guess")
def live_guess(sid: str, req: GuessRequest):
    try:
        return _live_session(sid).guess(req.uci or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/live/{sid}/events")
def live_events(sid: str):
    """Server-Sent Events: one JSON snapshot per update."""
    import json
    import queue as _q

    sess = _live_session(sid)
    q = sess.subscribe()

    def gen():
        try:
            while True:
                try:
                    snap = q.get(timeout=15)
                except _q.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(snap)}\n\n"
                if snap["status"] in ("finished", "error"):
                    return
        finally:
            sess.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
