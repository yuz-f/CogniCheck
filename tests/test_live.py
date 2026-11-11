import json
import os
import time

import chess
import pytest

os.environ["COGNICHECK_NO_ENGINE"] = "1"

from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.live import LiveSession, parse_lichess_stream, resolve_lichess_game, simulated_events  # noqa: E402
from cognicheck.predictor import Predictor  # noqa: E402
from cognicheck.profile import build_profile  # noqa: E402

from .test_core import SAMPLE_PGN  # noqa: E402


def test_resolve_game_ids_without_network():
    assert resolve_lichess_game("https://lichess.org/abcd1234") == "abcd1234"
    assert resolve_lichess_game("https://lichess.org/abcd1234wxyz") == "abcd1234"
    assert resolve_lichess_game("lichess.org/q7ZvsdUF/black") == "q7ZvsdUF"
    assert resolve_lichess_game("q7Zv5dUF") == "q7Zv5dUF"


def test_parse_lichess_stream():
    lines = [
        json.dumps({"id": "abcd1234", "variant": {"key": "standard"}, "speed": "blitz", "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "turns": 1, "status": {"id": 20, "name": "started"}, "players": {"white": {"user": {"name": "Alice"}}, "black": {"user": {"name": "Bob"}}}, "lastMove": "e2e4"}),
        json.dumps({"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR", "lm": "e7e5", "wc": 178, "bc": 179}),
        json.dumps({"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R", "lm": "g1f3", "wc": 175, "bc": 179}),
        json.dumps({"id": "abcd1234", "status": {"id": 31, "name": "resign"}, "winner": "white"}),
    ]
    evs = list(parse_lichess_stream(iter(lines)))
    assert [e["type"] for e in evs] == ["start", "move", "move", "end"]
    assert evs[0]["white"] == "Alice" and evs[0]["black"] == "Bob"
    assert evs[1]["uci"] == "e7e5" and evs[2]["clock"] == {"white": 175, "black": 179}
    assert evs[3]["result"] == "1-0"


def test_parse_castling_as_king_takes_rook():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    start = {"id": "x", "variant": {"key": "standard"}, "fen": b.fen(), "status": "started", "players": {}}
    evs = list(parse_lichess_stream(iter([json.dumps(start), json.dumps({"lm": "e1h1"})])))
    assert evs[1]["uci"] == "e1g1"


@pytest.fixture(scope="module")
def alice():
    return build_profile("Alice", games_of(parse_pgn(SAMPLE_PGN), "Alice"))


def test_live_session_predicts_before_each_move(alice):
    game = parse_pgn(SAMPLE_PGN)[0]  # Alice is White
    src = simulated_events(str(game), delay=0, sleep=lambda s: None)
    sess = LiveSession("t", src, lambda w, b: (Predictor(alice), chess.WHITE) if w == "Alice" else (None, None))
    sess.run()
    snap = sess.snapshot()
    assert snap["status"] == "finished" and snap["result"] == "1-0"
    n_white_moves = (len(list(game.mainline_moves())) + 1) // 2
    assert snap["score"]["predicted"] == n_white_moves
    for p in sess.predictions:
        assert p.actual is not None
        assert chess.Board(p.fen).turn == chess.WHITE


def test_live_guess_is_registered_before_move(alice):
    game = parse_pgn(SAMPLE_PGN)[0]
    gate = {"go": False}

    def slow_sleep(_):
        while not gate["go"]:
            time.sleep(0.01)

    src = simulated_events(str(game), delay=1, sleep=slow_sleep)
    sess = LiveSession("t2", src, lambda w, b: (Predictor(alice), chess.WHITE)).start()
    for _ in range(300):
        if sess.pending is not None:
            break
        time.sleep(0.01)
    assert sess.pending is not None
    sess.guess("e2e4")
    with pytest.raises(ValueError):
        sess.guess("e2e5")
    gate["go"] = True
    for _ in range(500):
        if sess.status == "finished":
            break
        time.sleep(0.01)
    first = sess.predictions[0]
    assert first.user_hit and first.guess_at <= time.time()


def test_unknown_player_reports_error():
    game = parse_pgn(SAMPLE_PGN)[0]
    sess = LiveSession("t3", simulated_events(str(game), delay=0, sleep=lambda s: None), lambda w, b: (None, None))
    sess.run()
    assert sess.status == "error" and "No profile" in sess.error
