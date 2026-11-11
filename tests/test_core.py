import io

import chess
import chess.pgn
import numpy as np
import pytest

from cognicheck import model, tactics
from cognicheck.features import FEATURE_NAMES, IDX, addresses_threat, analyse_position
from cognicheck.ingest import games_of, parse_pgn
from cognicheck.predictor import Predictor
from cognicheck.profile import Profile, build_profile, position_key

SAMPLE_PGN = """
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. Nxf7 Kxf7 7. Qf3+ Ke6 8. Nc3 Nb4 9. O-O c6 10. d4 1-0

[White "Bob"]
[Black "Alice"]
[Result "0-1"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 b6 7. Bg5 Bb7 8. f3 h6 9. Bh4 d5 10. e3 0-1

[White "Alice"]
[Black "Carol"]
[Result "1/2-1/2"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O 1/2-1/2

[White "Dave"]
[Black "Alice"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 1-0
"""


def test_see_simple_exchanges():
    # Pawn takes undefended knight: +3
    b = chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1")
    assert tactics.see(b, chess.Move.from_uci("e4d5")) == 3
    # Queen takes pawn defended by pawn: 1 - 9 = -8
    b = chess.Board("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1")
    assert tactics.see(b, chess.Move.from_uci("d2d5")) == -8


def test_threats_and_mates():
    # Black knight on f6 attacks the undefended pawn on e4.
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    threats = tactics.threats_against_side_to_move(b)
    assert threats and threats[0].move.uci() == "f6e4"
    # Scholar's mate pattern: Qxf7#
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert [m.uci() for m in tactics.mates_in_one(b)] == ["h5f7"]


def test_motifs_detect_knight_fork_and_backward():
    b = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    tags = tactics.motifs(b, chess.Move.from_uci("b5c7"))
    assert {"knight", "fork", "check"} <= tags
    b = chess.Board("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1")
    tags = tactics.motifs(b, chess.Move.from_uci("d4a1"))
    assert {"bishop", "backward", "long_diagonal"} <= tags


def test_features_shape_and_flags():
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    pa = analyse_position(b)
    assert pa.X.shape == (b.legal_moves.count(), len(FEATURE_NAMES))
    castle = pa.index(chess.Move.from_uci("e1g1"))
    assert pa.X[castle, IDX["castle"]] == 1
    # d3 defends e4, so it addresses the threat; a3 does not.
    assert addresses_threat(pa, pa.index(chess.Move.from_uci("d2d3")))
    assert not addresses_threat(pa, pa.index(chess.Move.from_uci("a2a3")))
    assert pa.X[pa.index(chess.Move.from_uci("a2a3")), IDX["ignores_threat"]] == 1


def test_model_recovers_planted_weights():
    rng = np.random.default_rng(0)
    d = 4
    true_w = np.array([2.0, -1.0, 0.5, 0.0])
    positions = []
    for _ in range(1500):
        n = rng.integers(5, 25)
        X = rng.normal(size=(n, d)).astype(np.float32)
        p = np.exp(X @ true_w)
        p /= p.sum()
        positions.append((X, int(rng.choice(n, p=p))))
    ds = model.Dataset.from_positions(positions)
    w = model.fit(ds, lam=1e-4, iters=600)
    assert np.allclose(w, true_w, atol=0.25)
    assert model.evaluate(ds, w)["top1"] > model.evaluate(ds, np.zeros(d))["top1"]


def test_prior_shrinkage():
    rng = np.random.default_rng(1)
    positions = [(rng.normal(size=(6, 2)), 0) for _ in range(5)]
    ds = model.Dataset.from_positions(positions)
    prior = np.array([1.0, -1.0])
    w = model.fit(ds, prior=prior, lam=100.0)
    assert np.allclose(w, prior, atol=0.1)


def test_ingest_selects_player_and_colour():
    games = parse_pgn(SAMPLE_PGN)
    assert len(games) == 4
    alice = games_of(games, "alice")
    assert [g.color for g in alice] == [chess.WHITE, chess.BLACK, chess.WHITE, chess.BLACK]


@pytest.fixture(scope="module")
def alice_profile():
    games = games_of(parse_pgn(SAMPLE_PGN), "Alice")
    return build_profile("Alice", games, engine=None)


def test_build_profile_roundtrip(alice_profile, tmp_path):
    prof = alice_profile
    assert prof.id == "alice"
    assert prof.stats["games"] == 4
    assert len(prof.heldout) == 1
    assert prof.book
    path = prof.save(tmp_path)
    again = Profile.load(path)
    assert np.allclose(again.weights, prof.weights)
    assert again.book == prof.book


def test_predictor_uses_opening_book(alice_profile):
    pred = Predictor(alice_profile)
    board = chess.Board()
    key = position_key(board)
    if key in alice_profile.book:  # Alice is White in the training split
        out = pred.predict(board)
        assert out["moves"][0]["uci"] == "e2e4"
        assert out["in_book"]
    out = pred.predict(chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"))
    assert abs(sum(out["all"].values()) - 1) < 1e-3
    per = out["perspective"]
    assert per["threats"] and per["threats"][0]["move"] == "Nxe4"
    assert 0 <= per["threats"][0]["p_notice"] <= 1
    assert per["notes"]


def test_game_session_flow(alice_profile):
    from cognicheck.game import SessionStore

    store = SessionStore()
    pgn = io.StringIO(SAMPLE_PGN)
    first = chess.pgn.read_game(pgn)
    s = store.create(Predictor(alice_profile), str(first))
    st = s.state()
    assert st["player_color"] == "white" and st["ply"] == 0
    r = s.guess("e2e4")
    assert r["round"]["user_points"] == 10
    assert r["state"]["ply"] == 2  # opponent reply auto-played
    while not s.finished:
        s.guess(None)
    assert s.state()["finished"]


def test_cohort_helpers():
    from cognicheck import cohort

    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 0.3, (20, 2)) + c for c in ([0, 0], [4, 4], [0, 4])])
    lab, C, _ = cohort._kmeans(Z, 3)
    assert len(set(lab)) == 3
    assert cohort._silhouette(Z, lab) > 0.7
    w = np.zeros(len(FEATURE_NAMES))
    v = cohort.cognition_vector(w, w, {"motif_index": {"knight": 2.0}}, {"engine_match": 0.4})
    assert v["blind:knight"] > 0 and v["blind:bishop"] == 0 and v["skill:engine_match"] == 0.4
    assert not any(k.startswith("style:eng_") for k in v)


def test_pgn_rating_for_added_players():
    from cognicheck import cohort

    pgn = SAMPLE_PGN.replace('[White "Alice"]', '[White "Alice"]\n[WhiteElo "1500"]').replace('[Black "Alice"]', '[Black "Alice"]\n[BlackElo "1600"]')
    assert cohort._pgn_rating(pgn, "alice") in (1500, 1550, 1600)
    assert cohort._pgn_rating(SAMPLE_PGN, "alice") == 0


def test_habit_feature_leave_one_out():
    from cognicheck.features import build_habits, habit_values

    # Three positions; code 7 is chosen whenever available, code 9 never.
    positions = [(np.array([7, 9, 11]), 0), (np.array([7, 9]), 0), (np.array([9, 11]), 1)]
    table, base = build_habits(positions)
    assert table[7] == [2, 2] and table[9] == [0, 3]
    full = habit_values(np.array([7, 9, 11]), table, base)
    assert full[0] > 0 > full[1]
    # Leave-one-out: in a position where 7 was chosen, its own choice must not count.
    loo = habit_values(np.array([7, 9, 11]), table, base, loo_chosen=0)
    assert loo[0] < full[0]
    assert np.all(habit_values(np.array([7, 9]), {}, 0.0) == 0)


def test_plans_in_the_opening_are_not_attacks():
    from cognicheck.predictor import _plan_of

    b = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6"):
        b.push_uci(u)
    pa = analyse_position(b)
    plans = {_plan_of(pa.X[i], pa.phase) for i in range(len(pa.moves))}
    assert "develop" in plans
    i = pa.index(chess.Move.from_uci("f1c4"))
    assert _plan_of(pa.X[i], pa.phase) == "develop"
    assert pa.X[pa.index(chess.Move.from_uci("d2d4")), IDX["pawn_storm"]] == 0
