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


def test_attention_and_plan_persistence_features():
    from cognicheck.features import previous_plan

    b = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"):
        b.push_uci(u)
    pa = analyse_position(b, prev_plan=previous_plan(b))
    # Black's previous move Nf6 was development; so is Bc5 now, so same_plan fires.
    assert pa.prev_plan == "develop"
    assert pa.X[pa.index(chess.Move.from_uci("f8c5")), IDX["same_plan"]] == 1
    # Moves to the d-file area, where White just played d3, pick up opponent attention.
    assert pa.X[pa.index(chess.Move.from_uci("d7d5")), IDX["attn_opp"]] > pa.X[pa.index(chess.Move.from_uci("a7a6")), IDX["attn_opp"]]
    # d3 threatened nothing, so reacting there counts as attention capture.
    assert pa.X[pa.index(chess.Move.from_uci("d7d5")), IDX["reacts_harmless"]] == 1


def test_think_time_interactions():
    from cognicheck.features import fill_time_interactions

    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    quick = analyse_position(b, think_time=1.0)
    slow = analyse_position(b, think_time=30.0)
    assert quick.quick and not quick.long and slow.long
    i = quick.index(chess.Move.from_uci("f3e5"))
    assert quick.X[i, IDX["quick_capture"]] == 1 and slow.X[i, IDX["quick_capture"]] == 0
    assert slow.X[quick.index(chess.Move.from_uci("b1c3")), IDX["long_quiet"]] == 1


def test_neural_features_if_available():
    from cognicheck import neural

    if not neural.available() or not (neural.MODEL_DIR / "blitz_model.pt").exists():
        pytest.skip("maia2 weights not installed")
    n = neural.Neural()
    b = chess.Board()
    assert analyse_position(b, neural=n, elo_self=1500, elo_oppo=1500).X[:, IDX["maia_top"]].sum() == 0  # too early
    for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4", "e5d4", "c3d4"):
        b.push_uci(u)
    pa = analyse_position(b, neural=n, elo_self=1500, elo_oppo=1500)
    logp = pa.X[:, IDX["maia_logp"]]
    assert pa.X[:, IDX["maia_top"]].sum() == 1
    assert logp.max() <= 0 and logp[pa.index(chess.Move.from_uci("c5b4"))] > logp[pa.index(chess.Move.from_uci("a7a6"))]


def test_working_memory_features():
    from cognicheck.capacity import capacity_stats
    from cognicheck.features import own_recency, refutation_complexity

    b = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c1g5", "d7d6", "b1c3", "c8g4", "h2h3", "g4f3", "d1f3", "c6d4", "f3d1", "h7h6"):
        b.push_uci(u)
    pa = analyse_position(b)
    last, k = own_recency(b, chess.WHITE)
    assert k == 9 and last[chess.D1] == 9 and last[chess.G5] == 5 and chess.E2 not in last and last[chess.E4] == 1
    # The three most recently moved white pieces: queen (d1), h-pawn (h3), knight (c3).
    assert pa.X[pa.index(chess.Move.from_uci("c3d5")), IDX["focus_recent"]] == 1
    assert pa.X[pa.index(chess.Move.from_uci("g5f6")), IDX["focus_recent"]] == 0
    # The a-pawn was never touched in 9 own moves: stale.
    assert pa.X[pa.index(chess.Move.from_uci("a2a3")), IDX["focus_stale"]] == 1
    assert pa.X[pa.index(chess.Move.from_uci("c3d5")), IDX["focus_stale"]] == 0
    assert 0 < pa.load <= 1
    # Castling moves the rook too.
    c = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "d7d6"):
        c.push_uci(u)
    last_c, _ = own_recency(c, chess.WHITE)
    assert last_c[chess.F1] == 4 and last_c[chess.G1] == 4 and chess.H1 not in last_c
    # A piece simply left hanging is a 2-piece tactic; a capture that also forks involves more.
    simple = chess.Board("4k3/8/8/8/8/8/8/R2r2K1 w - - 0 1")
    assert refutation_complexity(simple, chess.Move.from_uci("a1d1")) == 2
    forking = chess.Board("4k3/8/8/8/8/8/2p5/R3K1Q1 b - - 0 1")  # ...cxb1=N would fork? use a knight capture: Nxc2 forking a1 and e1
    forking = chess.Board("4k3/8/8/8/3n4/8/2P5/R3K3 b - - 0 1")
    assert refutation_complexity(forking, chess.Move.from_uci("d4c2")) >= 3
    # capacity_stats on a tiny synthetic dataset
    X = np.zeros((6, len(FEATURE_NAMES)))
    X[0, IDX["allows_simple"]] = 1
    X[3, IDX["focus_recent"]] = 1
    out = capacity_stats(X, np.array([0, 3]), np.array([0, 3]), np.array([0.2, 0.7]), np.array([False, True]))
    assert out["complexity"]["simple"]["walked_in"] == 1 and out["recent"]["chosen_share"] == 0.5
