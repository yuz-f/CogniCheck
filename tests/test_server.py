import os

import pytest

os.environ["COGNICHECK_NO_ENGINE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from cognicheck import cli, server  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.profile import build_profile  # noqa: E402

from .test_core import SAMPLE_PGN  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    prof = build_profile("Alice", games_of(parse_pgn(SAMPLE_PGN), "Alice"))
    prof.save(tmp_path)
    monkeypatch.setattr(server, "PROFILES", tmp_path)
    monkeypatch.setattr(cli, "PROFILES", tmp_path)
    from cognicheck import cohort

    monkeypatch.setattr(cohort, "PROFILES_REAL", tmp_path / "real")
    server._profiles.clear()
    return TestClient(server.app)


def test_players_and_predict(client):
    players = client.get("/api/players").json()
    assert [p["id"] for p in players] == ["alice"]
    summary = client.get("/api/players/alice").json()
    assert "blind_spots" in summary and "style" in summary
    r = client.post("/api/predict", json={"player_id": "alice", "moves": ["e2e4", "e7e5"]})
    assert r.status_code == 200
    assert r.json()["turn"] == "white"
    assert client.post("/api/predict", json={"player_id": "alice", "moves": ["e2e5"]}).status_code == 400


def test_game_endpoints(client):
    games = client.get("/api/players/alice/games").json()
    assert games
    s = client.post("/api/game", json={"player_id": "alice", "game_index": 0}).json()
    r = client.post(f"/api/game/{s['id']}/guess", json={"uci": s["legal"][0]})
    assert r.status_code == 200
    body = r.json()
    assert {"round", "prediction", "state", "reality"} <= set(body)
    h = client.post(f"/api/game/{s['id']}/hint").json()
    assert "notes" in h["hint"]


def test_position_endpoint(client):
    r = client.post("/api/position", json={"moves": ["e2e4", "e7e5", "g1f3"]}).json()
    assert r["history"] == ["e4", "e5", "Nf3"]
    assert r["turn"] == "black"


def test_index_served(client):
    assert client.get("/").status_code == 200
