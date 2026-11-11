"""Load a player's games from PGN text, Lichess or Chess.com."""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import chess
import chess.pgn

USER_AGENT = "CogniCheck/0.1 (chess move prediction)"


@dataclass
class PlayerGame:
    game: chess.pgn.Game
    color: chess.Color  # colour the player had in this game

    @property
    def headers(self):
        return self.game.headers

    def pgn(self) -> str:
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return self.game.accept(exporter)


def parse_pgn(text: str) -> list[chess.pgn.Game]:
    games = []
    stream = io.StringIO(text)
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        if g.errors:
            continue
        if g.headers.get("Variant", "Standard").lower() not in ("standard", "", "from position"):
            continue
        if g.headers.get("SetUp") == "1":
            continue
        games.append(g)
    return games


def player_names(games: list[chess.pgn.Game]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for g in games:
        for side in ("White", "Black"):
            n = g.headers.get(side, "?")
            counts[n] = counts.get(n, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def games_of(games: list[chess.pgn.Game], player: str | None = None) -> list[PlayerGame]:
    """Select the games of ``player`` (case-insensitive). Defaults to the most frequent name."""
    if not games:
        return []
    if player is None:
        player = next(iter(player_names(games)))
    key = player.lower()
    out = []
    for g in games:
        if g.headers.get("White", "").lower() == key:
            out.append(PlayerGame(g, chess.WHITE))
        elif g.headers.get("Black", "").lower() == key:
            out.append(PlayerGame(g, chess.BLACK))
    return out


def _get(url: str, accept: str = "*/*", timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_lichess(username: str, max_games: int = 300, perf: str | None = "blitz,rapid,classical") -> str:
    params = {"max": str(max_games), "clocks": "true", "evals": "false", "opening": "false"}
    if perf:
        params["perfType"] = perf
    url = f"https://lichess.org/api/games/user/{urllib.parse.quote(username)}?{urllib.parse.urlencode(params)}"
    return _get(url, accept="application/x-chess-pgn", timeout=120)


def fetch_chesscom(username: str, max_games: int = 300) -> str:
    user = urllib.parse.quote(username.lower())
    archives = json.loads(_get(f"https://api.chess.com/pub/player/{user}/games/archives"))["archives"]
    pgns: list[str] = []
    for url in reversed(archives):
        month = json.loads(_get(url))
        for g in reversed(month.get("games", [])):
            if g.get("rules") == "chess" and g.get("pgn"):
                pgns.append(g["pgn"])
            if len(pgns) >= max_games:
                return "\n\n".join(pgns)
    return "\n\n".join(pgns)
