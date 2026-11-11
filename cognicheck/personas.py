"""Synthetic "ghost" players with known, planted habits and blind spots.

These exist so the system can be demonstrated and *verified* offline: each
persona is a Stockfish-driven bot whose perception is deliberately broken
in one way (for example it cannot see the opponent's knight moves) and whose
taste is biased (for example it loves pawn storms). If the profiler is any
good, it should rediscover those planted traits from the games alone.

How a persona picks a move:
    perceived(m) = engine score of m, except when the engine's refutation is
                   a move of a kind the persona is blind to; then the persona
                   evaluates m as if that refutation did not exist.
    utility(m)   = perceived(m) / T + style . features(m)
    move        ~ softmax(utility) over the engine's top-K candidates
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

import chess
import chess.pgn
import numpy as np

from . import tactics
from .engine import MATE_CP, Engine
from .features import IDX, analyse_position


@dataclass
class Persona:
    name: str
    elo: int
    temperature: float  # centipawns; higher = sloppier
    style: dict[str, float] = field(default_factory=dict)
    blind: set[str] = field(default_factory=set)
    description: str = ""


PERSONAS = [
    Persona(
        name="Knightmare Kai",
        elo=1500,
        temperature=55,
        style={"knight_move": 0.9, "check": 0.4, "reacts": 0.3},
        blind={"knight"},
        description="Adores knight manoeuvres, but cannot see the opponent's knight jumps coming (forks!).",
    ),
    Persona(
        name="Stormy Sasha",
        elo=1450,
        temperature=70,
        style={"pawn_storm": 1.3, "forward": 0.5, "king_approach": 1.5, "backward": -0.6},
        blind={"backward"},
        description="Throws pawns at the enemy king and never retreats; blind to the opponent's backward moves.",
    ),
    Persona(
        name="Tidy Tomas",
        elo=1600,
        temperature=45,
        style={"equal_trade": 1.2, "castle": 1.5, "early_queen": -1.0, "develops": 0.6},
        blind={"long_diagonal"},
        description="Solid trader who castles early; long diagonal shots from bishops and queens escape him.",
    ),
]

OPPONENT = Persona(name="Club Opponent", elo=1500, temperature=90, description="Generic engine-softmax sparring partner.")


def _blind_to(board_after: chess.Board, reply: chess.Move | None, blind: set[str]) -> bool:
    if reply is None or not blind or reply not in board_after.legal_moves:
        return False
    tags = tactics.motifs(board_after, reply)
    return bool(tags & blind)


def choose_move(board: chess.Board, persona: Persona, engine: Engine, rng: random.Random, k: int = 8, opening_noise: bool = False) -> chess.Move:
    cands = engine.top_moves(board, k)
    if not cands:
        return next(iter(board.legal_moves))
    pa = analyse_position(board) if persona.style else None
    utils = []
    for move, score, reply in cands:
        perceived = score
        if persona.blind:
            board.push(move)
            if _blind_to(board, reply, persona.blind):
                visible = [m for m in board.legal_moves if not (tactics.motifs(board, m) & persona.blind)]
                if visible:
                    perceived = -engine.best_score(board, root_moves=visible, nodes=6000)
            board.pop()
        perceived = max(-MATE_CP, min(MATE_CP, perceived))
        t = persona.temperature * (2.5 if opening_noise else 1.0)
        u = perceived / t
        if pa is not None:
            x = pa.X[pa.index(move)]
            u += sum(w * float(x[IDX[f]]) for f, w in persona.style.items())
        utils.append(u)
    u = np.array(utils)
    p = np.exp(u - u.max())
    p /= p.sum()
    return cands[int(rng.choices(range(len(cands)), weights=p)[0])][0]


def play_game(white: Persona, black: Persona, engine: Engine, rng: random.Random, max_plies: int = 110, round_no: int = 1) -> chess.pgn.Game:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        p = white if board.turn == chess.WHITE else black
        board.push(choose_move(board, p, engine, rng, opening_noise=len(board.move_stack) < 6))
    result = board.result(claim_draw=True)
    if result == "*":
        s = engine.best_score(board, nodes=20000)
        s = s if board.turn == chess.WHITE else -s
        result = "1-0" if s > 300 else "0-1" if s < -300 else "1/2-1/2"
    g = chess.pgn.Game.from_board(board)
    g.headers.update(
        {
            "Event": "CogniCheck synthetic games",
            "Site": "local",
            "Date": dt.date.today().strftime("%Y.%m.%d"),
            "Round": str(round_no),
            "White": white.name,
            "Black": black.name,
            "WhiteElo": str(white.elo),
            "BlackElo": str(black.elo),
            "Result": result,
        }
    )
    return g


def generate(persona: Persona, n_games: int, seed: int = 0, engine: Engine | None = None) -> list[chess.pgn.Game]:
    own = engine is None
    engine = engine or Engine()
    rng = random.Random(seed)
    games = []
    try:
        for i in range(n_games):
            if i % 2 == 0:
                games.append(play_game(persona, OPPONENT, engine, rng, round_no=i + 1))
            else:
                games.append(play_game(OPPONENT, persona, engine, rng, round_no=i + 1))
    finally:
        if own:
            engine.close()
    return games


def by_name(name: str) -> Persona:
    for p in PERSONAS:
        if name.lower() in (p.name.lower(), *p.name.lower().split()):
            return p
    raise KeyError(name)

