"""Turn a position into one interpretable feature vector per legal move.

The per-player model is a softmax over these vectors, so every learned
weight has a human reading ("+0.8 on knight moves", "-0.3 on ignoring a
threat"). Keep features roughly in [-1, 1] / {0, 1} so weights compare.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from . import tactics
from .engine import Engine

FEATURES: list[tuple[str, str]] = [
    ("pawn_move", "moves a pawn"),
    ("knight_move", "moves a knight"),
    ("bishop_move", "moves a bishop"),
    ("rook_move", "moves a rook"),
    ("queen_move", "moves the queen"),
    ("king_move", "moves the king (not castling)"),
    ("castle", "castles"),
    ("capture", "captures"),
    ("capture_gain", "material won by a capture (exchange-evaluated)"),
    ("equal_trade", "offers/accepts an even trade"),
    ("check", "gives check"),
    ("mate", "delivers mate"),
    ("promotion", "promotes"),
    ("forward", "moves forward"),
    ("backward", "moves backward"),
    ("distance", "move length"),
    ("centrality", "lands near the centre"),
    ("enemy_half", "lands in the opponent's half"),
    ("hangs", "leaves material en prise"),
    ("saves", "rescues a threatened piece"),
    ("ignores_threat", "ignores an existing threat"),
    ("recapture", "recaptures immediately"),
    ("same_piece", "moves the same piece again"),
    ("reacts", "plays near the opponent's last move"),
    ("king_approach", "moves towards the enemy king"),
    ("pawn_storm", "pushes a pawn at the enemy king"),
    ("develops", "develops a minor piece (opening)"),
    ("early_queen", "brings the queen out early"),
    ("undo", "undoes its previous move"),
    ("long_diagonal", "uses a long diagonal"),
    ("eng_loss", "objective loss in pawns (engine)"),
    ("eng_best", "plays the engine's top move"),
    ("eng_blunder", "makes an objective blunder (engine)"),
]
FEATURE_NAMES = [f for f, _ in FEATURES]
FEATURE_LABEL = dict(FEATURES)
IDX = {f: i for i, f in enumerate(FEATURE_NAMES)}
ENGINE_FEATURES = {"eng_loss", "eng_best", "eng_blunder"}
D = len(FEATURE_NAMES)

PIECE_FEATURE = {
    chess.PAWN: "pawn_move",
    chess.KNIGHT: "knight_move",
    chess.BISHOP: "bishop_move",
    chess.ROOK: "rook_move",
    chess.QUEEN: "queen_move",
    chess.KING: "king_move",
}


def _centrality(sq: chess.Square) -> float:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    d = max(abs(f - 3.5), abs(r - 3.5))  # 0.5 .. 3.5
    return (3.5 - d) / 3.0


def _non_pawn_material(board: chess.Board) -> int:
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += tactics.VALUE[pt] * len(board.pieces(pt, chess.WHITE) | board.pieces(pt, chess.BLACK))
    return total


def phase(board: chess.Board) -> str:
    mat = _non_pawn_material(board)
    if board.fullmove_number <= 10 and mat >= 50:
        return "opening"
    if mat <= 20:
        return "endgame"
    return "middlegame"


@dataclass
class PositionAnalysis:
    """Everything we know about one decision point."""

    board: chess.Board
    moves: list[chess.Move]
    X: np.ndarray
    sans: list[str]
    after_gain: np.ndarray  # opponent's best exchange gain after each move
    after_capture: list[chess.Move | None]
    threat_gain: int  # what the opponent wins if we "passed"
    threats: list[tactics.Capture]  # opponent captures on the null-moved board
    mate_threats: list[chess.Move]
    opportunities: list[tactics.Capture]  # our captures winning >= 2
    mates: list[chess.Move]
    cp: np.ndarray | None = None  # engine score per move (mover pov)
    replies: list[chess.Move | None] = field(default_factory=list)
    phase: str = "middlegame"

    def index(self, move: chess.Move) -> int:
        return self.moves.index(move)

    @property
    def has_engine(self) -> bool:
        return self.cp is not None


def analyse_position(board: chess.Board, engine: Engine | None = None) -> PositionAnalysis:
    board = board.copy()
    color = board.turn
    moves = list(board.legal_moves)
    n = len(moves)
    X = np.zeros((n, D), dtype=np.float32)
    ph = phase(board)

    last = board.move_stack[-1] if board.move_stack else None
    my_last = board.move_stack[-2] if len(board.move_stack) >= 2 else None
    enemy_king = board.king(not color)

    threats = tactics.threats_against_side_to_move(board)
    threat_gain = threats[0].gain if threats else 0
    mate_threats = tactics.mate_threat(board)
    opportunities = tactics.winning_captures(board, min_gain=2)

    engine_scores = engine.evaluate_all(board) if engine is not None and n else None
    cp = None
    replies: list[chess.Move | None] = [None] * n
    if engine_scores:
        cp = np.array([engine_scores[m.uci()].cp for m in moves], dtype=np.float32)
        replies = [engine_scores[m.uci()].reply for m in moves]
        best = cp.max()

    after_gain = np.zeros(n, dtype=np.float32)
    after_capture: list[chess.Move | None] = [None] * n
    sans: list[str] = []
    mates: list[chess.Move] = []

    for i, m in enumerate(moves):
        piece = board.piece_at(m.from_square)
        pt = piece.piece_type
        row = X[i]
        sans.append(board.san(m))
        is_castle = board.is_castling(m)
        if is_castle:
            row[IDX["castle"]] = 1
        else:
            row[IDX[PIECE_FEATURE[pt]]] = 1

        is_cap = board.is_capture(m)
        gain = tactics.see(board, m)
        if is_cap:
            row[IDX["capture"]] = 1
            row[IDX["capture_gain"]] = np.clip(gain, -9, 9) / 3.0
            if gain == 0:
                row[IDX["equal_trade"]] = 1
            if last is not None and m.to_square == last.to_square:
                row[IDX["recapture"]] = 1
        gives_check = board.gives_check(m)
        if gives_check:
            row[IDX["check"]] = 1
        if m.promotion:
            row[IDX["promotion"]] = 1

        dy = chess.square_rank(m.to_square) - chess.square_rank(m.from_square)
        dy = dy if color == chess.WHITE else -dy
        dx = chess.square_file(m.to_square) - chess.square_file(m.from_square)
        dist = chess.square_distance(m.from_square, m.to_square)
        if not is_castle:
            if dy > 0:
                row[IDX["forward"]] = 1
            elif dy < 0:
                row[IDX["backward"]] = 1
            row[IDX["distance"]] = dist / 7.0
        if abs(dx) == abs(dy) and dist >= 3 and pt in (chess.BISHOP, chess.QUEEN):
            row[IDX["long_diagonal"]] = 1
        row[IDX["centrality"]] = _centrality(m.to_square)
        rel_rank = chess.square_rank(m.to_square) if color == chess.WHITE else 7 - chess.square_rank(m.to_square)
        if rel_rank >= 4:
            row[IDX["enemy_half"]] = 1

        if my_last is not None:
            if m.from_square == my_last.to_square:
                row[IDX["same_piece"]] = 1
                if m.to_square == my_last.from_square:
                    row[IDX["undo"]] = 1
        if last is not None and chess.square_distance(m.to_square, last.to_square) <= 2:
            row[IDX["reacts"]] = 1

        if enemy_king is not None and pt != chess.KING and not is_castle:
            d0 = chess.square_distance(m.from_square, enemy_king)
            d1 = chess.square_distance(m.to_square, enemy_king)
            row[IDX["king_approach"]] = (d0 - d1) / 7.0
            if pt == chess.PAWN and abs(chess.square_file(m.from_square) - chess.square_file(enemy_king)) <= 1 and dy > 0:
                row[IDX["pawn_storm"]] = 1

        if ph == "opening":
            back = 0 if color == chess.WHITE else 7
            if pt in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(m.from_square) == back:
                row[IDX["develops"]] = 1
            if pt == chess.QUEEN:
                row[IDX["early_queen"]] = 1

        board.push(m)
        if board.is_checkmate():
            row[IDX["mate"]] = 1
            mates.append(m)
            og, oc = 0, None
        else:
            og, oc = tactics.max_opponent_gain(board)
        board.pop()
        after_gain[i] = og
        after_capture[i] = oc
        row[IDX["hangs"]] = min(og, 9) / 3.0
        if threat_gain >= 1:
            if og < threat_gain:
                row[IDX["saves"]] = (threat_gain - og) / 3.0
            elif not gives_check and not (is_cap and gain >= threat_gain):
                row[IDX["ignores_threat"]] = 1

        if cp is not None:
            loss = max(0.0, float(best - cp[i])) / 100.0
            row[IDX["eng_loss"]] = min(loss, 6.0)
            row[IDX["eng_best"]] = 1.0 if loss <= 0.1 else 0.0
            row[IDX["eng_blunder"]] = 1.0 if loss >= 2.0 else 0.0

    return PositionAnalysis(
        board=board,
        moves=moves,
        X=X,
        sans=sans,
        after_gain=after_gain,
        after_capture=after_capture,
        threat_gain=threat_gain,
        threats=threats,
        mate_threats=mate_threats,
        opportunities=opportunities,
        mates=mates,
        cp=cp,
        replies=replies,
        phase=ph,
    )


def addresses_threat(pa: PositionAnalysis, i: int) -> bool:
    """Does move ``i`` deal with the strongest existing threat?"""
    if pa.threat_gain < 1:
        return True
    m = pa.moves[i]
    if pa.after_gain[i] < pa.threat_gain:
        return True
    if pa.X[i, IDX["mate"]] or pa.X[i, IDX["check"]]:
        return True
    if pa.board.is_capture(m) and pa.X[i, IDX["capture_gain"]] * 3 >= pa.threat_gain:
        return True
    return False
