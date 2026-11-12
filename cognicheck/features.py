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
    ("near_king", "lands next to the enemy king"),
    ("eng_loss", "objective loss in pawns (engine)"),
    ("eng_best", "plays the engine's top move"),
    ("eng_blunder", "makes an objective blunder (engine)"),
    ("eng_loss_small", "slightly inaccurate (0.1-0.5 pawns, engine)"),
    ("eng_loss_mid", "clearly inaccurate (0.5-2 pawns, engine)"),
    ("eng_rank", "engine rank of the move (0 = best, 1 = worst)"),
    ("allows_knight", "allows a knight tactic"),
    ("allows_bishop", "allows a bishop tactic"),
    ("allows_rook", "allows a rook tactic"),
    ("allows_queen", "allows a queen tactic"),
    ("allows_pawn", "allows a pawn tactic"),
    ("allows_check", "allows a check that wins material"),
    ("allows_fork", "allows a fork"),
    ("allows_discovered", "allows a discovered attack"),
    ("allows_backward", "allows a backward-move tactic"),
    ("allows_long_range", "allows a long-range tactic"),
    ("tp_eng_loss", "objective loss, when short of time"),
    ("tp_hangs", "leaves material en prise, when short of time"),
    ("tp_capture", "captures, when short of time"),
    ("habit", "a move this player habitually makes (piece and square)"),
    ("mobility_gain", "gains mobility for the moved piece"),
    ("passed_pawn", "creates or advances a passed pawn"),
    ("opens_file", "opens a file for a rook or queen"),
    ("attacks_weak_pawn", "attacks an undefended pawn"),
    ("weakens_shield", "moves a pawn in front of the own king"),
    ("isolates_pawn", "leaves a pawn isolated"),
    ("quick_habit", "habitual move, played quickly"),
    ("quick_eng_loss", "objective loss, played quickly"),
    ("quick_capture", "capture, played quickly"),
    ("long_eng_loss", "objective loss, after a long think"),
    ("long_quiet", "quiet move, after a long think"),
    ("attn_opp", "goes where the opponent has just been active"),
    ("attn_own", "goes where the player has just been active"),
    ("reacts_harmless", "answers the opponent's last move although it threatened nothing"),
    ("same_plan", "continues the player's previous plan"),
    ("maia_logp", "how typical the move is for a human of this rating (Maia-2)"),
    ("maia_top", "the move a typical human of this rating would play (Maia-2)"),
    ("allows_simple", "allows a simple tactic (2 pieces involved)"),
    ("allows_complex", "allows a complex tactic (4+ pieces involved)"),
    ("load_eng_loss", "objective loss, in a crowded position (many pieces in contact)"),
    ("load_hangs", "leaves material en prise, in a crowded position"),
    ("focus_recent", "moves one of the 3 pieces moved most recently"),
    ("focus_stale", "moves a piece untouched for 8+ own moves"),
]
NEURAL_FEATURES = {"maia_logp", "maia_top"}
NEURAL_FROM_PLY = 10
# Working-memory features: computable without an engine (the load_* products need
# the engine columns, which exist in the stored analysis).
MEMORY_FEATURES = ["allows_simple", "allows_complex", "focus_recent", "focus_stale"]
LOAD_FEATURES = {"load_eng_loss": "eng_loss", "load_hangs": "hangs"}
MAX_CONTACTS = 16  # attacker-victim pairs; normalises the "load" of a position
FOCUS_RECENT = 3  # pieces a player keeps in working memory (Cowan's 3-5)
FOCUS_STALE = 8  # own moves without touching a piece
PLANS = [
    ("mate", "go for mate"),
    ("attack", "attack the king"),
    ("grab", "win material"),
    ("defend", "defend / save a piece"),
    ("trade", "trade pieces"),
    ("develop", "develop & castle"),
    ("pawns", "push pawns / gain space"),
    ("manoeuvre", "reposition pieces"),
]
PLAN_LABEL = dict(PLANS)
QUICK_SECONDS = 3.0
LONG_SECONDS = 15.0
ALLOW_MOTIFS = ["knight", "bishop", "rook", "queen", "pawn", "check", "fork", "discovered", "backward", "long_range"]
TIME_PRESSURE_SECONDS = 20
HABIT_ALPHA = 2.0
FEATURE_NAMES = [f for f, _ in FEATURES]
FEATURE_LABEL = dict(FEATURES)
IDX = {f: i for i, f in enumerate(FEATURE_NAMES)}
ENGINE_FEATURES = {"eng_loss", "eng_best", "eng_blunder", "eng_loss_small", "eng_loss_mid", "eng_rank", "tp_eng_loss", "quick_eng_loss", "long_eng_loss", "load_eng_loss"}
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
    codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))  # (piece, to-square) per move
    time_pressure: bool = False
    quick: bool = False  # the move was (or is being) played within QUICK_SECONDS
    long: bool = False
    prev_plan: str | None = None
    load: float = 0.0  # pieces in contact / MAX_CONTACTS (position-level)

    def index(self, move: chess.Move) -> int:
        return self.moves.index(move)

    @property
    def has_engine(self) -> bool:
        return self.cp is not None


def move_code(board: chess.Board, move: chess.Move) -> int:
    """(piece type, destination) in 0..383: the unit of a player's habits."""
    piece = board.piece_at(move.from_square)
    return (piece.piece_type - 1) * 64 + move.to_square if piece else 0


def habit_values(codes, table: dict, base_rate: float, loo_chosen: int | None = None) -> np.ndarray:
    """Log-rate of choosing each (piece, square) relative to the player's base rate.

    ``table`` maps code -> [times chosen, times available]. With ``loo_chosen``
    the current position is itself in the table and is left out (exact
    leave-one-out), which stops the chosen move from recognising itself.
    """
    out = np.zeros(len(codes), dtype=np.float32)
    if not table or base_rate <= 0:
        return out
    for i, c in enumerate(codes):
        ch, av = table.get(int(c), (0, 0))
        if loo_chosen is not None:
            av -= 1
            if i == loo_chosen:
                ch -= 1
        if av <= 0:
            continue
        r = (ch + HABIT_ALPHA * base_rate) / (av + HABIT_ALPHA)
        out[i] = np.clip(np.log(r / base_rate), -3, 3)
    return out


def build_habits(positions) -> tuple[dict, float]:
    """``positions``: iterable of (codes, chosen index). Returns (table, base rate)."""
    table: dict[int, list[int]] = {}
    chosen = avail = 0
    for codes, c in positions:
        for i, code in enumerate(codes):
            row = table.setdefault(int(code), [0, 0])
            row[1] += 1
            avail += 1
            if i == c:
                row[0] += 1
                chosen += 1
    return table, (chosen / avail if avail else 0.0)


def _last_was_capture(board: chess.Board) -> bool:
    mv = board.move_stack[-1]
    b = board.copy()
    b.pop()
    return b.is_capture(mv)


def _is_passed(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ranks = range(r + 1, 8) if color == chess.WHITE else range(0, r)
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            for rr in ranks:
                p = board.piece_at(chess.square(ff, rr))
                if p and p.piece_type == chess.PAWN and p.color != color:
                    return False
    return True


def _is_isolated(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    f = chess.square_file(sq)
    own = board.pieces(chess.PAWN, color)
    return not any(chess.square_file(s) in (f - 1, f + 1) for s in own)


def fill_time_interactions(X: np.ndarray, quick: bool, long: bool) -> None:
    """Think-time interactions; called after the habit column is filled."""
    X[:, IDX["quick_habit"]] = X[:, IDX["habit"]] if quick else 0.0
    X[:, IDX["quick_eng_loss"]] = X[:, IDX["eng_loss"]] if quick else 0.0
    X[:, IDX["quick_capture"]] = X[:, IDX["capture"]] if quick else 0.0
    X[:, IDX["long_eng_loss"]] = X[:, IDX["eng_loss"]] if long else 0.0
    X[:, IDX["long_quiet"]] = (1.0 - X[:, IDX["capture"]]) if long else 0.0


def plan_of(x: np.ndarray, phase: str = "middlegame", threat_gain: int = 0) -> str:
    """Read an intention off a move's features. Order matters: the most
    specific reading wins, and in the opening development beats 'attack'.
    Parrying a mere pawn threat with a developing move reads as development."""
    if x[IDX["mate"]]:
        return "mate"
    develops = x[IDX["develops"]] or x[IDX["castle"]]
    if x[IDX["saves"]] > 0 and (threat_gain >= 2 or not develops):
        return "defend"
    if x[IDX["capture_gain"]] > 0:
        return "grab"
    if x[IDX["equal_trade"]]:
        return "trade"
    if phase == "opening" and develops:
        return "develop"
    if x[IDX["check"]] or x[IDX["pawn_storm"]] or x[IDX["near_king"]]:
        return "attack"
    if develops:
        return "develop"
    if x[IDX["pawn_move"]]:
        return "pawns"
    return "manoeuvre"


def _attention(stack: list[chess.Move], sq_from: chess.Square, sq_to: chess.Square) -> float:
    """Decayed attention from recent moves (most recent first) to a move's squares."""
    total = 0.0
    for k, mv in enumerate(stack[:3]):
        w = 0.5**k
        for s_ in (sq_to, sq_from):
            d = min(chess.square_distance(s_, mv.to_square), chess.square_distance(s_, mv.from_square) + 1)
            if d <= 1:
                total += w * (1.0 if d == 0 else 0.5)
    return min(total, 1.0)


def contacts(board: chess.Board) -> int:
    """Number of attacker-victim pairs on the board: how much there is to keep track of."""
    n = 0
    for sq, p in board.piece_map().items():
        n += len(board.attackers(not p.color, sq))
    return n


def refutation_complexity(board: chess.Board, cap: chess.Move) -> int:
    """Pieces a player must hold in mind to see the tactic ``cap`` (board after our move).

    Counts every piece bearing on the target square (attackers and defenders of both
    colours, the capturer included), the target itself, further targets the capturer
    hits from its landing square (forks) and pieces it unmasks (discovered attacks).
    """
    tgt = cap.to_square
    involved = set(board.attackers(chess.WHITE, tgt)) | set(board.attackers(chess.BLACK, tgt))
    involved.add(tgt)
    b = board.copy(stack=False)
    b.push(cap)
    mover = not b.turn
    for t in tactics._valuable_targets(b, tgt, mover):
        involved.add(t)
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s_ in b.pieces(pt, mover):
            if s_ == tgt:
                continue
            for t in tactics._valuable_targets(b, s_, mover):
                if cap.from_square in chess.SquareSet.between(s_, t):
                    involved.add(s_)
                    involved.add(t)
    return len(involved)


def own_recency(board: chess.Board, color: chess.Color) -> tuple[dict[int, int], int]:
    """For each square holding a piece of ``color``: the ordinal of the player's own
    move that last touched it (0 = never). Also returns how many own moves were made.
    Reconstructed from the move stack without replaying the game."""
    stack = board.move_stack
    n = len(stack)
    last: dict[int, int] = {}
    k = 0
    for i, mv in enumerate(stack):
        mover = board.turn if (n - i) % 2 == 0 else not board.turn
        if mover == color:
            k += 1
            last.pop(mv.from_square, None)
            last[mv.to_square] = k
            if mv.from_square in (chess.E1, chess.E8) and abs(chess.square_file(mv.to_square) - 4) == 2:
                rank = chess.square_rank(mv.from_square)
                if chess.square_file(mv.to_square) == 6:
                    last.pop(chess.square(7, rank), None)
                    last[chess.square(5, rank)] = k
                else:
                    last.pop(chess.square(0, rank), None)
                    last[chess.square(3, rank)] = k
        else:
            last.pop(mv.to_square, None)  # our piece was captured there
    return last, k


def previous_plan(board: chess.Board) -> str | None:
    """Plan of the side-to-move's previous move, read from the position before it."""
    if len(board.move_stack) < 2:
        return None
    b = board.copy()
    opp = b.pop()
    mine = b.pop()
    pa = analyse_position(b, None)  # no engine: plan_of does not need it
    try:
        i = pa.index(mine)
    except ValueError:
        return None
    return plan_of(pa.X[i], pa.phase, pa.threat_gain)


def analyse_position(
    board: chess.Board,
    engine: Engine | None = None,
    time_left: float | None = None,
    base_time: float | None = None,
    think_time: float | None = None,
    prev_plan: str | None = None,
    neural=None,
    elo_self: int | None = None,
    elo_oppo: int | None = None,
) -> PositionAnalysis:
    board = board.copy()
    color = board.turn
    moves = list(board.legal_moves)
    n = len(moves)
    X = np.zeros((n, D), dtype=np.float32)
    ph = phase(board)
    tp = time_left is not None and (time_left < TIME_PRESSURE_SECONDS or (bool(base_time) and time_left < 0.1 * base_time))
    quick = think_time is not None and think_time <= QUICK_SECONDS
    long = think_time is not None and think_time >= LONG_SECONDS
    codes = np.array([move_code(board, m) for m in moves], dtype=np.int16)
    own_king = board.king(color)
    back = 0 if color == chess.WHITE else 7
    king_castled = own_king is not None and chess.square_rank(own_king) == back and chess.square_file(own_king) not in (3, 4)

    last = board.move_stack[-1] if board.move_stack else None
    my_last = board.move_stack[-2] if len(board.move_stack) >= 2 else None
    opp_recent = board.move_stack[-1::-2][:3]  # opponent's last moves, most recent first
    own_recent = board.move_stack[-2::-2][:3]
    last_threatening = last is not None and (board.is_check() or (len(board.move_stack) >= 1 and _last_was_capture(board)))
    enemy_king = board.king(not color)

    threats = tactics.threats_against_side_to_move(board)
    threat_gain = threats[0].gain if threats else 0
    mate_threats = tactics.mate_threat(board)
    opportunities = tactics.winning_captures(board, min_gain=2)

    load = min(contacts(board), MAX_CONTACTS) / MAX_CONTACTS
    last_touch, own_moves = own_recency(board, color)
    recent_squares = {sq for sq, _ in sorted(last_touch.items(), key=lambda kv: -kv[1])[:FOCUS_RECENT]}

    engine_scores = engine.evaluate_all(board) if engine is not None and n else None
    cp = None
    replies: list[chess.Move | None] = [None] * n
    if engine_scores:
        cp = np.array([engine_scores[m.uci()].cp for m in moves], dtype=np.float32)
        replies = [engine_scores[m.uci()].reply for m in moves]
        best = cp.max()
        rank = np.empty(n, dtype=np.float32)
        rank[np.argsort(-cp, kind="stable")] = np.arange(n)

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
        if m.from_square in recent_squares:
            row[IDX["focus_recent"]] = 1
        if own_moves >= FOCUS_STALE and own_moves - last_touch.get(m.from_square, 0) >= FOCUS_STALE:
            row[IDX["focus_stale"]] = 1

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
        row[IDX["attn_opp"]] = _attention(opp_recent, m.from_square, m.to_square)
        row[IDX["attn_own"]] = _attention(own_recent, m.from_square, m.to_square)

        if enemy_king is not None and pt != chess.KING and not is_castle:
            d0 = chess.square_distance(m.from_square, enemy_king)
            d1 = chess.square_distance(m.to_square, enemy_king)
            row[IDX["king_approach"]] = (d0 - d1) / 7.0
            if pt != chess.PAWN and d1 <= 2:
                row[IDX["near_king"]] = 1
            king_file = chess.square_file(enemy_king)
            # A pawn storm aims at a king that has left the centre (or is already close).
            if pt == chess.PAWN and dy > 0 and rel_rank >= 4 and abs(chess.square_file(m.to_square) - king_file) <= 1 and (king_file not in (3, 4) or d1 <= 2):
                row[IDX["pawn_storm"]] = 1

        if ph == "opening":
            back = 0 if color == chess.WHITE else 7
            if pt in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(m.from_square) == back:
                row[IDX["develops"]] = 1
            if pt == chess.QUEEN:
                row[IDX["early_queen"]] = 1

        if pt != chess.PAWN and not is_castle:
            mob_before = len(board.attacks(m.from_square))
        else:
            mob_before = None
        if pt == chess.PAWN and king_castled and abs(chess.square_file(m.from_square) - chess.square_file(own_king)) <= 1 and chess.square_rank(m.from_square) == (1 if color == chess.WHITE else 6):
            row[IDX["weakens_shield"]] = 1

        board.push(m)
        if mob_before is not None:
            row[IDX["mobility_gain"]] = np.clip((len(board.attacks(m.to_square)) - mob_before) / 10.0, -1, 1)
        if pt == chess.PAWN and not m.promotion:
            if _is_passed(board, m.to_square, color):
                row[IDX["passed_pawn"]] = 1
            if _is_isolated(board, m.to_square, color):
                row[IDX["isolates_pawn"]] = 1
            ff = chess.square_file(m.from_square)
            if True:
                own_pawns_on_file = any(chess.square_file(sq) == ff for sq in board.pieces(chess.PAWN, color))
                heavy_on_file = any(chess.square_file(sq) == ff for sq in board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color))
                if not own_pawns_on_file and heavy_on_file:
                    row[IDX["opens_file"]] = 1
        if pt != chess.KING:
            for t in board.attacks(m.to_square):
                p_ = board.piece_at(t)
                if p_ and p_.color != color and p_.piece_type == chess.PAWN and not board.is_attacked_by(not color, t):
                    row[IDX["attacks_weak_pawn"]] = 1
                    break
        if board.is_checkmate():
            row[IDX["mate"]] = 1
            mates.append(m)
            og, oc = 0, None
        else:
            og, oc = tactics.max_opponent_gain(board)
            if og >= 1 and oc is not None:
                for t in tactics.motifs(board, oc):
                    if t in ALLOW_MOTIFS:
                        row[IDX["allows_" + t]] = 1
                cx = refutation_complexity(board, oc)
                if cx <= 2:
                    row[IDX["allows_simple"]] = 1
                elif cx >= 4:
                    row[IDX["allows_complex"]] = 1
        board.pop()
        after_gain[i] = og
        after_capture[i] = oc
        row[IDX["hangs"]] = min(og, 9) / 3.0
        row[IDX["load_hangs"]] = load * row[IDX["hangs"]]
        if tp:
            row[IDX["tp_hangs"]] = row[IDX["hangs"]]
            row[IDX["tp_capture"]] = row[IDX["capture"]]
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
            row[IDX["eng_loss_small"]] = 1.0 if 0.1 < loss <= 0.5 else 0.0
            row[IDX["eng_loss_mid"]] = 1.0 if 0.5 < loss < 2.0 else 0.0
            row[IDX["eng_rank"]] = rank[i] / (n - 1) if n > 1 else 0.0
            row[IDX["load_eng_loss"]] = load * row[IDX["eng_loss"]]
            if tp:
                row[IDX["tp_eng_loss"]] = row[IDX["eng_loss"]]

    # Maia-2 is unreliable in the first plies (its training data starts later);
    # the opening book and habits cover those, so the prior starts at ply 10.
    if neural is not None and n and board.ply() >= NEURAL_FROM_PLY:
        es = int(elo_self or 1500)
        logp, top = neural.features(board, moves, es, int(elo_oppo or es))
        X[:, IDX["maia_logp"]] = logp / 3.0  # roughly [-2, 0]
        X[:, IDX["maia_top"]] = top
    harmless = last is not None and threat_gain == 0 and not mate_threats and not last_threatening
    for i in range(n):
        if harmless and X[i, IDX["reacts"]]:
            X[i, IDX["reacts_harmless"]] = 1
        if prev_plan is not None and plan_of(X[i], ph, threat_gain) == prev_plan:
            X[i, IDX["same_plan"]] = 1
    pa_ = PositionAnalysis(
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
        codes=codes,
        time_pressure=bool(tp),
        quick=bool(quick),
        long=bool(long),
        prev_plan=prev_plan,
        load=load,
    )
    fill_time_interactions(X, bool(quick), bool(long))
    return pa_


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
