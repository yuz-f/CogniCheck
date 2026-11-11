"""Predict a specific player's next move and read the position through their eyes."""
from __future__ import annotations

import chess
import numpy as np

from . import tactics
from .engine import Engine
from .features import FEATURE_LABEL, FEATURE_NAMES, IDX, PositionAnalysis, analyse_position
from .model import softmax

BOOK_ALPHA = 2.0  # pseudo-count of the model inside the opening book

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


def _plan_of(x: np.ndarray) -> str:
    if x[IDX["mate"]]:
        return "mate"
    if x[IDX["saves"]] > 0:
        return "defend"
    if x[IDX["capture_gain"]] > 0:
        return "grab"
    if x[IDX["equal_trade"]]:
        return "trade"
    if x[IDX["check"]] or x[IDX["pawn_storm"]] or x[IDX["king_approach"]] > 0.15:
        return "attack"
    if x[IDX["develops"]] or x[IDX["castle"]]:
        return "develop"
    if x[IDX["pawn_move"]]:
        return "pawns"
    return "manoeuvre"


class Predictor:
    def __init__(self, profile, engine: Engine | None = None):
        self.profile = profile
        self.engine = engine
        self.w = np.asarray(profile.weights, dtype=np.float64)
        self.prior = np.asarray(profile.prior, dtype=np.float64)
        self.motif_index: dict[str, float] = (profile.blind_spots or {}).get("motif_index", {})
        self._rows = {(r["kind"], r["motif"]): r for r in (profile.blind_spots or {}).get("rows", [])}

    # ---------------------------------------------------------------- core
    def model_distribution(self, pa: PositionAnalysis) -> np.ndarray:
        return softmax(pa.X.astype(np.float64) @ self.w)

    def book_counts(self, pa: PositionAnalysis) -> dict[str, int]:
        from .profile import position_key

        return self.profile.book.get(position_key(pa.board), {})

    def distribution(self, pa: PositionAnalysis) -> np.ndarray:
        p = self.model_distribution(pa)
        counts = self.book_counts(pa)
        total = sum(counts.values())
        if total:
            c = np.array([counts.get(m.uci(), 0) for m in pa.moves], dtype=np.float64)
            p = (c + BOOK_ALPHA * p) / (total + BOOK_ALPHA)
        return p

    def analyse(self, board: chess.Board) -> PositionAnalysis:
        return analyse_position(board, self.engine)

    # ------------------------------------------------------------- explain
    def reasons(self, pa: PositionAnalysis, i: int, k: int = 3) -> list[dict]:
        X = pa.X.astype(np.float64)
        centred = X[i] - X.mean(axis=0)
        total = self.w * centred
        personal = (self.w - self.prior) * centred
        out = []
        for j in np.argsort(-np.abs(total))[:k]:
            if abs(total[j]) < 0.15:
                continue
            f = FEATURE_NAMES[j]
            out.append(
                {
                    "feature": f,
                    "label": FEATURE_LABEL[f],
                    "effect": round(float(total[j]), 2),
                    "personal": round(float(personal[j]), 2),
                }
            )
        return out

    def motif_note(self, tags: set[str]) -> tuple[float, str | None]:
        """Worst blind-spot index among the motifs, and its label."""
        best, label = 1.0, None
        for t in tags:
            idx = self.motif_index.get(t)
            if idx and idx > best:
                best, label = idx, tactics.MOTIF_LABEL.get(t, t)
        return best, label

    # --------------------------------------------------------- perspective
    def perspective(self, pa: PositionAnalysis, probs: np.ndarray) -> dict:
        board = pa.board
        notes: list[str] = []

        # Threats against the player.
        threats = []
        seen_targets = set()
        for t in pa.threats[:4]:
            if t.move.to_square in seen_targets:
                continue
            seen_targets.add(t.move.to_square)
            board.push(chess.Move.null())
            san = board.san(t.move)
            tags = tactics.motifs(board, t.move)
            board.pop()
            p_react = 0.0
            for i, m in enumerate(pa.moves):
                board.push(m)
                still = t.move in board.legal_moves and tactics.see(board, t.move) >= 1
                board.pop()
                if not still or pa.X[i, IDX["check"]] or pa.X[i, IDX["mate"]]:
                    p_react += probs[i]
            idx, label = self.motif_note(tags)
            target = board.piece_at(t.move.to_square)
            threats.append(
                {
                    "move": san,
                    "uci": t.move.uci(),
                    "target": chess.square_name(t.move.to_square),
                    "piece": tactics.PIECE_NAME[target.piece_type] if target else "?",
                    "gain": t.gain,
                    "motifs": sorted(tags),
                    "p_notice": round(float(p_react), 3),
                    "blind_index": round(idx, 2),
                    "blind_label": label,
                }
            )
        for m in pa.mate_threats[:1]:
            board.push(chess.Move.null())
            san = board.san(m)
            tags = tactics.motifs(board, m)
            board.pop()
            p_react = 0.0
            for i, mv in enumerate(pa.moves):
                board.push(mv)
                if m not in board.legal_moves:
                    p_react += probs[i]
                else:
                    board.push(m)
                    if not board.is_checkmate():
                        p_react += probs[i]
                    board.pop()
                board.pop()
            idx, label = self.motif_note(tags)
            threats.insert(0, {"move": san, "uci": m.uci(), "target": "king", "piece": "king", "gain": 100, "motifs": sorted(tags | {"mate"}), "p_notice": round(float(p_react), 3), "blind_index": round(idx, 2), "blind_label": label})

        # Chances the player has.
        chances = []
        for m in pa.mates[:2]:
            i = pa.index(m)
            tags = tactics.motifs(board, m)
            idx, label = self.motif_note(tags)
            chances.append({"move": pa.sans[i], "uci": m.uci(), "gain": 100, "kind": "mate", "motifs": sorted(tags), "p_find": round(float(probs[i]), 3), "blind_index": round(idx, 2), "blind_label": label})
        for c in pa.opportunities[:3]:
            i = pa.index(c.move)
            tags = tactics.motifs(board, c.move)
            idx, label = self.motif_note(tags)
            chances.append({"move": pa.sans[i], "uci": c.move.uci(), "gain": c.gain, "kind": "material", "motifs": sorted(tags), "p_find": round(float(probs[i]), 3), "blind_index": round(idx, 2), "blind_label": label})

        # Likely moves that walk into something.
        risky = []
        for i in np.argsort(-probs)[:8]:
            if pa.after_gain[i] >= 2 and probs[i] >= 0.03 and pa.after_capture[i] is not None:
                board.push(pa.moves[i])
                reply = pa.after_capture[i]
                rsan = board.san(reply)
                tags = tactics.motifs(board, reply)
                board.pop()
                idx, label = self.motif_note(tags)
                risky.append({"move": pa.sans[i], "uci": pa.moves[i].uci(), "p": round(float(probs[i]), 3), "punished_by": rsan, "loses": int(pa.after_gain[i]), "motifs": sorted(tags), "blind_index": round(idx, 2), "blind_label": label})
        p_drop = float(probs[pa.after_gain >= 2].sum())

        # Where the player's eyes are.
        heat = np.zeros(64)
        for i, m in enumerate(pa.moves):
            heat[m.to_square] += probs[i]
            heat[m.from_square] += 0.6 * probs[i]
        heat = heat / heat.max() if heat.max() > 0 else heat
        attention = {chess.square_name(s): round(float(heat[s]), 3) for s in range(64) if heat[s] >= 0.05}

        plans: dict[str, float] = {}
        for i in range(len(pa.moves)):
            k = _plan_of(pa.X[i])
            plans[k] = plans.get(k, 0.0) + float(probs[i])
        plan_rows = [{"plan": k, "label": PLAN_LABEL[k], "p": round(v, 3)} for k, v in sorted(plans.items(), key=lambda kv: -kv[1]) if v >= 0.03]

        # Narrative.
        if plan_rows:
            notes.append(f"Most likely intention: {plan_rows[0]['label']} ({plan_rows[0]['p']:.0%}).")
        for t in threats:
            if t["p_notice"] < 0.5:
                why = f" - {t['blind_label']} are a blind spot for this player" if t["blind_label"] and t["blind_index"] >= 1.15 else ""
                notes.append(f"Probably does NOT see {t['move']} hitting the {t['piece']} on {t['target']} (reacts {t['p_notice']:.0%}){why}.")
            else:
                notes.append(f"Will likely notice {t['move']} against the {t['piece']} on {t['target']} ({t['p_notice']:.0%}).")
        for c in chances:
            if c["p_find"] < 0.35:
                why = f" ({c['blind_label']} are a weak point)" if c["blind_label"] and c["blind_index"] >= 1.15 else ""
                what = "mates" if c["kind"] == "mate" else f"wins {c['gain']} pawns of material"
                notes.append(f"Likely overlooks {c['move']}, which {what} (finds it {c['p_find']:.0%}){why}.")
            else:
                notes.append(f"Should spot {c['move']} ({c['p_find']:.0%}).")
        if p_drop >= 0.25:
            notes.append(f"{p_drop:.0%} chance their next move drops material.")

        perceived = None
        if pa.cp is not None:
            obj = float(pa.cp.max())
            expected = float((probs * pa.cp).sum())
            perceived = {
                "objective_cp": round(obj),
                "expected_after_their_move_cp": round(expected),
                "expected_loss_cp": round(obj - expected),
            }

        return {
            "threats": threats,
            "chances": chances,
            "risky_moves": risky,
            "p_drops_material": round(p_drop, 3),
            "attention": attention,
            "plans": plan_rows,
            "evaluation": perceived,
            "notes": notes,
            "phase": pa.phase,
        }

    # ---------------------------------------------------------------- API
    def predict(self, board: chess.Board, top_k: int = 5, pa: PositionAnalysis | None = None) -> dict:
        pa = pa or self.analyse(board)
        if not pa.moves:
            return {"moves": [], "perspective": None, "game_over": True}
        probs = self.distribution(pa)
        book = self.book_counts(pa)
        order = np.argsort(-probs)
        moves = []
        for i in order[:top_k]:
            m = pa.moves[i]
            moves.append(
                {
                    "uci": m.uci(),
                    "san": pa.sans[i],
                    "p": round(float(probs[i]), 4),
                    "book": book.get(m.uci(), 0),
                    "reasons": self.reasons(pa, i),
                    "eval_cp": int(pa.cp[i]) if pa.cp is not None else None,
                }
            )
        engine_best = None
        if pa.cp is not None:
            b = int(np.argmax(pa.cp))
            rank = int((probs > probs[b]).sum()) + 1
            engine_best = {"uci": pa.moves[b].uci(), "san": pa.sans[b], "p": round(float(probs[b]), 4), "rank": rank, "eval_cp": int(pa.cp[b])}
        return {
            "player": self.profile.name,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "moves": moves,
            "all": {pa.moves[i].uci(): round(float(probs[i]), 4) for i in order},
            "in_book": bool(book),
            "engine_best": engine_best,
            "perspective": self.perspective(pa, probs),
        }
