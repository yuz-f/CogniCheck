"""Predict a specific player's next move and read the position through their eyes."""
from __future__ import annotations

import chess
import numpy as np

from . import tactics
from .engine import Engine
from .features import ENGINE_FEATURES, FEATURE_LABEL, FEATURE_NAMES, IDX, PLAN_LABEL, PLANS, PositionAnalysis, analyse_position, habit_values, plan_of as _plan_of, previous_plan

# Collinear feature families are explained as one line.
REASON_GROUPS = {
    "engine": (set(ENGINE_FEATURES), "objectively sound (engine)", "objectively weak (engine)"),
    "hangs": ({"hangs", "tp_hangs"}, "leaves material en prise", "leaves material en prise"),
    "capture": ({"capture", "tp_capture"}, "captures", "captures"),
}
_GROUP_OF = {f: g for g, (fs, _, _) in REASON_GROUPS.items() for f in fs}
from .model import softmax

BOOK_ALPHA = 2.0  # pseudo-count of the model inside the opening book

class Predictor:
    def __init__(self, profile, engine: Engine | None = None, neural=None):
        self.profile = profile
        self.engine = engine
        self.neural = neural
        self.rating = (profile.stats or {}).get("rating")
        self.w = np.asarray(profile.weights, dtype=np.float64)
        self.prior = np.asarray(profile.prior, dtype=np.float64)
        self.motif_index: dict[str, float] = (profile.blind_spots or {}).get("motif_index", {})
        self._rows = {(r["kind"], r["motif"]): r for r in (profile.blind_spots or {}).get("rows", [])}
        # Blind spots the model itself learned ('allows a ... tactic' weights), odds vs. the average player.
        self.learned: dict[str, float] = {r["motif"]: r["odds"] for r in (profile.blind_spots or {}).get("learned", []) if r.get("reliable")}
        self.habits = getattr(profile, "habits", None) or {}

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

    def analyse(self, board: chess.Board, think_time: float | None = None, time_left: float | None = None) -> PositionAnalysis:
        from .features import fill_time_interactions

        pa = analyse_position(board, self.engine, time_left=time_left, think_time=think_time, prev_plan=previous_plan(board), neural=self.neural, elo_self=self.rating, elo_oppo=self.rating)
        table = self.habits.get("table")
        if table:
            pa.X[:, IDX["habit"]] = habit_values(pa.codes, table, self.habits.get("base_rate", 0.0))
        fill_time_interactions(pa.X, pa.quick, pa.long)
        return pa

    # ------------------------------------------------------------- explain
    def reasons(self, pa: PositionAnalysis, i: int, k: int = 3) -> list[dict]:
        """Why this move rather than what the model expected.

        log P(i) decomposes along features as w_j * (x_ij - E[x_j]) where E is
        the model's own expectation over the legal moves, so each term is the
        evidence that pulled the player towards move i, feature by feature.
        """
        X = pa.X.astype(np.float64)
        probs = self.model_distribution(pa)
        centred = X[i] - probs @ X
        total = self.w * centred
        personal = (self.w - self.prior) * centred
        # Sum collinear families; keep the sign of the move's own feature value.
        items: dict[str, list] = {}
        for j, f in enumerate(FEATURE_NAMES):
            g = _GROUP_OF.get(f, f)
            it = items.setdefault(g, [0.0, 0.0, 0.0])
            it[0] += float(total[j])
            it[1] += float(personal[j])
            it[2] += float(centred[j]) * (1 if f not in ENGINE_FEATURES or f in ("eng_best",) else -1 if f in ("eng_loss", "eng_loss_small", "eng_loss_mid", "eng_rank", "tp_eng_loss") else 1)
        out = []
        for g, (eff, pers, c) in sorted(items.items(), key=lambda kv: -abs(kv[1][0]))[:k]:
            if abs(eff) < 0.15:
                continue
            if g in REASON_GROUPS:
                _, pos, neg = REASON_GROUPS[g]
                label, has = (pos if eff > 0 else neg), True
                if g != "engine":
                    has = c > 0
            else:
                label, has = FEATURE_LABEL[g], c > 0
            out.append({"feature": g, "label": label, "has": bool(has), "effect": round(eff, 2), "personal": round(pers, 2)})
        return out

    def motif_note(self, tags: set[str]) -> tuple[float, str | None]:
        """Worst blind spot among the motifs: the model's learned odds when
        reliable, otherwise the counted miss-rate index. Returns (index, label)."""
        best, label = 1.0, None
        for t in tags:
            idx = self.learned.get(t) or self.motif_index.get(t)
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
            k = _plan_of(pa.X[i], pa.phase, pa.threat_gain)
            plans[k] = plans.get(k, 0.0) + float(probs[i])
        plan_rows = [{"plan": k, "label": PLAN_LABEL[k], "p": round(v, 3)} for k, v in sorted(plans.items(), key=lambda kv: -kv[1]) if v >= 0.03]

        # Narrative.
        if plan_rows:
            cont = " - continuing the previous plan" if pa.prev_plan == plan_rows[0]["plan"] else (f" - switching from '{PLAN_LABEL.get(pa.prev_plan, pa.prev_plan)}'" if pa.prev_plan else "")
            notes.append(f"Most likely intention: {plan_rows[0]['label']} ({plan_rows[0]['p']:.0%}){cont}.")
        if pa.quick:
            notes.append("Played quickly: habit and reflex weigh more than calculation.")
        elif pa.long:
            notes.append("After a long think: calculation weighs more, habits less.")
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
            "previous_plan": pa.prev_plan,
            "think": "quick" if pa.quick else "long" if pa.long else None,
        }

    # ---------------------------------------------------------------- API
    def predict(self, board: chess.Board, top_k: int = 5, pa: PositionAnalysis | None = None, think_time: float | None = None, time_left: float | None = None) -> dict:
        pa = pa or self.analyse(board, think_time=think_time, time_left=time_left)
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
