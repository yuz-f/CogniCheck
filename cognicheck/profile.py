"""Build a player profile: opening book, personal move model, blind spots.

Blind spots are measured three independent ways, each relative to the
player's *own* baseline so a weak player is not just "blind to everything":

* threats     - an opponent capture threat is on the board; did they react?
                Grouped by the motif of the threatening move.
* walk-ins    - after their move, can the opponent win material? Compared to
                how often a random legal move would have allowed the same
                motif, i.e. "how well do you avoid knight tactics vs. others".
* missed wins - they could win material / mate; did they take it?
                Grouped by the motif of the winning move.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from . import model, tactics
from .engine import Engine
from .features import ALLOW_MOTIFS, FEATURE_LABEL, FEATURE_NAMES, IDX, PositionAnalysis, addresses_threat, analyse_position, build_habits, fill_time_interactions, habit_values, plan_of
from .ingest import PlayerGame

BOOK_PLIES = 24
MIN_EVENTS = 10


def position_key(board: chess.Board) -> str:
    """FEN without move counters, used for the opening book."""
    return " ".join(board.fen().split(" ")[:4])


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "player"


@dataclass
class Decision:
    """One move the player actually had to choose."""

    pa: PositionAnalysis
    chosen: int
    game_index: int
    ply: int


def base_time(time_control: str | None) -> float | None:
    """'180+2' -> 180."""
    if not time_control:
        return None
    try:
        return float(time_control.split("+")[0])
    except ValueError:
        return None


def increment(time_control: str | None) -> float:
    try:
        return float(time_control.split("+")[1]) if time_control and "+" in time_control else 0.0
    except ValueError:
        return 0.0


def _elo(headers, side: str) -> int | None:
    try:
        return int(headers.get(f"{side}Elo", ""))
    except ValueError:
        return None


def iter_decisions(pg: PlayerGame, game_index: int, engine: Engine | None, max_plies: int = 200, neural=None):
    board = pg.game.board()
    h = pg.game.headers
    elo_self = _elo(h, "White" if pg.color == chess.WHITE else "Black")
    elo_oppo = _elo(h, "Black" if pg.color == chess.WHITE else "White")
    tc = pg.game.headers.get("TimeControl")
    base, inc = base_time(tc), increment(tc)
    my_clock = None  # seconds left after the player's previous move
    prev_plan = None  # plan of the player's previous move
    for ply, node in enumerate(pg.game.mainline()):
        move = node.move
        if ply >= max_plies:
            break
        if board.turn == pg.color and board.legal_moves.count() > 1:
            # Time spent on this move (known after the fact; in live mode it is the elapsed time).
            c_after = node.clock()
            think = (my_clock + inc - c_after) if (my_clock is not None and c_after is not None) else None
            pa = analyse_position(board, engine, time_left=my_clock, base_time=base, think_time=think, prev_plan=prev_plan, neural=neural, elo_self=elo_self, elo_oppo=elo_oppo)
            try:
                idx = pa.index(move)
            except ValueError:
                break
            prev_plan = plan_of(pa.X[idx], pa.phase, pa.threat_gain)
            yield Decision(pa, idx, game_index, ply)
        if board.turn == pg.color:
            c = node.clock()
            if c is not None:
                my_clock = c
        board.push(move)


def fill_habits(decisions: list["Decision"], table: dict, base_rate: float, loo: bool) -> None:
    """Write the habit feature into each decision's feature matrix."""
    for d in decisions:
        d.pa.X[:, IDX["habit"]] = habit_values(d.pa.codes, table, base_rate, d.chosen if loo else None)
        fill_time_interactions(d.pa.X, d.pa.quick, d.pa.long)


class _Tally:
    def __init__(self):
        self.n = defaultdict(float)
        self.miss = defaultdict(float)
        self.examples: dict[str, list] = defaultdict(list)

    def add(self, tags, missed: bool, weight: float = 1.0, example=None):
        for t in tags | {"_all"}:
            self.n[t] += weight
            self.miss[t] += weight if missed else 0.0
            if missed and example is not None and len(self.examples[t]) < 3:
                self.examples[t].append(example)


def _example(d: Decision, note: str) -> dict:
    return {
        "fen": d.pa.board.fen(),
        "played": d.pa.sans[d.chosen],
        "game": d.game_index,
        "move_number": d.pa.board.fullmove_number,
        "note": note,
    }


def blind_spot_stats(decisions: list[Decision]) -> dict:
    threats, wins = _Tally(), _Tally()
    walk_actual = defaultdict(float)
    walk_expected = defaultdict(float)
    mate_threat = [0, 0]

    for d in decisions:
        pa = d.pa
        board = pa.board
        # --- existing threats -------------------------------------------
        if pa.threats:
            t = pa.threats[0]
            board.push(chess.Move.null())
            tags = tactics.motifs(board, t.move)
            san = board.san(t.move)
            board.pop()
            missed = not addresses_threat(pa, d.chosen)
            if pa.has_engine and missed:
                missed = float(pa.X[d.chosen, IDX["eng_loss"]]) >= 1.0
            threats.add(tags, missed, example=_example(d, f"left {san} on the board"))
        if pa.mate_threats:
            mate_threat[0] += 1
            board.push(pa.moves[d.chosen])
            if not board.is_checkmate() and any(m in board.legal_moves for m in pa.mate_threats):
                board.push(next(m for m in pa.mate_threats if m in board.legal_moves))
                if board.is_checkmate():
                    mate_threat[1] += 1
                board.pop()
            board.pop()

        # --- walk-ins ---------------------------------------------------
        n = len(pa.moves)
        bad = pa.after_gain >= 2
        if bad.any() and not bad.all():
            for i in np.flatnonzero(bad):
                board.push(pa.moves[i])
                tags = tactics.motifs(board, pa.after_capture[i])
                board.pop()
                for t in tags | {"_all"}:
                    walk_expected[t] += 1.0 / n
                    if i == d.chosen:
                        walk_actual[t] += 1.0
        # --- missed wins ------------------------------------------------
        best_win = None
        if pa.mates:
            best_win = pa.mates[0]
        elif pa.opportunities:
            best_win = pa.opportunities[0].move
        if best_win is not None:
            chosen_move = pa.moves[d.chosen]
            winning = set(pa.mates) | {c.move for c in pa.opportunities}
            missed = chosen_move not in winning
            if missed and pa.has_engine:
                missed = float(pa.X[d.chosen, IDX["eng_loss"]]) >= 1.0
            tags = tactics.motifs(board, best_win)
            if pa.mates:
                tags.add("mate")
            wins.add(tags, missed, example=_example(d, f"missed {board.san(best_win)}"))

    def summarise(t: _Tally, kind: str, label: str):
        base_n, base_m = t.n["_all"], t.miss["_all"]
        base = (base_m + 1) / (base_n + 2)
        out = []
        for tag in tactics.MOTIFS + ["mate"]:
            n = t.n.get(tag, 0)
            if n < MIN_EVENTS:
                continue
            k = 12.0  # shrink small categories towards the player's average
            rate = (t.miss[tag] + k * base) / (n + k)
            out.append(
                {
                    "kind": kind,
                    "motif": tag,
                    "label": label.format(tactics.MOTIF_LABEL.get(tag, "mates")),
                    "events": int(n),
                    "missed": int(t.miss[tag]),
                    "miss_rate": round(t.miss[tag] / n, 3),
                    "index": round(rate / base, 3) if base > 0 else 1.0,
                    "examples": t.examples.get(tag, []),
                }
            )
        return out, {"events": int(base_n), "miss_rate": round(base_m / base_n, 3) if base_n else 0.0}

    threat_rows, threat_base = summarise(threats, "threat", "Overlooks threats from {}")
    win_rows, win_base = summarise(wins, "opportunity", "Misses winning {}")
    walk_rows = []
    base_ratio = (walk_actual["_all"] + 1) / (walk_expected["_all"] + 1)
    for tag in tactics.MOTIFS:
        e = walk_expected.get(tag, 0)
        if e < 1.5:
            continue
        k = 3.0
        ratio = (walk_actual[tag] + k * base_ratio) / (e + k)
        walk_rows.append(
            {
                "kind": "walk_in",
                "motif": tag,
                "label": f"Walks into {tactics.MOTIF_LABEL[tag]}",
                "events": round(e, 1),
                "missed": int(walk_actual[tag]),
                "miss_rate": round(walk_actual[tag] / e, 3),
                "index": round(ratio / base_ratio, 3),
                "examples": [],
            }
        )
    rows = threat_rows + win_rows + walk_rows
    # Combine the three views into one score per motif (weighted by evidence).
    combined: dict[str, list] = defaultdict(list)
    for r in rows:
        combined[r["motif"]].append((r["index"], r["events"]))
    motif_index = {
        m: round(float(np.exp(sum(np.log(max(i, 1e-3)) * min(e, 40) for i, e in v) / sum(min(e, 40) for _, e in v))), 3)
        for m, v in combined.items()
    }
    return {
        "rows": sorted(rows, key=lambda r: -r["index"]),
        "motif_index": motif_index,
        "threat_baseline": threat_base,
        "opportunity_baseline": win_base,
        "walk_in_baseline": round(base_ratio, 3),
        "mate_threats": {"events": mate_threat[0], "missed": mate_threat[1]},
    }


def engine_refutations(decisions: list[Decision]) -> dict:
    """When the player blunders, which kind of move punishes them?"""
    blunder = defaultdict(int)
    allm = defaultdict(int)
    nb = na = 0
    for d in decisions:
        pa = d.pa
        if not pa.has_engine:
            continue
        reply = pa.replies[d.chosen]
        if reply is None:
            continue
        b = pa.board
        b.push(pa.moves[d.chosen])
        if reply not in b.legal_moves:
            b.pop()
            continue
        tags = tactics.motifs(b, reply)
        b.pop()
        na += 1
        for t in tags:
            allm[t] += 1
        if pa.X[d.chosen, IDX["eng_loss"]] >= 2.0:
            nb += 1
            for t in tags:
                blunder[t] += 1
    if nb < 3:
        return {"blunders": nb, "rows": []}
    rows = []
    for t in tactics.MOTIFS:
        if allm[t] < 5:
            continue
        pb = (blunder[t] + 0.5) / (nb + 1)
        pa_ = (allm[t] + 0.5) / (na + 1)
        rows.append({"motif": t, "label": tactics.MOTIF_LABEL[t], "share_in_blunders": round(blunder[t] / nb, 3), "share_overall": round(allm[t] / na, 3), "lift": round(pb / pa_, 3)})
    rows.sort(key=lambda r: -r["lift"])
    return {"blunders": nb, "rows": rows}


def style_summary(weights: np.ndarray, prior: np.ndarray, se: np.ndarray | None = None) -> list[dict]:
    """Deviations from the population as multiplicative odds, with uncertainty.

    ``reliable`` means the deviation is at least two standard errors from zero;
    anything else may be noise from too few relevant positions.
    """
    from .features import POSITION_FEATURES

    out = []
    for i, f in enumerate(FEATURE_NAMES):
        if f in POSITION_FEATURES:
            continue  # constant within a position: carries no weight of its own
        dlt = float(weights[i] - prior[i])
        s_ = float(se[i]) if se is not None and np.isfinite(se[i]) else None
        z = dlt / s_ if s_ else 0.0
        out.append(
            {
                "feature": f,
                "label": FEATURE_LABEL[f],
                "weight": round(float(weights[i]), 3),
                "delta": round(dlt, 3),
                "odds": round(float(np.exp(dlt)), 3),
                "se": round(s_, 3) if s_ is not None else None,
                "z": round(z, 2),
                "reliable": bool(abs(z) >= 2),
            }
        )
    out.sort(key=lambda r: (-r["reliable"], -abs(r["z"]) if r["se"] else -abs(r["delta"])))
    return out


def learned_blind_spots(style: list[dict]) -> list[dict]:
    """Blind spots as the model learned them: the 'allows a ... tactic' weights.

    Odds > 1 means this player allows that kind of tactic more readily than
    the average player (after accounting for how much material it costs).
    """
    rows = [dict(r, motif=r["feature"].replace("allows_", "")) for r in style if r["feature"].startswith("allows_")]
    # Refutation motifs the player fails to see (needs engine columns; zero weight otherwise).
    rows += [dict(r, motif="ref_" + r["feature"].replace("ref_", ""), label=r["label"].replace("loses to", "overlooks refutations by")) for r in style if r["feature"].startswith("ref_") and r["delta"] != 0]
    rows.sort(key=lambda r: -r["delta"])
    return rows


def time_pressure_stats(decisions: list[Decision]) -> dict | None:
    tp = [d for d in decisions if d.pa.time_pressure]
    normal = [d for d in decisions if not d.pa.time_pressure]
    if len(tp) < 20 or not normal or not decisions[0].pa.has_engine:
        return None
    loss = lambda ds: np.array([float(d.pa.X[d.chosen, IDX["eng_loss"]]) for d in ds])
    lt, ln = loss(tp), loss(normal)
    return {
        "decisions": len(tp),
        "share": round(len(tp) / len(decisions), 3),
        "blunder_rate": round(float((lt >= 2).mean()), 3),
        "blunder_rate_normal": round(float((ln >= 2).mean()), 3),
        "avg_loss": round(float(lt.mean()), 3),
        "avg_loss_normal": round(float(ln.mean()), 3),
    }


@dataclass
class Profile:
    id: str
    name: str
    weights: np.ndarray
    prior: np.ndarray
    book: dict[str, dict[str, int]]
    blind_spots: dict
    refutations: dict
    style: list[dict]
    evaluation: dict
    heldout: list[str]
    stats: dict
    engine: bool = False
    created: float = field(default_factory=time.time)
    habits: dict = field(default_factory=dict)  # {"table": {code: [chosen, available]}, "base_rate": r}
    head: dict | None = None  # non-linear residual (see head.py), plain arrays

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "feature_names": FEATURE_NAMES,
            "weights": [float(x) for x in self.weights],
            "prior": [float(x) for x in self.prior],
            "book": self.book,
            "blind_spots": self.blind_spots,
            "refutations": self.refutations,
            "style": self.style,
            "evaluation": self.evaluation,
            "heldout": self.heldout,
            "stats": self.stats,
            "engine": self.engine,
            "created": self.created,
            "habits": {"table": {str(k): v for k, v in self.habits.get("table", {}).items()}, "base_rate": self.habits.get("base_rate", 0.0)},
            "head": self.head,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Profile":
        names = d["feature_names"]

        def align(vec):
            m = dict(zip(names, vec))
            return np.array([m.get(f, 0.0) for f in FEATURE_NAMES])

        return cls(
            id=d["id"],
            name=d["name"],
            weights=align(d["weights"]),
            prior=align(d["prior"]),
            book=d.get("book", {}),
            blind_spots=d.get("blind_spots", {}),
            refutations=d.get("refutations", {}),
            style=d.get("style", []),
            evaluation=d.get("evaluation", {}),
            heldout=d.get("heldout", []),
            stats=d.get("stats", {}),
            engine=d.get("engine", False),
            created=d.get("created", 0),
            habits={"table": {int(k): v for k, v in (d.get("habits") or {}).get("table", {}).items()}, "base_rate": (d.get("habits") or {}).get("base_rate", 0.0)},
            head=d.get("head"),
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.id}.json"
        p.write_text(json.dumps(self.to_json()))
        return p

    @classmethod
    def load(cls, path: Path) -> "Profile":
        return cls.from_json(json.loads(Path(path).read_text()))

    def summary(self) -> dict:
        d = self.to_json()
        for k in ("book", "heldout", "weights", "prior", "feature_names", "habits", "head"):
            d.pop(k)
        d["has_head"] = self.head is not None
        d["heldout_games"] = len(self.heldout)
        d["book_positions"] = len(self.book)
        return d


def build_book(games: list[PlayerGame]) -> dict[str, dict[str, int]]:
    book: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pg in games:
        b = pg.game.board()
        for ply, mv in enumerate(pg.game.mainline_moves()):
            if ply >= BOOK_PLIES:
                break
            if b.turn == pg.color:
                book[position_key(b)][mv.uci()] += 1
            b.push(mv)
    return {k: dict(v) for k, v in book.items()}


def load_prior(path: Path | None) -> np.ndarray:
    from .population import DEFAULT_PRIOR

    if path and Path(path).exists():
        d = json.loads(Path(path).read_text())
        m = dict(zip(d["feature_names"], d["weights"]))
        return np.array([m.get(f, 0.0) for f in FEATURE_NAMES])
    return np.array([DEFAULT_PRIOR.get(f, 0.0) for f in FEATURE_NAMES])


def _fit_head(ds_train, game_train, w):
    """Non-linear residual for a single-player build, when a population head exists."""
    try:
        from .cohort import fit_head

        return fit_head(ds_train, game_train, w)
    except Exception:
        return None


def build_profile(
    name: str,
    games: list[PlayerGame],
    engine: Engine | None = None,
    prior: np.ndarray | None = None,
    holdout_frac: float = 0.15,
    lam: float = 0.01,
    seed: int = 7,
    progress=None,
    neural=None,
) -> Profile:
    if not games:
        raise ValueError(f"no games found for {name!r}")
    prior = load_prior(None) if prior is None else prior
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    n_hold = max(1, int(round(len(games) * holdout_frac))) if len(games) >= 4 else 0
    hold_idx = set(order[:n_hold])
    train_games = [g for i, g in enumerate(games) if i not in hold_idx]
    hold_games = [g for i, g in enumerate(games) if i in hold_idx]

    train: list[Decision] = []
    test: list[Decision] = []
    for gi, pg in enumerate(games):
        target = test if gi in hold_idx else train
        target.extend(iter_decisions(pg, gi, engine, neural=neural))
        if progress:
            progress(gi + 1, len(games))

    table, base_rate = build_habits((d.pa.codes, d.chosen) for d in train)
    fill_habits(train, table, base_rate, loo=True)
    fill_habits(test, table, base_rate, loo=False)
    ds_train = model.Dataset.from_positions([(d.pa.X, d.chosen) for d in train])
    ds_test = model.Dataset.from_positions([(d.pa.X, d.chosen) for d in test])
    w = model.fit(ds_train, prior=prior, lam=lam)
    book = build_book(train_games)
    game_train = np.array([d.game_index for d in train if len(d.pa.moves) > 1])
    head_eval = _fit_head(ds_train, game_train, w)

    evaluation = {}
    if len(ds_test):
        from .predictor import Predictor

        personal = model.evaluate(ds_test, w)
        population = model.evaluate(ds_test, prior)
        if head_eval is not None:
            evaluation_head = model.evaluate(ds_test, None, scores=ds_test.X @ w + head_eval.residual(ds_test.X))
        tmp = Profile(slugify(name), name, w, prior, book, {}, {}, [], {}, [], {}, engine is not None, habits={"table": table, "base_rate": base_rate}, head=head_eval.to_json() if head_eval is not None else None)
        pred = Predictor(tmp)
        hits1 = hits3 = 0
        for d in test:
            probs = pred.distribution(d.pa)
            order_ = np.argsort(-probs)
            hits1 += order_[0] == d.chosen
            hits3 += d.chosen in order_[:3]
        personal_book = {"positions": len(test), "top1": hits1 / len(test), "top3": hits3 / len(test)}
        evaluation = {"personal": personal, "personal_with_book": personal_book, "population": population}
        if head_eval is not None:
            evaluation["personal_head"] = evaluation_head
        if engine is not None:
            hits = sum(int(np.argmax(d.pa.cp)) == d.chosen for d in test if d.pa.cp is not None)
            evaluation["engine_top_move"] = {"positions": len(test), "top1": hits / len(test)}
        evaluation["random"] = {"top1": float(np.mean([1 / len(d.pa.moves) for d in test]))}

    all_dec = train + test
    # Final weights use every game; habits from every game too.
    table_all, base_all = build_habits((d.pa.codes, d.chosen) for d in all_dec)
    fill_habits(all_dec, table_all, base_all, loo=True)
    ds_all = model.Dataset.from_positions([(d.pa.X, d.chosen) for d in all_dec])
    w_all = model.fit(ds_all, prior=prior, lam=lam)
    head_all = _fit_head(ds_all, np.array([d.game_index for d in all_dec if len(d.pa.moves) > 1]), w_all)
    se_all = model.standard_errors(w_all, ds_all, lam)
    style = style_summary(w_all, prior, se_all)
    blind = blind_spot_stats(all_dec)
    blind["learned"] = learned_blind_spots(style)
    stats = {
        "games": len(games),
        "train_games": len(train_games),
        "decisions": len(all_dec),
        "white_games": sum(g.color == chess.WHITE for g in games),
        "black_games": sum(g.color == chess.BLACK for g in games),
    }
    if engine is not None and all_dec:
        losses = np.array([float(d.pa.X[d.chosen, IDX["eng_loss"]]) for d in all_dec])
        stats["avg_loss_pawns"] = round(float(losses.mean()), 3)
        stats["blunder_rate"] = round(float((losses >= 2).mean()), 3)
        stats["engine_match"] = round(float((losses <= 0.1).mean()), 3)
        tps = time_pressure_stats(all_dec)
        if tps:
            stats["time_pressure"] = tps
    if all_dec:
        from .capacity import capacity_stats

        cap = capacity_stats(ds_all.X, ds_all.starts, ds_all.chosen, np.array([d.pa.load for d in all_dec if len(d.pa.moves) > 1]), np.array([d.pa.time_pressure for d in all_dec if len(d.pa.moves) > 1]))
        if cap:
            stats["capacity"] = cap

    return Profile(
        id=slugify(name),
        name=name,
        weights=w_all,
        prior=prior,
        book=book,
        blind_spots=blind,
        refutations=engine_refutations(all_dec),
        style=style,
        evaluation=evaluation,
        heldout=[g.pgn() for g in hold_games],
        stats=stats,
        engine=engine is not None,
        habits={"table": table_all, "base_rate": base_all},
        head=head_all.to_json() if head_all is not None else None,
    )
