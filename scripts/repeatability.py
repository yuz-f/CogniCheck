"""Noise ceiling: how often a player repeats their own earlier move in the exact same position.

Leave-one-out: each repeated decision is predicted by the modal move of the player's other
visits to that position (ties share the credit). No model can be more accurate than the
player is consistent, so this bounds top-1 from above on positions that repeat.

Reported by game phase (the same phase labels as scripts/evaluate.py) and separately for two
halves of the players, to check that the ceiling is stable. Writes data/real/ceiling.json.

Caveats: late positions that repeat are mostly simple endgames, and few repeats per position
bias the estimate slightly low.
"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from cognicheck import cohort  # noqa: E402
from cognicheck.features import phase  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402
from cognicheck.profile import position_key  # noqa: E402

PHASES = ["opening", "middlegame", "endgame"]


def player_counts(p):
    seen = defaultdict(list)
    for pg in games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"]):
        b = pg.game.board()
        for mv in pg.game.mainline_moves():
            if b.turn == pg.color and b.legal_moves.count() > 1:
                seen[position_key(b)].append((phase(b), mv.uci()))
            b.push(mv)
    hit, tot = Counter(), Counter()
    for occ in seen.values():
        if len(occ) < 2:
            continue
        c = Counter(m for _, m in occ)
        for ph, m in occ:
            c[m] -= 1
            best = max(c.values())
            if best > 0:
                pred = [x for x, v in c.items() if v == best]
                tot[ph] += 1
                hit[ph] += (m in pred) / len(pred)
            c[m] += 1
    return hit, tot


def summary(rows):
    out = {}
    for ph in PHASES + ["all"]:
        h = sum(r[0][ph] if ph != "all" else sum(r[0].values()) for r in rows)
        t = sum(r[1][ph] if ph != "all" else sum(r[1].values()) for r in rows)
        out[ph] = {"decisions": int(t), "ceiling": round(h / t, 4) if t else None}
    return out


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.GAMES / f"{p['id']}.pgn").exists()]
    rows = [player_counts(p) for p in players]
    res = {"players": len(rows), "all": summary(rows), "half_a": summary(rows[0::2]), "half_b": summary(rows[1::2])}
    (cohort.REAL / "ceiling.json").write_text(json.dumps(res, indent=1))
    print(f"{len(rows)} players. Share of repeated decisions where the player repeated their own modal move.\n")
    print(f"{'phase':12s} {'decisions':>10s} {'ceiling':>8s} {'half A':>8s} {'half B':>8s}")
    for ph in PHASES + ["all"]:
        a, ha, hb = res["all"][ph], res["half_a"][ph], res["half_b"][ph]
        print(f"{ph:12s} {a['decisions']:10d} {a['ceiling'] or 0:8.3f} {ha['ceiling'] or 0:8.3f} {hb['ceiling'] or 0:8.3f}")
