"""Noise ceiling: how often a player repeats their own earlier move in the exact same position.

Leave-one-out: each repeated decision is predicted by the modal move of the player's other visits
to that position. No model can be more accurate than the player is consistent.
"""
import json, sys
from collections import defaultdict, Counter
sys.path.insert(0, ".")
from cognicheck import cohort
from cognicheck.ingest import games_of, parse_pgn
from cognicheck.profile import position_key
buckets = [(0,10),(10,20),(20,30),(30,40),(40,60),(60,400)]
hit = Counter(); tot = Counter(); npos = Counter()
for p in json.loads(cohort.PLAYERS.read_text()):
    f = cohort.GAMES / f"{p['id']}.pgn"
    if not f.exists(): continue
    seen = defaultdict(list)
    for pg in games_of(parse_pgn(f.read_text()), p["username"]):
        b = pg.game.board()
        for ply, mv in enumerate(pg.game.mainline_moves()):
            if b.turn == pg.color:
                seen[position_key(b)].append((ply, mv.uci()))
            b.push(mv)
    for key, occ in seen.items():
        if len(occ) < 2: continue
        c = Counter(m for _, m in occ)
        for ply, m in occ:
            c[m] -= 1
            best = max(c.values())
            pred = [x for x, v in c.items() if v == best]
            bk = next(i for i,(lo,hi) in enumerate(buckets) if lo <= ply < hi)
            if best > 0:
                tot[bk] += 1; hit[bk] += (m in pred) / len(pred)
            c[m] += 1
        npos[next(i for i,(lo,hi) in enumerate(buckets) if lo <= occ[0][0] < hi)] += 1
for i,(lo,hi) in enumerate(buckets):
    if tot[i]: print(f"plies {lo:3d}-{hi:3d}: repeated decisions {tot[i]:7d}  same move as before {hit[i]/tot[i]:.3f}")
print("all:", sum(hit.values())/sum(tot.values()), sum(tot.values()))
