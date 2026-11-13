"""Plan step 3: clock and session context per decision, all known before the move is played.

Per decision (aligned with the rows of the player's analysis file):
  own_clock     own clock left / base time, before this move
  opp_clock     opponent's clock left / base time
  clock_lead    own_clock - opp_clock
  prev_think    own think time on the previous move / base time
  opp_think     opponent's think time on their last move / base time
  session_game  number of games this player started in the 2 hours before this game (capped at 10)
  after_loss    1 if the player's previous game in the session was lost
  move_number   ply / 100

Writes {pid}.clock.npz into the analysis folder. No engine, no network.

  python scripts/clock_columns.py            # cohort
"""
import json
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort  # noqa: E402
from cognicheck.profile import base_time, increment  # noqa: E402
from cognicheck.ingest import games_of, parse_pgn  # noqa: E402

NAMES = ["own_clock", "opp_clock", "clock_lead", "prev_think", "opp_think", "session_game", "after_loss", "move_number"]
SESSION_SECONDS = 2 * 3600


def _start(h):
    try:
        return datetime.strptime(h.get("UTCDate", "") + " " + h.get("UTCTime", ""), "%Y.%m.%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def _lost(pg):
    r = pg.game.headers.get("Result", "*")
    return (r == "0-1") if pg.color else (r == "1-0")


def player_columns(p):
    pgs = games_of(parse_pgn((cohort.GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    starts = [_start(pg.game.headers) for pg in pgs]
    rows = []
    for gi, pg in enumerate(pgs):
        h = pg.game.headers
        base, inc = base_time(h.get("TimeControl")), increment(h.get("TimeControl"))
        base = base or 180.0
        t0 = starts[gi]
        # games are newest first: earlier games in the session have larger indices
        session, after_loss = 0, 0.0
        if t0 is not None:
            prev = [j for j in range(gi + 1, len(pgs)) if starts[j] is not None and 0 < t0 - starts[j] <= SESSION_SECONDS]
            session = min(len(prev), 10)
            if prev:
                after_loss = float(_lost(pgs[min(prev)]))
        board = pg.game.board()
        clocks = {True: None, False: None}
        last_think = {True: 0.0, False: 0.0}
        for ply, node in enumerate(pg.game.mainline()):
            if ply >= 200:
                break
            mover = board.turn
            if mover == pg.color and board.legal_moves.count() > 1:
                own = clocks[pg.color] if clocks[pg.color] is not None else base
                opp = clocks[not pg.color] if clocks[not pg.color] is not None else base
                rows.append([own / base, opp / base, (own - opp) / base, last_think[pg.color] / base,
                             last_think[not pg.color] / base, session, after_loss, ply / 100.0])
            c = node.clock()
            if c is not None:
                before = clocks[mover] if clocks[mover] is not None else base
                last_think[mover] = max(0.0, before + inc - c)
                clocks[mover] = c
            board.push(node.move)
    return np.array(rows, dtype=np.float32)


def run(players, analysis_dir):
    for p in players:
        out = analysis_dir / f"{p['id']}.clock.npz"
        z = np.load(analysis_dir / f"{p['id']}.npz")
        C = player_columns(p)
        if len(C) != len(z["starts"]):
            print(f"  SKIP {p['id']}: {len(C)} clock rows vs {len(z['starts'])} decisions", file=sys.stderr)
            continue
        np.savez(out, C=C, names=np.array(NAMES))
    print(f"clock columns for {len(players)} players", file=sys.stderr)


if __name__ == "__main__":
    players = [p for p in json.loads(cohort.PLAYERS.read_text()) if (cohort.ANALYSIS / f"{p['id']}.npz").exists()]
    run(players, cohort.ANALYSIS)
