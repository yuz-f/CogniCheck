"""Command line entry points.

    python -m cognicheck generate --persona kai --games 100
    python -m cognicheck population data/games/*.pgn
    python -m cognicheck build data/games/kai.pgn --player "Knightmare Kai"
    python -m cognicheck lichess <username>
    python -m cognicheck serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROFILES = DATA / "profiles"
GAMES = DATA / "games"
POPULATION = DATA / "population.json"


def _engine(args):
    if getattr(args, "no_engine", False):
        return None
    from .engine import Engine, find_stockfish

    if not find_stockfish():
        print("Stockfish not found - building without engine features", file=sys.stderr)
        return None
    return Engine(depth=args.depth)


def _progress(i, n):
    if i == n or i % 10 == 0:
        print(f"  analysed {i}/{n} games", file=sys.stderr)


def cmd_generate(args):
    from .personas import by_name, generate

    p = by_name(args.persona)
    games = generate(p, args.games, seed=args.seed)
    GAMES.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else GAMES / f"{p.name.split()[-1].lower()}.pgn"
    out.write_text("\n\n".join(str(g) for g in games) + "\n")
    print(f"wrote {len(games)} games to {out}")


def cmd_population(args):
    import numpy as np

    from . import model
    from .features import FEATURE_NAMES
    from .ingest import PlayerGame, parse_pgn
    from .profile import iter_decisions, load_prior

    import chess

    engine = _engine(args)
    positions = []
    n = 0
    for path in args.pgn:
        for g in parse_pgn(Path(path).read_text())[: args.max_games]:
            for color in (chess.WHITE, chess.BLACK):
                name = g.headers.get("White" if color else "Black", "")
                if args.only and name not in args.only:
                    continue
                for d in iter_decisions(PlayerGame(g, color), n, engine):
                    positions.append((d.pa.X, d.chosen))
            n += 1
            if n % 20 == 0:
                print(f"  {n} games", file=sys.stderr)
    ds = model.Dataset.from_positions(positions)
    w = model.fit(ds, prior=load_prior(None), lam=0.002)
    POPULATION.parent.mkdir(parents=True, exist_ok=True)
    POPULATION.write_text(json.dumps({"feature_names": FEATURE_NAMES, "weights": [float(x) for x in w], "positions": len(ds)}, indent=1))
    print(f"population prior from {len(ds)} positions -> {POPULATION}")
    print(json.dumps(model.evaluate(ds, w), indent=1))
    if engine:
        engine.close()


def build_from_pgn_text(text: str, player: str | None, engine=None, max_games: int | None = None, progress=None):
    from .ingest import games_of, parse_pgn, player_names
    from .profile import build_profile, load_prior

    games = parse_pgn(text)
    if player is None:
        player = next(iter(player_names(games)), None)
    pgs = games_of(games, player)
    if max_games:
        pgs = pgs[:max_games]
    prof = build_profile(player, pgs, engine=engine, prior=load_prior(POPULATION), progress=progress)
    prof.save(PROFILES)
    return prof


def cmd_build(args):
    text = "\n\n".join(Path(p).read_text() for p in args.pgn)
    engine = _engine(args)
    prof = build_from_pgn_text(text, args.player, engine, args.max_games, _progress)
    if engine:
        engine.close()
    print_profile(prof)


def cmd_lichess(args):
    from .ingest import fetch_chesscom, fetch_lichess

    text = fetch_chesscom(args.username, args.max_games) if args.chesscom else fetch_lichess(args.username, args.max_games)
    engine = _engine(args)
    prof = build_from_pgn_text(text, args.username, engine, None, _progress)
    if engine:
        engine.close()
    print_profile(prof)


def print_profile(prof):
    print(f"\n== {prof.name}  ({prof.stats.get('games')} games, {prof.stats.get('decisions')} decisions)")
    ev = prof.evaluation
    if ev:
        print("Held-out prediction accuracy:")
        for k in ("personal_with_book", "personal", "population", "engine_top_move", "random"):
            if k in ev:
                v = ev[k]
                extra = f"  top3 {v['top3']:.1%}" if "top3" in v else ""
                print(f"  {k:20s} top1 {v['top1']:.1%}{extra}")
    print("Style (vs. population):")
    for r in prof.style[:8]:
        print(f"  {r['label']:45s} x{r['odds']:.2f}")
    print("Blind spots (index > 1 = worse than own average):")
    for r in prof.blind_spots.get("rows", [])[:8]:
        print(f"  {r['label']:55s} index {r['index']:.2f}  ({r['missed']}/{r['events']})")
    print("Combined motif index:", json.dumps(dict(sorted(prof.blind_spots.get("motif_index", {}).items(), key=lambda kv: -kv[1])[:6])))
    if prof.refutations.get("rows"):
        print(f"Engine: what punishes the blunders ({prof.refutations['blunders']} blunders):")
        for r in prof.refutations["rows"][:5]:
            print(f"  {r['label']:40s} lift {r['lift']:.2f}")


def cmd_cohort(args):
    from . import cohort

    if args.stage in ("select", "all"):
        cohort.select_players(per_bin=args.per_bin, min_games=args.min_games)
    if args.stage in ("fetch", "all"):
        cohort.fetch_games(args.games)
    if args.stage in ("analyse", "all"):
        cohort.analyse_all(workers=args.workers, depth=args.depth)
    if args.stage in ("fit", "all"):
        cohort.fit_all()
    if args.stage in ("pca", "all"):
        cohort.print_report(cohort.run_pca())


def cmd_add(args):
    from . import cohort

    res = cohort.add_player(lichess=args.lichess, chesscom=args.chesscom, pgn=Path(args.pgn).read_text() if args.pgn else None, player=args.player, rating=args.rating)
    print(json.dumps(res, indent=1))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("cognicheck.server:app", host=args.host, port=args.port, reload=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cognicheck")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create synthetic games for a persona")
    g.add_argument("--persona", required=True)
    g.add_argument("--games", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")
    g.set_defaults(func=cmd_generate)

    for name, fn, hlp in (("build", cmd_build, "build a profile from PGN files"), ("population", cmd_population, "train the population prior")):
        b = sub.add_parser(name, help=hlp)
        b.add_argument("pgn", nargs="+")
        b.add_argument("--no-engine", action="store_true")
        b.add_argument("--depth", type=int, default=6)
        if name == "build":
            b.add_argument("--player")
            b.add_argument("--max-games", type=int)
        else:
            b.add_argument("--only", nargs="*", help="restrict to these player names")
            b.add_argument("--max-games", type=int, help="per PGN file")
        b.set_defaults(func=fn)

    li = sub.add_parser("lichess", help="download a player's games and build a profile")
    li.add_argument("username")
    li.add_argument("--max-games", type=int, default=200)
    li.add_argument("--chesscom", action="store_true", help="use Chess.com instead of Lichess")
    li.add_argument("--no-engine", action="store_true")
    li.add_argument("--depth", type=int, default=6)
    li.set_defaults(func=cmd_lichess)

    co = sub.add_parser("cohort", help="profile many real Lichess players and run the PCA")
    co.add_argument("stage", choices=["select", "fetch", "analyse", "fit", "pca", "all"])
    co.add_argument("--per-bin", type=int, default=9, help="players per rating band")
    co.add_argument("--min-games", type=int, default=400, help="minimum rated blitz games on Lichess")
    co.add_argument("--games", type=int, default=200, help="games to download per player")
    co.add_argument("--workers", type=int, default=4)
    co.add_argument("--depth", type=int, default=5)
    co.set_defaults(func=cmd_cohort)

    ad = sub.add_parser("add", help="add one player to the cohort: profile + place on the PCA map")
    ad.add_argument("--lichess")
    ad.add_argument("--chesscom")
    ad.add_argument("--pgn", help="PGN file")
    ad.add_argument("--player", help="name of the player in the PGN")
    ad.add_argument("--rating", type=int, help="rating, if the PGN has none")
    ad.set_defaults(func=cmd_add)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
