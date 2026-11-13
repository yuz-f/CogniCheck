"""Lockbox: a final test on real players the project has never seen.

Fresh Lichess blitz players (none of them in the cohort), stratified by rating. Each
player's 30 most recent games are the test set; the model is trained on their older
games, with the population prior and head learned from the cohort only. Nothing about
these players may influence any design decision, so `evaluate` runs once: it writes a
seal file and refuses to run again unless the code under test is explicitly re-sealed.

Stages: select, fetch, analyse, columns, evaluate.
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cognicheck import cohort  # noqa: E402

ROOT = cohort.REAL / "lockbox"
GAMES, ANALYSIS, PLAYERS = ROOT / "games", ROOT / "analysis", ROOT / "players.json"
SEAL = ROOT / "SEALED.json"
PER_BIN, N_GAMES, TEST_GAMES = 3, 200, 30


def _patch():
    cohort.GAMES, cohort.ANALYSIS = GAMES, ANALYSIS


def players() -> list[dict]:
    return json.loads(PLAYERS.read_text())


def select():
    if PLAYERS.exists():
        print("lockbox players already selected", file=sys.stderr)
        return
    seen = {p["id"] for p in json.loads(cohort.PLAYERS.read_text())}
    orig = cohort.PLAYERS
    ROOT.mkdir(parents=True, exist_ok=True)
    cohort.PLAYERS = ROOT / "candidates.json"
    try:
        cands = cohort.select_players(per_bin=PER_BIN + 3, min_games=400, seed=2026)
    finally:
        cohort.PLAYERS = orig
    out, per = [], {}
    for p in cands:
        if p["id"] in seen:
            continue
        b = next(i for i, (lo, hi) in enumerate(cohort.RATING_BINS) if lo <= p["rating"] < hi)
        if per.get(b, 0) < PER_BIN:
            out.append(p)
            per[b] = per.get(b, 0) + 1
    PLAYERS.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} fresh lockbox players", file=sys.stderr)


def fetch():
    GAMES.mkdir(parents=True, exist_ok=True)
    for p in players():
        out = GAMES / f"{p['id']}.pgn"
        if out.exists() and out.stat().st_size > 1000:
            continue
        url = f"{cohort.LICHESS}/api/games/user/{p['id']}?max={N_GAMES}&rated=true&perfType=blitz&clocks=true&evals=false&opening=false"
        for attempt in range(5):
            try:
                text = cohort._get(url, "application/x-chess-pgn", timeout=600)
                out.write_text(text)
                print(f"  fetched {p['username']}: {text.count('[Event ')} games", file=sys.stderr)
                break
            except Exception as e:
                print(f"  retry {attempt + 1} for {p['username']}: {e}", file=sys.stderr)
                time.sleep(20 * (attempt + 1))
        time.sleep(1)


def _analyse(pid, username):
    _patch()
    return cohort._analyse_player(pid, username, 5)


def _maia(pid, username):
    _patch()
    return cohort._maia_player(pid, username)


def _derived(pid, username):
    _patch()
    return cohort._memory_player(pid, username)


def stage(fn, label, workers=3):
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, p["id"], p["username"]): p for p in players()}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                f.result()
                print(f"  {label} {p['username']} ({i}/{len(futs)}, {time.time() - t0:.0f}s)", file=sys.stderr)
            except Exception as e:
                print(f"  FAILED {label} {p['username']}: {e}", file=sys.stderr)


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted(Path("cognicheck").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def evaluate(force: bool = False):
    from evaluate import evaluate_player, summarise  # scripts/evaluate.py: the shared harness

    if SEAL.exists() and not force:
        seal = json.loads(SEAL.read_text())
        print(f"Lockbox already opened on {seal['date']} for code {seal['code']}. Results:", file=sys.stderr)
        print(json.dumps(seal["summary"], indent=1))
        return
    _patch()
    rows = []
    for p in players():
        if (ANALYSIS / f"{p['id']}.npz").exists():
            rows.append(evaluate_player(p, test_games=TEST_GAMES))
            print(f"  evaluated {p['username']}", file=sys.stderr)
    summary = summarise(rows)
    SEAL.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "code": code_fingerprint(), "players": len(rows), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    st = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if st in ("select", "prepare"):
        select()
    if st in ("fetch", "prepare"):
        fetch()
    if st in ("analyse", "prepare"):
        stage(_analyse, "analysed")
    if st in ("columns", "prepare"):
        stage(_maia, "maia", workers=2)
        stage(_derived, "derived")
    if st == "evaluate":
        evaluate(force="--force" in sys.argv)
