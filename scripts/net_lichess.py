"""Stage B data: stream the public Lichess database and encode generic training positions.

The monthly archive (~30 GB compressed) is streamed, never stored. Kept: rated blitz games with
clock times where both ratings are in the cohort range. Dropped: every game in which a cohort or
lockbox player took part, so none of their games (test games included) can reach training.

Every move of both sides is encoded (cognicheck/chessnet.py), with the game number, ply and both
clocks, so positions of one game are consecutive rows (game history can be added later without
downloading again). Shards of SHARD positions go to data/real/net/lichess/.

  python scripts/net_lichess.py --month 2026-10 --positions 20000000 --workers 3
"""
import argparse
import io
import json
import multiprocessing
import re
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

sys.path.insert(0, ".")
from cognicheck import chessnet, cohort  # noqa: E402

OUT = cohort.REAL / "net" / "lichess"
SHARD = 1_000_000
MAX_PLIES = 200
ELO_MIN, ELO_MAX = 800, 3000
_HDR = re.compile(r'^\[(\w+) "(.*)"\]$')


def excluded_names() -> set[str]:
    names = {p["username"].lower() for p in json.loads(cohort.PLAYERS.read_text())}
    lb = cohort.REAL / "lockbox" / "players.json"
    if lb.exists():
        names |= {p["username"].lower() for p in json.loads(lb.read_text())}
    return names


def keep_headers(h: dict, names: set[str]) -> bool:
    if "Rated Blitz" not in h.get("Event", ""):
        return False
    if h.get("White", "").lower() in names or h.get("Black", "").lower() in names:
        return False
    try:
        w, b = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
    except ValueError:
        return False
    return ELO_MIN <= w <= ELO_MAX and ELO_MIN <= b <= ELO_MAX


def base_time(tc: str) -> float:
    try:
        return max(float(tc.split("+")[0]), 15.0)
    except (ValueError, AttributeError):
        return 180.0


def encode(texts: list[str]):
    """Encode a list of PGN game texts. Returns arrays for all their positions."""
    planes, elo, legal, nlegal, tgt, ply_a, clk, gid = [], [], [], [], [], [], [], []
    for g_i, text in enumerate(texts):
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        h = game.headers
        w, b = float(h.get("WhiteElo", 1500)), float(h.get("BlackElo", 1500))
        base = base_time(h.get("TimeControl", "180+0"))
        clocks = {chess.WHITE: base, chess.BLACK: base}
        board = game.board()
        for ply, node in enumerate(game.mainline()):
            if ply >= MAX_PLIES:
                break
            if board.legal_moves.count() > 1:
                moves, idx = chessnet.legal_indices(board)
                es, eo = (w, b) if board.turn == chess.WHITE else (b, w)
                planes.append(np.packbits(chessnet.planes(board)))
                elo.append(chessnet.elo_input(es, eo))
                legal.append(idx)
                nlegal.append(len(idx))
                tgt.append(moves.index(node.move))
                ply_a.append(ply)
                clk.append((clocks[board.turn] / base, clocks[not board.turn] / base))
                gid.append(g_i)
            c = node.clock()
            if c is not None:
                clocks[board.turn] = c
            board.push(node.move)
    if not tgt:
        return None
    return {"planes": np.stack(planes), "elo": np.stack(elo).astype(np.float16), "legal": np.concatenate(legal).astype(np.int16),
            "nlegal": np.array(nlegal, np.int16), "target": np.array(tgt, np.int16), "ply": np.array(ply_a, np.int16),
            "clock": np.array(clk, np.float16), "game": np.array(gid, np.int32), "games": len(texts)}


def games_from_stream(url: str, names: set[str]):
    """Yield PGN texts of kept games from the streamed archive.

    A dropped connection restarts the stream from the beginning and skips the games already
    yielded (the header filter is cheap), so the output is the same as one unbroken read."""
    import zstandard

    yielded = 0
    for attempt in range(20):
        proc = subprocess.Popen(["curl", "-sS", "--http1.1", url], stdout=subprocess.PIPE)
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(proc.stdout), encoding="utf-8", errors="replace")
        lines, headers, in_moves, seen = [], {}, False, 0
        try:
            for line in reader:
                if line.startswith("[Event ") and in_moves:
                    if keep_headers(headers, names) and "%clk" in "".join(lines[-3:]):
                        seen += 1
                        if seen > yielded:
                            yielded += 1
                            yield "".join(lines)
                    lines, headers, in_moves = [], {}, False
                lines.append(line)
                m = _HDR.match(line.strip())
                if m:
                    headers[m.group(1)] = m.group(2)
                elif line.strip():
                    in_moves = True
        except Exception as e:  # truncated stream: zstd raises on a cut-off frame
            print(f"  stream error after {yielded:,} games: {e}", file=sys.stderr)
        finally:
            rc = proc.poll()
            proc.kill()
            proc.wait()
        if rc == 0:  # curl finished the whole archive
            return
        print(f"  restarting stream (attempt {attempt + 2}), skipping {yielded:,} kept games", file=sys.stderr)
        time.sleep(10)


def write_shard(parts, k, game_base, out):
    keys = ["planes", "elo", "legal", "nlegal", "target", "ply", "clock"]
    d = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    games, off = [], game_base
    for p in parts:
        games.append(p["game"] + off)
        off += p["games"]
    d["game"] = np.concatenate(games)
    d["off"] = np.concatenate([[0], np.cumsum(d["nlegal"].astype(np.int64))])
    del d["nlegal"]
    np.savez(out / f"shard_{k:03d}.npz", **d)
    return len(d["target"]), off


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-10")
    ap.add_argument("--positions", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    SHARD, OUT = a.shard, Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{a.month}.pgn.zst"
    names = excluded_names()
    done = sorted(OUT.glob("shard_*.npz"))
    if done:
        print(f"{len(done)} shards already exist; delete them to rebuild", file=sys.stderr)
        sys.exit(0)
    t0, total, k, game_base, parts, in_shard = time.time(), 0, 0, 0, [], 0
    def chunk_iter():
        buf = []
        for text in games_from_stream(url, names):
            buf.append(text)
            if len(buf) == 200:
                yield buf
                buf = []

    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for res in pool.imap(encode, chunk_iter(), chunksize=1):
            if res is None:
                continue
            parts.append(res)
            in_shard += len(res["target"])
            if in_shard >= SHARD:
                n, game_base = write_shard(parts, k, game_base, OUT)
                total += n
                k += 1
                parts, in_shard = [], 0
                print(f"  shard {k}: {total:,} positions, {game_base:,} games ({time.time() - t0:.0f}s)", file=sys.stderr)
                if total >= a.positions:
                    pool.terminate()
                    break
        else:
            if parts:
                n, game_base = write_shard(parts, k, game_base, OUT)
                total += n
    json.dump({"month": a.month, "positions": total, "games": game_base, "elo_range": [ELO_MIN, ELO_MAX],
               "excluded_players": len(names)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"done: {total:,} positions from {game_base:,} games in {time.time() - t0:.0f}s", file=sys.stderr)
