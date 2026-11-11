"""Profile a cohort of real Lichess players and map their cognition with PCA.

Pipeline (each stage caches to ``data/real/`` so it can be resumed):

1. ``select``   pick active blitz players spread across the rating range,
                from recent Lichess arena tournaments.
2. ``fetch``    download N rated blitz games per player (one request at a time).
3. ``analyse``  per player, in parallel: features of every decision + blind-spot
                statistics for all games and for two disjoint halves.
4. ``fit``      train the population prior on the pooled games, then each
                player's personal model (all games, half A, half B) and save
                full profiles for the web app.
5. ``pca``      cognition vectors -> remove rating -> standardise -> PCA,
                split-half reliability per component, k-means player types.

Raw games and per-player results stay local (``data/real`` is git-ignored):
these are real people, so only aggregate results belong in the repository.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import numpy as np

from . import model, tactics
from .features import FEATURE_LABEL, FEATURE_NAMES, IDX
from .ingest import USER_AGENT, games_of, parse_pgn

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"
GAMES = REAL / "games"
ANALYSIS = REAL / "analysis"
PROFILES_REAL = REAL / "profiles"
PLAYERS = REAL / "players.json"
POPULATION_REAL = REAL / "population.json"
PCA_OUT = REAL / "pca.json"

LICHESS = "https://lichess.org"
RATING_BINS = [(800, 1100), (1100, 1300), (1300, 1500), (1500, 1700), (1700, 1900), (1900, 2100), (2100, 2300), (2300, 2600), (2600, 3400)]


# ----------------------------------------------------------------- network
def _get(url: str, accept: str = "application/json", data: bytes | None = None, timeout: int = 120) -> str:
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Lichess asks clients to wait a full minute
                print("  rate limited, waiting 60s", file=sys.stderr)
                time.sleep(61)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url}")


def _ndjson(text: str) -> list[dict]:
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


# ------------------------------------------------------------------ select
def select_players(per_bin: int = 9, min_games: int = 400, seed: int = 1) -> list[dict]:
    """Active blitz players from recent arenas, stratified by blitz rating."""
    tours = json.loads(_get(f"{LICHESS}/api/tournament"))
    cands: dict[str, int] = {}
    for t in tours.get("finished", []):
        if t["perf"]["key"] != "blitz" or t.get("nbPlayers", 0) < 30:
            continue
        for r in _ndjson(_get(f"{LICHESS}/api/tournament/{t['id']}/results?nb=1500", "application/x-ndjson")):
            if r.get("title") == "BOT":
                continue
            cands.setdefault(r["username"].lower(), r.get("rating", 0))
        time.sleep(1)
    print(f"{len(cands)} candidate players from arenas", file=sys.stderr)

    rng = random.Random(seed)
    by_bin: dict[int, list[str]] = defaultdict(list)
    for name, rating in cands.items():
        for i, (lo, hi) in enumerate(RATING_BINS):
            if lo <= rating < hi:
                by_bin[i].append(name)
    chosen: list[dict] = []
    for i in range(len(RATING_BINS)):
        pool = by_bin[i]
        rng.shuffle(pool)
        picked = 0
        for chunk_start in range(0, len(pool), 100):
            if picked >= per_bin:
                break
            chunk = pool[chunk_start : chunk_start + 100]
            users = json.loads(_get(f"{LICHESS}/api/users", data=",".join(chunk).encode()))
            time.sleep(1)
            for u in users:
                bl = (u.get("perfs") or {}).get("blitz") or {}
                if u.get("disabled") or u.get("tosViolation") or bl.get("prov") or bl.get("games", 0) < min_games:
                    continue
                chosen.append({"id": u["id"], "username": u["username"], "rating": bl["rating"], "blitz_games": bl["games"], "title": u.get("title")})
                picked += 1
                if picked >= per_bin:
                    break
        print(f"  bin {RATING_BINS[i]}: {picked} players", file=sys.stderr)
    REAL.mkdir(parents=True, exist_ok=True)
    PLAYERS.write_text(json.dumps(chosen, indent=1))
    return chosen


# ------------------------------------------------------------------- fetch
def fetch_games(n_games: int = 60) -> None:
    players = json.loads(PLAYERS.read_text())
    GAMES.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(players):
        out = GAMES / f"{p['id']}.pgn"
        if out.exists() and out.stat().st_size > 1000:
            continue
        url = f"{LICHESS}/api/games/user/{p['id']}?max={n_games}&rated=true&perfType=blitz&clocks=true&evals=false&opening=false"
        text = _get(url, "application/x-chess-pgn", timeout=300)
        out.write_text(text)
        print(f"  [{i + 1}/{len(players)}] {p['username']}: {text.count('[Event ')} games", file=sys.stderr)
        time.sleep(1)


# ----------------------------------------------------------------- analyse
def _analyse_player(pid: str, username: str, depth: int, progress=None) -> str:
    from .engine import Engine, find_stockfish
    from .profile import blind_spot_stats, engine_refutations, iter_decisions

    out = ANALYSIS / f"{pid}.npz"
    if out.exists():
        return pid
    engine = Engine(depth=depth) if find_stockfish() else None
    try:
        games = games_of(parse_pgn((GAMES / f"{pid}.pgn").read_text()), username)
        decisions = []
        for gi, pg in enumerate(games):
            decisions.extend(iter_decisions(pg, gi, engine))
            if progress:
                progress(gi + 1, len(games))
    finally:
        if engine:
            engine.close()
    ds = model.Dataset.from_positions([(d.pa.X, d.chosen) for d in decisions])
    game_of = np.array([d.game_index for d in decisions if len(d.pa.moves) > 1])
    halves = {
        "all": decisions,
        "A": [d for d in decisions if d.game_index % 2 == 0],
        "B": [d for d in decisions if d.game_index % 2 == 1],
    }
    extra = {k: {"blind_spots": blind_spot_stats(v), "refutations": engine_refutations(v)} for k, v in halves.items()}
    extra["games"] = len(games)
    extra["white_games"] = sum(g.color == chess.WHITE for g in games)
    np.savez_compressed(out, X=ds.X.astype(np.float16), starts=ds.starts, chosen=ds.chosen, game=game_of)
    (ANALYSIS / f"{pid}.json").write_text(json.dumps(extra))
    return pid


def analyse_all(workers: int = 4, depth: int = 5) -> None:
    players = json.loads(PLAYERS.read_text())
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    todo = [p for p in players if (GAMES / f"{p['id']}.pgn").exists() and not (ANALYSIS / f"{p['id']}.npz").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_analyse_player, p["id"], p["username"], depth): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                f.result()
                print(f"  analysed {p['username']} ({i}/{len(todo)}, {time.time() - t0:.0f}s)", file=sys.stderr)
            except Exception as e:  # keep going; report at the end
                print(f"  FAILED {p['username']}: {e}", file=sys.stderr)


# --------------------------------------------------------------------- fit
def _load_ds(pid: str) -> tuple[model.Dataset, np.ndarray]:
    z = np.load(ANALYSIS / f"{pid}.npz")
    return model.Dataset(z["X"].astype(np.float64), z["starts"], z["chosen"]), z["game"]


def _subset(ds: model.Dataset, mask_pos: np.ndarray) -> model.Dataset:
    ends = np.append(ds.starts[1:], len(ds.X))
    pos = [(ds.X[s:e], int(c - s)) for s, e, c, m in zip(ds.starts, ends, ds.chosen, mask_pos) if m]
    return model.Dataset.from_positions(pos)


def _engine_stats(ds: model.Dataset) -> dict:
    if len(ds) == 0:
        return {}
    loss = ds.X[ds.chosen, IDX["eng_loss"]]
    return {"engine_match": float((loss <= 0.1).mean()), "avg_loss_pawns": float(loss.mean()), "blunder_rate": float((loss >= 2).mean())}


def train_population(players: list[dict]) -> np.ndarray:
    """Population prior: an equal number of positions from every player."""
    rng = np.random.default_rng(0)
    pooled = []
    for p in players:
        ds, _ = _load_ds(p["id"])
        idx = rng.choice(len(ds), size=min(600, len(ds)), replace=False)
        pooled.append(_subset(ds, np.isin(np.arange(len(ds)), idx)))
    big = model.Dataset.from_positions([(d.X[s:e], int(c - s)) for d in pooled for s, e, c in zip(d.starts, np.append(d.starts[1:], len(d.X)), d.chosen)])
    prior = model.fit(big, lam=1e-3)
    POPULATION_REAL.write_text(json.dumps({"feature_names": FEATURE_NAMES, "weights": [float(x) for x in prior], "positions": len(big), "players": len(players)}, indent=1))
    print(f"population prior from {len(big)} positions of {len(players)} players", file=sys.stderr)
    return prior


def load_population() -> np.ndarray:
    if not POPULATION_REAL.exists():
        raise RuntimeError("no real-player population yet - run: python -m cognicheck cohort all")
    from .profile import load_prior

    return load_prior(POPULATION_REAL)


def fit_player(p: dict, prior: np.ndarray, lam: float = 0.01, holdout_frac: float = 0.15):
    """Personal model (all games + two halves), evaluation, profile and cognition vector."""
    from .profile import Profile, build_book, slugify, style_summary

    ds, game = _load_ds(p["id"])
    extra = json.loads((ANALYSIS / f"{p['id']}.json").read_text())
    n_games = extra["games"]
    hold = set(range(n_games)[: max(1, int(round(n_games * holdout_frac)))])
    train_mask = np.array([g not in hold for g in game])
    ds_train, ds_test = _subset(ds, train_mask), _subset(ds, ~train_mask)
    w_eval = model.fit(ds_train, prior=prior, lam=lam)
    ev = {
        "personal": model.evaluate(ds_test, w_eval),
        "population": model.evaluate(ds_test, prior),
    }
    loss_test = ds_test.X[:, IDX["eng_loss"]]
    ends = np.append(ds_test.starts[1:], len(ds_test.X))
    ev["engine_top_move"] = {"top1": float(np.mean([loss_test[c] == loss_test[s:e].min() for s, e, c in zip(ds_test.starts, ends, ds_test.chosen)]))}
    ev["random"] = {"top1": float(np.mean(1.0 / np.diff(np.append(ds_test.starts, len(ds_test.X)))))}
    masks = {"all": np.ones(len(ds), bool), "A": game % 2 == 0, "B": game % 2 == 1}
    fits = {k: model.fit(_subset(ds, m), prior=prior, lam=lam) for k, m in masks.items()}
    vectors = {k: cognition_vector(fits[k], prior, extra[k]["blind_spots"], _engine_stats(_subset(ds, masks[k]))) for k in fits}
    # Full profile for the web app (book + held-out games from the PGN).
    pgs = games_of(parse_pgn((GAMES / f"{p['id']}.pgn").read_text()), p["username"])
    prof = Profile(
        id=slugify(p["id"]),
        name=p["username"],
        weights=fits["all"],
        prior=prior,
        book=build_book([g for i, g in enumerate(pgs) if i not in hold]),
        blind_spots=extra["all"]["blind_spots"],
        refutations=extra["all"]["refutations"],
        style=style_summary(fits["all"], prior),
        evaluation=ev,
        heldout=[g.pgn() for i, g in enumerate(pgs) if i in hold],
        stats={"games": n_games, "decisions": len(ds), "white_games": extra["white_games"], "black_games": n_games - extra["white_games"], "rating": p["rating"], "source": p.get("source", "lichess blitz")} | {k: round(v, 3) for k, v in _engine_stats(ds).items()},
        engine=True,
    )
    PROFILES_REAL.mkdir(parents=True, exist_ok=True)
    prof.save(PROFILES_REAL)
    (ANALYSIS / f"{p['id']}.vec.json").write_text(json.dumps({"id": prof.id, "name": p["username"], "rating": p["rating"], "added": bool(p.get("added")), "source": p.get("source", "lichess blitz"), "evaluation": ev, "vectors": vectors}))
    return prof


def fit_all(lam: float = 0.01, holdout_frac: float = 0.15) -> None:
    players = [p for p in json.loads(PLAYERS.read_text()) if (ANALYSIS / f"{p['id']}.npz").exists()]
    cohort_players = [p for p in players if not p.get("added")] or players
    prior = train_population(cohort_players)
    for p in players:
        fit_player(p, prior, lam, holdout_frac)
    print(f"fitted {len(players)} players", file=sys.stderr)


# --------------------------------------------------------------- add player
import threading as _threading

_ADD_LOCK = _threading.Lock()  # one analysis + PCA rewrite at a time
def _lichess_rating(username: str, perf: str = "blitz") -> tuple[str, str, int, int]:
    users = json.loads(_get(f"{LICHESS}/api/users", data=username.encode()))
    if not users:
        raise ValueError(f"Lichess user {username!r} not found")
    u = users[0]
    if u.get("disabled"):
        raise ValueError(f"{username} has closed their account")
    pf = (u.get("perfs") or {}).get(perf) or {}
    return u["id"], u["username"], int(pf.get("rating", 0)), int(pf.get("games", 0))


def _pgn_rating(text: str, name: str) -> int:
    ratings = []
    for g in parse_pgn(text):
        for side in ("White", "Black"):
            if g.headers.get(side, "").lower() == name.lower():
                try:
                    ratings.append(int(g.headers.get(f"{side}Elo", "")))
                except ValueError:
                    pass
    return int(np.median(ratings)) if ratings else 0


def add_player(
    lichess: str | None = None,
    chesscom: str | None = None,
    pgn: str | None = None,
    player: str | None = None,
    rating: int | None = None,
    n_games: int = 75,
    depth: int = 5,
    progress=None,
) -> dict:
    """Add one player to the cohort: fetch, analyse, fit, and re-run the PCA.

    The population prior is kept fixed so existing profiles do not change;
    the PCA axes are recomputed with the new player included.
    """
    from .ingest import fetch_chesscom
    from .profile import slugify

    def step(msg, frac=None):
        print(f"  {msg}", file=sys.stderr)
        if progress:
            progress(msg, frac)

    prior = load_population()
    if lichess:
        step(f"looking up {lichess} on Lichess", 0.02)
        pid, username, r, n_blitz = _lichess_rating(lichess)
        if n_blitz < 20:
            raise ValueError(f"{username} has only {n_blitz} rated blitz games on Lichess (need 20+)")
        step(f"downloading {n_games} rated blitz games", 0.05)
        url = f"{LICHESS}/api/games/user/{pid}?max={n_games}&rated=true&perfType=blitz&clocks=true&evals=false&opening=false"
        text = _get(url, "application/x-chess-pgn", timeout=300)
        source = "lichess blitz"
    elif chesscom:
        step(f"downloading games of {chesscom} from Chess.com", 0.05)
        text = fetch_chesscom(chesscom, n_games)
        username = chesscom
        pid = "chesscom-" + slugify(chesscom)
        r = rating or _pgn_rating(text, chesscom)
        source = "chess.com"
    elif pgn:
        if not player:
            from .ingest import player_names

            player = next(iter(player_names(parse_pgn(pgn))), None)
        if not player:
            raise ValueError("could not find a player name in the PGN")
        text, username, pid = pgn, player, "pgn-" + slugify(player)
        r = rating or _pgn_rating(pgn, player)
        source = "pgn upload"
    else:
        raise ValueError("give a Lichess username, a Chess.com username, or a PGN")
    if rating:
        r = rating
    if not r:
        raise ValueError("no rating found - please enter the player's rating (needed to separate style from strength)")

    games = games_of(parse_pgn(text), username)
    if len(games) < 20:
        raise ValueError(f"only {len(games)} usable games for {username} (need 20+)")
    GAMES.mkdir(parents=True, exist_ok=True)
    (GAMES / f"{pid}.pgn").write_text(text)

    players = json.loads(PLAYERS.read_text()) if PLAYERS.exists() else []
    players = [p for p in players if p["id"] != pid]
    entry = {"id": pid, "username": username, "rating": r, "source": source, "added": True}
    players.append(entry)
    PLAYERS.write_text(json.dumps(players, indent=1))

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for f in ANALYSIS.glob(f"{pid}.*"):
        f.unlink()  # re-adding a player re-analyses their latest games
    step(f"analysing {len(games)} games with Stockfish (a few minutes)", 0.1)
    with _ADD_LOCK:
        return _finish_add(entry, pid, username, depth, prior, step)


def _finish_add(entry, pid, username, depth, prior, step) -> dict:
    _analyse_player(pid, username, depth, progress=lambda i, n: step(f"analysing game {i}/{n}", 0.1 + 0.8 * i / n))
    step("fitting the personal model", 0.92)
    prof = fit_player(entry, prior)
    step("updating the cognition map", 0.97)
    out = run_pca()
    me = next(p for p in out["players"] if p["id"] == prof.id)
    step("done", 1.0)
    return {"player_id": prof.id, "name": username, "rating": entry["rating"], "map": me}


# ---------------------------------------------------------------------- PCA
STYLE_SKIP = {"eng_loss", "eng_best", "eng_blunder"}  # represented by the stats block


def cognition_vector(w: np.ndarray, prior: np.ndarray, blind: dict, stats: dict) -> dict[str, float]:
    v: dict[str, float] = {}
    for i, f in enumerate(FEATURE_NAMES):
        if f not in STYLE_SKIP:
            v[f"style:{f}"] = float(w[i] - prior[i])
    mi = blind.get("motif_index", {})
    for m in tactics.MOTIFS:
        v[f"blind:{m}"] = float(np.log(max(mi.get(m, 1.0), 1e-3)))
    for k in ("engine_match", "avg_loss_pawns", "blunder_rate"):
        v[f"skill:{k}"] = float(stats.get(k, 0.0))
    return v


VECTOR_LABEL = {
    **{f"style:{f}": FEATURE_LABEL[f] for f in FEATURE_NAMES},
    **{f"blind:{m}": f"blind to {tactics.MOTIF_LABEL[m]}" for m in tactics.MOTIFS},
    "skill:engine_match": "matches the engine's top move",
    "skill:avg_loss_pawns": "average loss per move",
    "skill:blunder_rate": "blunder rate",
}


def _kmeans(Z: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(20):
        C = Z[rng.choice(len(Z), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((Z - C[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best


def _silhouette(Z: np.ndarray, lab: np.ndarray) -> float:
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    s = []
    for i in range(len(Z)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def run_pca(n_components: int = 6) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(ANALYSIS.glob("*.vec.json"))]
    if len(rows) < 10:
        raise SystemExit("need at least 10 analysed players")
    keys = list(rows[0]["vectors"]["all"].keys())

    def mat(k):
        return np.array([[r["vectors"][k][c] for c in keys] for r in rows])

    V, VA, VB = mat("all"), mat("A"), mat("B")
    rating = np.array([r["rating"] for r in rows], dtype=float)

    # How much of each trait is just strength?
    rating_r = {c: float(np.corrcoef(V[:, j], rating)[0, 1]) if V[:, j].std() > 0 else 0.0 for j, c in enumerate(keys)}

    # Remove the linear effect of rating (fit on the full vectors, apply to halves too).
    R = np.column_stack([np.ones_like(rating), (rating - rating.mean()) / rating.std()])
    beta, *_ = np.linalg.lstsq(R, V, rcond=None)

    def resid(M):
        return M - R @ beta

    E = resid(V)
    mu, sd = E.mean(0), E.std(0)
    keep = sd > 1e-9
    keys_k = [c for c, k in zip(keys, keep) if k]

    def std(M):
        return (resid(M) - mu)[:, keep] / sd[keep]

    Z = std(V)
    U, S, Wt = np.linalg.svd(Z, full_matrices=False)
    var = S**2 / (S**2).sum()
    nc = min(n_components, len(S))
    W = Wt[:nc]  # components x features
    scores = Z @ W.T
    sA, sB = std(VA) @ W.T, std(VB) @ W.T

    # Split-half reliability: do the two halves of each player's games land at the same spot?
    comps = []
    for c in range(nc):
        r = float(np.corrcoef(sA[:, c], sB[:, c])[0, 1])
        sb = 2 * r / (1 + r) if r > -1 else r
        order = np.argsort(-np.abs(W[c]))[:8]
        comps.append(
            {
                "index": c + 1,
                "explained": round(float(var[c]), 4),
                "split_half_r": round(r, 3),
                "reliability": round(float(sb), 3),
                "loadings": [{"feature": keys_k[j], "label": VECTOR_LABEL.get(keys_k[j], keys_k[j]), "loading": round(float(W[c, j]), 3)} for j in order],
            }
        )
    # Null baseline: same reliability computation with the halves shuffled across players.
    rng = np.random.default_rng(0)
    null = [float(np.corrcoef(sA[:, 0], rng.permutation(sB[:, 0]))[0, 1]) for _ in range(200)]

    # Player types on the reliable components.
    reliable = [c for c in range(nc) if comps[c]["split_half_r"] >= 0.4] or [0, 1]
    Zr = scores[:, reliable] / scores[:, reliable].std(0)
    clusters = {}
    for k in range(2, 7):
        lab, C, _ = _kmeans(Zr, k)
        clusters[k] = (lab, C, _silhouette(Zr, lab))
    best_k = max(clusters, key=lambda k: clusters[k][2])
    lab, C, sil = clusters[best_k]

    types = []
    for j in range(best_k):
        members = lab == j
        centre = Z[members].mean(0)
        top = np.argsort(-np.abs(centre))[:6]
        types.append(
            {
                "type": j,
                "size": int(members.sum()),
                "mean_rating": round(float(rating[members].mean())),
                "traits": [{"feature": keys_k[i], "label": VECTOR_LABEL.get(keys_k[i], keys_k[i]), "z": round(float(centre[i]), 2)} for i in top],
            }
        )

    # Trait-level reliability (before PCA) for transparency.
    trait_rel = {}
    for j, c in enumerate(keys):
        if VA[:, j].std() > 0 and VB[:, j].std() > 0:
            trait_rel[c] = round(float(np.corrcoef(VA[:, j], VB[:, j])[0, 1]), 3)

    acc = {
        k: round(float(np.mean([r["evaluation"][k]["top1"] for r in rows])), 4)
        for k in ("personal", "population", "engine_top_move", "random")
    }
    gain = np.array([r["evaluation"]["personal"]["top1"] - r["evaluation"]["population"]["top1"] for r in rows])
    ll_gain = np.array([r["evaluation"]["population"]["logloss"] - r["evaluation"]["personal"]["logloss"] for r in rows])

    out = {
        "players": [
            {"id": r["id"], "name": r["name"], "rating": r["rating"], "added": r.get("added", False), "source": r.get("source", "lichess blitz"), "pc": [round(float(x), 3) for x in scores[i]], "pcA": [round(float(x), 3) for x in sA[i]], "pcB": [round(float(x), 3) for x in sB[i]], "type": int(lab[i])}
            for i, r in enumerate(rows)
        ],
        "components": comps,
        "null_split_half_r_95": round(float(np.quantile(np.abs(null), 0.95)), 3),
        "types": types,
        "k": best_k,
        "silhouette": round(sil, 3),
        "silhouette_by_k": {k: round(v[2], 3) for k, v in clusters.items()},
        "reliable_components": [c + 1 for c in reliable],
        "rating_correlation": dict(sorted(rating_r.items(), key=lambda kv: -abs(kv[1]))[:12]),
        "trait_reliability": dict(sorted(trait_rel.items(), key=lambda kv: -kv[1])),
        "prediction": {
            "mean_top1": acc,
            "personal_beats_population_top1": int((gain > 0).sum()),
            "mean_top1_gain": round(float(gain.mean()), 4),
            "mean_logloss_gain": round(float(ll_gain.mean()), 4),
            "players_with_better_logloss": int((ll_gain > 0).sum()),
            "n": len(rows),
        },
        "n_players": len(rows),
        "n_features": len(keys_k),
    }
    PCA_OUT.write_text(json.dumps(out, indent=1))
    return out


def print_report(out: dict) -> None:
    print(f"\n{out['n_players']} players, {out['n_features']} cognition features (rating regressed out)")
    pr = out["prediction"]
    print(f"Prediction top-1 (held-out games, mean over players): {pr['mean_top1']}")
    print(f"  personal beats population for {pr['personal_beats_population_top1']}/{pr['n']} players (top-1); better log-loss for {pr['players_with_better_logloss']}/{pr['n']}")
    print(f"Split-half null (shuffled players) |r| 95%: {out['null_split_half_r_95']}")
    for c in out["components"]:
        print(f"\nPC{c['index']}: {c['explained']:.1%} of variance, split-half r={c['split_half_r']:.2f} (reliability {c['reliability']:.2f})")
        for l in c["loadings"][:6]:
            print(f"   {l['loading']:+.2f}  {l['label']}")
    print(f"\nPlayer types: k={out['k']} (silhouette {out['silhouette']}, by k {out['silhouette_by_k']})")
    for t in out["types"]:
        print(f"  type {t['type']}: {t['size']} players, mean rating {t['mean_rating']}: " + "; ".join(f"{x['label']} {x['z']:+.1f}" for x in t["traits"][:4]))
    print("\nMost strength-related raw traits (r with rating):")
    for k, v in list(out["rating_correlation"].items())[:8]:
        print(f"  {v:+.2f}  {VECTOR_LABEL.get(k, k)}")
