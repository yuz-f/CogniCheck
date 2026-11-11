# CogniCheck

**Predict what a *specific* chess player will play next — not the best move, *their* move.**

CogniCheck learns one player's habits from their own games and then, for any
position, tells you:

* the moves **this player** is most likely to play, with probabilities and the reasons
  ("knight move — +0.8, *especially for this player*"),
* what they will probably **notice** and what they will probably **overlook**
  (threats against them, chances they have, likely moves that backfire),
* where their **attention** is on the board and what they are probably **planning**,
* their personal **blind spots** (e.g. "overlooks threats from knight moves ×1.9 more
  than their own average") and their **style fingerprint** vs. an average player.

On top of this sits a game: replay a held-out game of the player (one the model has
never seen), guess each of their moves, and compete against the model.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# optional but recommended: Stockfish on PATH (or STOCKFISH_PATH=/path/to/stockfish)
python -m cognicheck serve           # http://127.0.0.1:8000
```

The repository ships three synthetic demo players (see below) so the app works
immediately. To add a real player:

```bash
python -m cognicheck lichess <username>             # downloads up to 200 games
python -m cognicheck lichess <username> --chesscom  # Chess.com instead
python -m cognicheck build my_games.pgn --player "My Name"
```

…or use **Players → Add a player** in the web app (Lichess, Chess.com, or PGN upload).
More games = more personal. ~50 games gives a usable profile, 200+ a sharp one.

## How it works

```
games ──► every decision the player faced ──► features of every legal move
                                                     │
          population prior ──(L2 pull)──► personal softmax model  +  opening book
                                                     │
              position ──► P(move | player) ──► perspective: notices / overlooks /
                                                attention / plans / blind spots
```

1. **Move features** (`cognicheck/features.py`). Every legal move is described by ~33
   interpretable features: piece moved, capture & exchange value (SEE), check, forward /
   backward, distance, centralisation, *leaves material en prise*, *rescues a threatened
   piece*, *ignores an existing threat*, recapture, reacts to the opponent's last move,
   approaches the enemy king, pawn storm, development, early queen, and — if Stockfish is
   available — objective loss in pawns / top-move / blunder.

2. **Personal model** (`cognicheck/model.py`). A conditional logit: `P(m) = softmax(w·x_m)`
   over the legal moves, fit by Newton's method. The weights are pulled towards a
   **population prior** (`data/population.json`) with an L2 penalty, so a player with
   few games looks average and their quirks emerge as games accumulate. The deviation
   `w − w_prior` *is* the player's style. The engine is just one feature: a strong
   player gets a big weight on "objective loss", a beginner a small one, so the model
   predicts human mistakes rather than engine moves.

3. **Opening book.** The player's own repertoire (first 24 plies) is blended with the
   model via Dirichlet smoothing.

4. **Blind spots** (`cognicheck/profile.py`). Measured three ways, each relative to the
   player's *own* average (so a weak player is not simply "blind to everything"), and
   grouped by motif — knight / bishop / … moves, backward moves, sideways moves,
   long-range moves, long diagonals, checks, forks, discovered attacks:
   * **threats** – an opponent capture threat is on the board: did they react?
   * **walk-ins** – how often their move allows a tactic of that motif, compared with
     a random legal move,
   * **missed wins** – they could win material or mate: did they?
   * with Stockfish also: *which kind of move punishes their blunders*, vs. base rate.

5. **Perspective** (`cognicheck/predictor.py`). For the current position the probability
   that the player reacts to each threat is the model's probability mass on moves that
   neutralise it — so it's grounded in the same personal model, and combined with
   the blind-spot history for the explanation.

## Validation with planted blind spots

Lichess was not reachable from the build environment, so the demo players are
**synthetic personas** (`cognicheck/personas.py`): Stockfish-driven bots with a deliberately
broken perception and biased taste. A persona blind to knight moves evaluates each
candidate as if the opponent's knight replies did not exist. This lets us check that the
profiler *rediscovers* the planted traits from games alone — see the results section below.

| Persona | Planted style | Planted blind spot |
|---|---|---|
| Knightmare Kai | loves knight moves, checks | opponent's knight moves |
| Stormy Sasha | pawn storms, king attack, never retreats | opponent's backward moves |
| Tidy Tomas | trades, early castling, no early queen | long diagonals |

Regenerate with `python -m cognicheck generate --persona kai --games 120` and rebuild
with `scripts/build_demo.sh`.

### Results (120 games per persona, 18 held-out games each, Stockfish depth 6)

**Style is recovered clearly.** The largest personal deviations from the population are
exactly the planted tastes:

| Persona | Top recovered style traits |
|---|---|
| Kai | knight moves ×1.56, reacts to last move ×1.44, checks ×1.26 |
| Sasha | pawn storm ×2.18, king approach ×2.08, backward moves ×0.56, forward ×1.56 |
| Tomas | castles ×2.06, even trades ×1.69, early queen ×0.66, development ×1.45 |

**Blind spots are recovered, but not cleanly for every persona:**

* Sasha (blind to backward moves): *backward* is the #1 combined motif (×1.42); "walks
  into backward moves" ×1.79 is the top row. ✔
* Tomas (blind to long diagonals): "overlooks threats from long diagonals" ×1.56 and
  "from bishop moves" ×1.51 are the top *threat* rows. ✔ But the combined ranking puts
  "misses quiet wins" and "misses mates" above them — noise from the opportunity view.
* Kai (blind to knight moves): "overlooks threats from knight moves" ×1.33 and
  "walks into knight moves" ×1.24 show up, and forks are ×3.5 over-represented among
  the moves punishing his blunders; but *backward* ranks above *knight* in the combined
  score. Partial ✔.

**Move prediction (top-1 on held-out games):**

| | Kai | Sasha | Tomas |
|---|---|---|---|
| personal model + book | 28.6% | **35.0%** | 32.7% |
| average-player model | 28.0% | 30.8% | 32.3% |
| Stockfish top move | 27.0% | 28.7% | 29.2% |
| random legal move | 4.4% | 5.0% | 4.0% |

The personal model beats the engine for all three, but beats the population model by a
meaningful margin only for Sasha, whose style is strongest. For Kai and Tomas the gap
(+0.4–0.6 pts on ~900 positions) is within noise. The personas pick moves mostly by
noisy engine evaluation, which caps how predictable they are. Real humans are much
more habitual (repertoires, recurring plans), so the opening book and style terms
should matter more on real games — this still needs to be checked on real Lichess
data.

**Known weakness:** the predictive model only knows blind spots indirectly (through
"ignores threat" / "hangs material"). Adding motif-specific features (e.g. "leaves a
knight capture available") would let it learn the blind spot directly and should
improve prediction for players like Kai.

## Live games

The **Live** tab follows a game while it is being played on Lichess (game link, game
ID, or a username, which follows their current game). It uses Lichess's public game
stream, so no login is needed. As soon as it is the profiled player's turn, the model
makes and timestamps its prediction *before* the move arrives. You lock in your own
guess, then the real move comes in and both are scored, with clocks shown. The
profile is matched by player name, or you pick one and say which colour it plays.

Chess.com has no public live move feed, so live mode is Lichess-only (Lichess
Broadcasts also cover over-the-board events).

**Simulate live** replays a held-out game with the same event format and a delay
per move, for demos and offline testing.

Data gathering, in two layers:

1. **History → profile:** Lichess export / Chess.com archives, with clocks. Planned:
   incremental fetches (`since=` last game), profiles per time control, and a cache of
   engine evaluations per position.
2. **Live → prediction:** Lichess stream (`/api/stream/game/{id}`), pushed to the
   browser as Server-Sent Events. Every prediction is logged with timestamp, actual
   move, guess and clock (`GET /api/live/{id}/log`). Planned: feed finished live games
   back into the player's profile, and use clock time as a feature (blind spots under
   time pressure).

> The Lichess stream parser was written against Lichess's documented format and is
> unit-tested with sample lines, but it has not yet run against the real Lichess
> server: Lichess was not reachable from the build environment.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/players` | list profiles |
| GET | `/api/players/{id}` | profile: accuracy, style, blind spots |
| POST | `/api/players` | build from `{pgn, player}` / `{lichess}` / `{chesscom}` → job |
| GET | `/api/jobs/{id}` | build progress |
| POST | `/api/predict` | `{player_id, fen?, moves?[]}` → predictions + perspective |
| POST | `/api/game` | start a prediction game on a held-out game |
| POST | `/api/game/{sid}/guess` | `{uci}` → reveal, model reading, next position |
| POST | `/api/game/{sid}/hint` | perspective without the answer (−2 points) |
| POST | `/api/live` | `{source}` (Lichess link/id/username) or `{player_id, simulate_game, delay}` |
| GET | `/api/live/{id}/events` | Server-Sent Events: a snapshot per update |
| POST | `/api/live/{id}/guess` | `{uci}` lock in a guess before the move arrives |
| GET | `/api/live/{id}/log` | every prediction with timestamps, actual move, guess |

## Tests

```bash
pip install pytest httpx && python -m pytest -q
```

## Ideas for next steps

* clock-aware features (Lichess PGNs include `%clk`): blind spots under time pressure,
* a neural backbone (Maia-style) as the population prior, with the personal model as a
  fine-tuned head,
* blind spots by phase / by board region, "tilt" after losing material.
