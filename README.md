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

RESULTS_PLACEHOLDER

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

## Tests

```bash
pip install pytest httpx && python -m pytest -q
```

## Ideas for next steps

* clock-aware features (Lichess PGNs include `%clk`): blind spots under time pressure,
* a neural backbone (Maia-style) as the population prior, with the personal model as a
  fine-tuned head,
* blind spots by phase / by board region, "tilt" after losing material.
