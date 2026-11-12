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

Requires Python 3.9 or newer (on macOS use `python3`; once the virtualenv is active,
`python` works too).

```bash
python3 -m venv .venv && . .venv/bin/activate
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

1. **Move features** (`cognicheck/features.py`). Every legal move is described by ~50
   interpretable features: piece moved, capture & exchange value (SEE), check, forward /
   backward, distance, centralisation, *leaves material en prise*, *rescues a threatened
   piece*, *ignores an existing threat*, recapture, reacts to the opponent's last move,
   lands next to the enemy king, pawn storm, development, early queen; with Stockfish,
   objective loss in pawns in several bands plus the move's engine rank; **motif-specific
   vulnerability** ("allows a knight / bishop / fork / discovered / backward / long-range
   tactic"), so the model can *learn* a player's blind spots rather than only count them;
   **time-pressure interactions** (loss, hanging material and captures when under 20 s
   or 10 % of the clock, from the `%clk` tags in Lichess PGNs); and a **habit** feature:
   how often this player chooses this (piece, square) when it is available, relative to
   their base rate, computed leave-one-out so a move cannot recognise itself.

2. **Personal model** (`cognicheck/model.py`). A conditional logit: `P(m) = softmax(w·x_m)`
   over the legal moves, fit by Newton's method. The inverse Hessian gives a standard
   error per weight, so every style trait and learned blind spot is shown with ± and
   flagged as *reliable* only when it is at least two standard errors from zero. The weights are pulled towards a
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
   neutralise it, so it is grounded in the same personal model. Explanations decompose
   `log P(move)` feature by feature as `w_j · (x_j − E[x_j])`, the evidence that pulled
   the player towards this move relative to what the model expected. Blind-spot notes
   use the learned "allows a … tactic" weights when they are reliable, and the counted
   miss rates otherwise.

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

### Adding your own players

Use **Players → Add a player** in the app (Lichess username, Chess.com username or PGN),
or the command line:

```bash
python -m cognicheck add --lichess <username>
python -m cognicheck add --pgn games.pgn --player "Name" --rating 1650
```

The player gets the same treatment as the cohort:
- their 75 most recent rated blitz games are analysed (about a minute),
- a personal model is fitted against the fixed population prior, so existing profiles
  don't change,
- the PCA is re-run with them included.

Added players can be predicted right away (Play, Live, Analyse) and appear ringed and
named on the Map.

For PGN and Chess.com players a rating is needed to remove strength from the axes. A
Chess.com rating is not on the Lichess scale, so their map position is approximate.

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

## Cognition map of real players (PCA)

`python -m cognicheck cohort all` profiles a cohort of real Lichess players and maps
them. Its stages are `select`, `fetch`, `analyse`, `memory` (engine-free working-memory columns), `maia`, `fit` and `pca`, and each can be run alone.

The study run behind the numbers below:
- **Cohort:** 81 Lichess blitz players rated 812–2929. 72 were sampled evenly across
  rating bands from recent arenas, plus 9 from the blitz leaderboard. Each had 400+ rated
  blitz games on Lichess.
- **Games:** 75 recent rated blitz games per player, about 48,600 decisions in total,
  analysed with Stockfish at depth 5.
- **Prior:** the population prior is trained on the pooled real games, with an equal
  number of positions from every player.
- **Cognition vector:** 48 numbers per player: 30 style deviations from the population,
  15 blind-spot indices (log) and 3 accuracy stats.
- **Rating removed:** rating is regressed out of the vector before PCA. Without this,
  the first axis is just strength: raw engine match correlates r = 0.92 with rating.
- **Validity check:** each player is profiled twice more, from two disjoint halves of
  their games (odd/even). An axis is a real trait only if both halves put the player in
  the same place. With shuffled players, |r| stays below 0.22 (95%).

Real player data stays local: `data/real/` is git-ignored. Only aggregate results are
reported here.

### Adding your own players

Use **Players → Add a player** in the app (Lichess username, Chess.com username or PGN),
or the command line:

```bash
python -m cognicheck add --lichess <username>
python -m cognicheck add --pgn games.pgn --player "Name" --rating 1650
```

The player gets the same treatment as the cohort:
- their 75 most recent rated blitz games are analysed (about a minute),
- a personal model is fitted against the fixed population prior, so existing profiles
  don't change,
- the PCA is re-run with them included.

Added players can be predicted right away (Play, Live, Analyse) and appear ringed and
named on the Map.

For PGN and Chess.com players a rating is needed to remove strength from the axes. A
Chess.com rating is not on the Lichess scale, so their map position is approximate.

### Results

| Axis | Variance | Split-half r | Reading (after removing rating) |
|---|---|---|---|
| PC1 | 9.0% | **0.80** | *Careful vs. loose*: fewer blunders and less loss than their rating predicts, castles, central play; vs. king walks and early queen sorties |
| PC2 | 8.5% | **0.72** | *Classical developer vs. attacker*: develops minor pieces; vs. marching on the enemy king, pawn storms, knight and queen raids, long moves |
| PC3 | 7.2% | **0.67** | *Long-range vision*: castles, plays into the opponent's half; vs. blind to long diagonals, long-range and bishop moves, and pushes forward |
| PC4 | 5.8% | 0.40 | weak; mixed blind-spot terms |
| PC5 | 5.4% | **0.67** | *Trader / materialist*: even trades, wins material, central; vs. rook manoeuvres |
| PC6 | 4.8% | 0.58 | *Alert developer*: develops, punishes material, rarely ignores threats |

**What this says**

- **The axes are real.** Five of the six axes reproduce across independent halves of a
  player's games (r = 0.58–0.80, against a null of about 0.2). They are stable traits of
  how a player thinks, not noise, and they hold after removing playing strength.
- **Cognition is many-dimensional.** Each axis explains only 5–9% of the variance. No
  single "style" number captures a player, which is why the profile keeps the full vector.
- **There are no discrete player types.** k-means separation is weak (silhouette
  0.11–0.14 for k = 2–6). Players form a continuum, and the map should be read through
  its axes, not as boxes like "the attacker" or "the defender". The best split even
  isolates a single outlier as its own "type".
- **Blind spots partly track strength.** Before rating is removed, blindness to forks,
  long diagonals and long-range moves correlates r ≈ −0.55 with rating: stronger players
  see further.

**Prediction on held-out games (mean over 82 players, top-1)**

| Model | v1 (75 games) | v2 (75 games) | **v3 (200 games, Maia-2)** |
|---|---|---|---|
| Personal model | 38.8% | 40.4% | **50.9%** |
| Average-player model (rating band) | 38.1% | 39.5% | 50.5% |
| Average-player model (global) | — | — | 49.9% |
| Stockfish top move | 34.5% | 34.5% | 34.7% |
| Random legal move | 5.3% | 5.3% | 5.3% |
| Personal beats average-player (top-1) | 47 / 82 | 60 / 82 | 53 / 82 |
| Personal beats average-player (log-loss) | 75 / 82 | 77 / 82 | **78 / 82** |
| Personal log-loss | — | 1.973 | **1.563** |

v3 adds the Maia-2 human-move prior as a feature, 200 games per player instead of 75,
positional, think-time, attention and plan-persistence features, and rating-band priors.
The jump to ~51 % reaches the level of the published neural predictors, while every
prediction is still a sum of readable terms.

Ablation on the v3 split (zero a feature family, refit, 82 players):

| removed | personal top-1 | log-loss | personal beats population (top-1) |
|---|---|---|---|
| nothing | 50.9% | 1.563 | 53 / 82 |
| Maia-2 prior | 41.0% | 1.939 | 80 / 82 |
| all engine features | 50.4% | 1.603 | 59 / 82 |
| habit feature | 50.2% | 1.576 | 61 / 82 |
| engine-loss bands + rank | 50.9% | 1.578 | 54 / 82 |
| think-time interactions | 50.9% | 1.570 | 49 / 82 |
| time pressure | 50.8% | 1.568 | 54 / 82 |
| attention map | 50.7% | 1.567 | 54 / 82 |
| learned blind spots (`allows_*`) | 50.8% | 1.564 | 52 / 82 |
| rating-band prior (global instead) | 50.9% | 1.563 | 58 / 82 |
| plan persistence | 50.9% | 1.563 | 55 / 82 |
| positional | 50.9% | 1.564 | 56 / 82 |

How to read this: the **Maia-2 prior is now the backbone** (−10 points without it) and
it absorbs most of what the engine, habit and style terms used to carry on their own.
Without Maia the personal model still beats its population counterpart for 80 of 82
players, so the personal signal is real; with Maia the average-player model is already
strong and the personal gain shows mainly in log-loss (78 / 82) rather than top-1. The
new families (attention, plan persistence, think time, positional) each cost ≤0.2
points when removed: they are weak as *predictors* but they are the terms that make a
profile readable, and several come out as reliable traits for individual players
(e.g. own-attention ×1.8, plan persistence ×1.3 for a 2000-rated player).

With 200 games the PCA axes are markedly more stable than with 75 (split-half
r = 0.90 / 0.71 / 0.72 / 0.84 / 0.74 / 0.75 for PC1–PC6, null ≈ 0.2). The axis readings
above still hold; the new attention trait ("answers the opponent's last move although it
threatened nothing") loads on PC6.

On the synthetic personas, the model-learned blind spots rank the planted one
first for Kai (knight ×1.21, reliable) and Sasha (backward moves ×1.66, reliable);
for Tomas the long-range weight is first but not beyond noise.

### Working memory (v4)

Chess cognition research puts a player's working memory at roughly 3–5 "chunks". We
test that in three ways, all computed from the games themselves (`cognicheck/capacity.py`):

| Measure | What is counted | Cohort (82 players, median and 10–90 % range) | r with rating |
|---|---|---|---|
| Tactic complexity | walk-in rate for tactics that need 4+ pieces held in mind vs. plain 2-piece ones | 0.81× (0.69–1.00) | −0.62 |
| Overload | blunder rate in crowded positions (many pieces in contact) vs. quiet ones | 1.94× (1.34–2.70) | −0.48 |
| Focus span | share of moves made with one of the 3 most recently moved pieces, vs. chance | 1.06× (1.00–1.16) | −0.54 |
| | share of moves returning to a piece untouched for 8+ moves, vs. chance | 0.93× (0.84–0.99) | +0.51 |

**What this says**

- **Everyone overloads.** Across the cohort the blunder rate doubles from quiet
  positions (7 %) to crowded ones (14 %); when short of time it is 13 % vs. 18 %.
  Weaker players degrade more (r = −0.48).
- **Complex tactics are missed *relatively* more by weaker players.** Per candidate,
  everybody walks into a 4-piece tactic less often than into a plain hanging piece
  (crowded tactics are rarer and more often forced), but the ratio climbs steeply as
  rating falls (r = −0.62): a 1200 player is nearly as likely to miss a fork as a
  hanging piece, a 2500 player misses mostly the complex ones.
- **Narrow focus is a trait of weaker players.** The 3-most-recent-pieces effect is
  modest on average (×1.06) but correlates −0.54 with rating, and stronger players
  return more often to pieces they have not touched for a while (+0.51). That is the
  "tunnel vision" of club play made measurable.
- **After removing rating, the individual weights are still real.** The six model
  terms (`allows_simple`, `allows_complex`, `load_eng_loss`, `load_hangs`,
  `focus_recent`, `focus_stale`) have split-half reliabilities of 0.61 / 0.36 / 0.51 /
  0.34 / 0.55 / 0.52 against a null of 0.2, and are essentially uncorrelated with
  rating once in the model. 59 of 82 players have at least one working-memory trait
  beyond noise; the most common are "walks into simple tactics" (either direction) and
  "sticks to recently moved pieces" (narrow focus, 12 players, vs. 5 with the opposite).
- **As predictors they add little** (ablation: top-1 unchanged, log-loss 1.563 → 1.560).
  As with attention and plan persistence, the value is in the profile, not the hit
  rate. The crowdedness term now loads on PCA axis 2.

The profile page shows these under **Working memory** with the counts behind each ratio.

**Where the ceiling is.** Top-1 counts only an exact hit, among ~35 legal moves.
Published neural human-move predictors reach ~50–55 % on blitz; v3/v4 is in that range
with a 79-weight model that can explain every prediction. The personal terms beyond
Maia-2 now mostly improve calibration (log-loss) and readability rather than top-1;
the next real gain would need per-player state that Maia does not see, such as a
latent plan carried across moves or the player's own repertoire beyond the book.

## Optional: Maia-2 human-move prior

With `torch` and `maia2` installed, every legal move also gets two features from
Maia-2 (Tang et al., 2024), a network trained on millions of Lichess games to predict
what a human *of a given rating* plays: the log-probability of the move and whether it
is Maia's top choice. The personal model then learns how far each player deviates from
a typical player of their strength, and the explanations stay readable.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~200 MB
pip install --no-deps maia2 && pip install einops gdown pandas pyyaml pyzstd requests tqdm
python -m cognicheck cohort maia   # adds the neural columns to an existing analysis
python -m cognicheck cohort fit && python -m cognicheck cohort pca
```

The weights (280 MB) download on first use into `data/maia2_models/`. If the Google
Drive download fails behind a proxy, fetch them directly:

```bash
mkdir -p data/maia2_models && cd data/maia2_models
curl -L -o blitz_model.pt "https://drive.usercontent.google.com/download?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt&export=download&confirm=t"
curl -L -o config.yaml   "https://drive.usercontent.google.com/download?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0&export=download&confirm=t"
```

Inference costs about 20 ms per position on a CPU core. Without the package the
features are zero and everything else works as before.

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

* a two-stage plan → move mixture model (the current plan terms are soft features),
* blind spots by phase / by board region, "tilt" after losing material,
* repertoire memory beyond the opening book: recurring middlegame set-ups per player,
* fine-tuning Maia-2 per rating band on the cohort to tighten the prior.
