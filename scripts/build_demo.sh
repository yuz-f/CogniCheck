#!/usr/bin/env bash
# Rebuild the population prior and the three demo persona profiles.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m cognicheck population data/games/*.pgn --only "Club Opponent" --max-games 40
for p in kai sasha tomas; do
  name=$($PY -c "from cognicheck.personas import by_name; print(by_name('$p').name)")
  $PY -m cognicheck build "data/games/$p.pgn" --player "$name" > "data/profiles/$p.txt" 2>&1 &
done
wait
cat data/profiles/*.txt
