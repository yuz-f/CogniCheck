# Training the from-scratch network on a Mac

For Apple Silicon Macs (M1 or newer, 16 GB memory or more). Training uses the Mac's GPU through
PyTorch's `mps` device. Real player data stays on the Mac, in the git-ignored `data/real/`.

## 1. Set up (once)

You need Python 3.10 or newer and the data bundle `cognicheck-mac-data.tar.gz` (about 25 MB). It
holds the encoded cohort games, including the test set, and the two player lists that keep cohort
and lockbox players out of the Lichess download.

```bash
git clone https://github.com/yuz-f/CogniCheck.git
cd CogniCheck
git checkout claude/keen-babbage-xv4ocy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch zstandard pytest
tar xzf ~/Downloads/cognicheck-mac-data.tar.gz
python -m pytest -q tests/test_chessnet.py
```

The tests should report `3 passed`.

## 2. Speed test (about 2 minutes)

```bash
python scripts/net_train.py --limit 60000 --epochs 1 --device mps --tag speed_mps
python scripts/net_train.py --limit 60000 --epochs 1 --device cpu --tag speed_cpu
```

Compare the seconds in the two `epoch 1:` lines. The cloud machine needs about 270 seconds for
that amount of data. If `mps` is not clearly faster than `cpu` on your Mac, use `--device cpu` below.

## 3. Stage B data: 20 million Lichess positions (about 1 hour)

```bash
caffeinate -i python scripts/net_lichess.py --month 2026-10 --positions 20000000 --workers 6
```

`caffeinate -i` keeps the Mac awake. The download streams a public monthly archive and stores
about 4.5 GB of encoded positions in `data/real/net/lichess/`. A dropped connection resumes by itself.

## 4. Stage B training

```bash
caffeinate -i python scripts/net_train.py --train-shards data/real/net/lichess --epochs 2 --device mps --tag B_b6c64
```

The `step` lines show positions per second, so the remaining time is easy to estimate. At the end
the script prints the test result on the cohort's held-out decisions, the same positions as
today's model (55.1% top-1) and Maia-2 (49.2%).

## 5. Send back

- the last two lines the training printed, or the file `data/real/net/result_B_b6c64.json`
- the trained network `data/real/net/net_B_b6c64.pt` (about 2 MB), for stages C and D
