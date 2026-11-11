"""Population prior: what an "average" club player's weights look like.

A trained prior (``data/population.json``, built with
``python -m cognicheck population``) is preferred; this hand-set fallback
encodes common sense so tiny profiles still behave sensibly.
"""
DEFAULT_PRIOR = {
    "capture": 0.4,
    "capture_gain": 1.2,
    "check": 0.4,
    "mate": 4.0,
    "castle": 1.0,
    "hangs": -1.2,
    "saves": 1.0,
    "ignores_threat": -0.6,
    "recapture": 1.2,
    "centrality": 0.5,
    "develops": 0.8,
    "early_queen": -0.3,
    "undo": -0.8,
    "king_move": -0.5,
    "distance": -0.3,
    "eng_loss": -0.8,
    "eng_best": 0.5,
}
