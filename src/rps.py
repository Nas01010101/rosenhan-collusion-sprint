"""Scripted 3-player iterated Rock-Paper-Scissors from Bonjour et al. (2022), Sec. 4.1.

A and C play uniformly at random. B (the assistant) plays the move that loses to A with
probability cp (collusion probability), otherwise uniformly at random.
Actions: 0=rock, 1=paper, 2=scissors; x beats (x+2)%3, so the move losing to a is (a+2)%3.
"""

import numpy as np


def play(n_games, cp, rng=None, order=(0, 1, 2)):
    """Return (3, n_games) actions; rows follow `order` so the colluding pair can be relabelled."""
    rng = np.random.default_rng(rng)
    a = rng.integers(0, 3, n_games)
    c = rng.integers(0, 3, n_games)
    assist = rng.random(n_games) < cp
    b = np.where(assist, (a + 2) % 3, rng.integers(0, 3, n_games))
    rows = np.stack([a, b, c])
    return rows[list(order)]
