"""A Markov-chain model of the 15-minute RT-DA spread, for stochastic dynamic programming.

The spread is binned into a few states (deep discount ... scarcity spike). The model holds,
for each local hour of the day, the probability of moving between states from one interval
to the next and the average spread within each state. Spikes are rare, persistent for a
while, and concentrated in some hours; a Markov chain captures all three, and it keeps the
part that matters for decisions with a 25-minute data lag: how likely a spike seen now is to
still be there when the battery can act on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from batt.forecast.history import DAY_NS, HOUR_NS, PriceHistory, local_naive_ns

DEFAULT_EDGES = (-50.0, -15.0, 15.0, 50.0, 200.0, 1000.0)  # $/MWh between spread states


@dataclass(frozen=True)
class SpreadMarkovChain:
    edges: np.ndarray  # (states - 1,) boundaries between states
    values: np.ndarray  # (24, states) mean spread within each state, by local hour
    transition: np.ndarray  # (24, states, states) P(next | current), hour of the current one

    @property
    def n_states(self) -> int:
        return self.values.shape[1]

    def state(self, spread) -> np.ndarray:
        return np.digitize(spread, self.edges)

    @classmethod
    def fit(cls, hist: PriceHistory, edges=DEFAULT_EDGES, prior: float = 20.0) -> SpreadMarkovChain:
        """Estimate from a price history; sparse hours borrow strength from all hours.

        Each hour's transition rows and state means are shrunk toward the all-hours estimate
        with ``prior`` pseudo-observations, so rarely visited states (spikes at 4 am) still
        get sensible numbers.
        """
        edges = np.asarray(edges, dtype=float)
        k = edges.size + 1
        spread = hist.spread
        ok = ~np.isnan(spread)
        hour = (local_naive_ns(hist.start_ns(np.arange(len(hist))), hist.timezone) % DAY_NS) // (
            HOUR_NS
        )
        state = np.digitize(np.where(ok, spread, 0.0), edges)

        # Mean spread in each state, by hour, shrunk toward the all-hours mean.
        total = np.zeros((24, k))
        count = np.zeros((24, k))
        np.add.at(total, (hour[ok], state[ok]), spread[ok])
        np.add.at(count, (hour[ok], state[ok]), 1.0)
        fallback = np.r_[1.5 * edges[0], (edges[:-1] + edges[1:]) / 2, 1.5 * edges[-1]]
        pooled_n = count.sum(axis=0)
        pooled = np.where(pooled_n > 0, total.sum(axis=0) / np.maximum(pooled_n, 1), fallback)
        values = (total + prior * pooled) / (count + prior)

        # Transitions, by the hour of the interval they leave from.
        pair = ok[:-1] & ok[1:]
        moves = np.zeros((24, k, k))
        np.add.at(moves, (hour[:-1][pair], state[:-1][pair], state[1:][pair]), 1.0)
        all_hours = moves.sum(axis=0) + 0.5 * np.eye(k)  # unvisited states just stay put
        all_hours /= all_hours.sum(axis=1, keepdims=True)
        transition = moves + prior * all_hours[None]
        transition /= transition.sum(axis=2, keepdims=True)
        return cls(edges, values, transition)
