"""Spot history, the correlated driving noise, and the local volatility projection."""

import numpy as np
import pandas as pd

from config import (CORRELATION_ORDER, INDICES, PATHS, SEEDS, SPOT_FILE, STEPS,
                    TICKER, TRADING_DAYS_PER_YEAR)
from surface import interp_linear


def load_spot():
    """Daily closes of the three indices, columns renamed to the index names."""
    frame = pd.read_csv(SPOT_FILE, index_col=0, parse_dates=True).dropna()
    frame = frame.rename(columns={TICKER[name]: name for name in INDICES})
    return frame[list(CORRELATION_ORDER)]


def cholesky_factor(spot):
    """Lower Cholesky factor of the daily log return correlation matrix."""
    spot = spot[list(CORRELATION_ORDER)]
    returns = np.log(spot / spot.shift(1)).dropna()
    return np.linalg.cholesky(returns.corr().to_numpy())


def driving_noise(factor):
    """One correlated Gaussian increment array per index, keyed by index name."""
    independent = []
    for seed in SEEDS:
        generator = np.random.RandomState(seed)
        independent.append(generator.normal(size=(PATHS, STEPS)))
    correlated = [sum(factor[i, j] * independent[j] for j in range(i + 1))
                  for i in range(len(INDICES))]
    return dict(zip(CORRELATION_ORDER, correlated))


def project(spot_0, rate, dividend, local_surface, grid, noise):
    """Index paths of equation (5), the local variance interpolated linearly across maturity."""
    floor = 1.0  # smallest non-negative variance so far, at most 1; replaces negative ones
    path = np.full((PATHS, STEPS + 1), spot_0)
    forward = spot_0 * np.exp((rate(grid) - dividend(grid)) * grid)
    step = 1 / TRADING_DAYS_PER_YEAR
    for j in range(1, path.shape[1]):
        forward_rate = (grid[j] * rate(grid[j]) - grid[j - 1] * rate(grid[j - 1])) / step
        # The curves were fitted in ln(K / F), so the path level enters as the strike.
        variance = interp_linear(
            local_surface, np.log(path[:, j - 1] / forward[j - 1]), grid[j - 1])
        valid = variance >= 0
        if valid.any():
            floor = min(floor, variance[valid].min())
        variance = np.where(valid, variance, floor)
        sigma = np.sqrt(variance)
        path[:, j] = path[:, j - 1] * np.exp(
            (forward_rate - sigma ** 2 / 2) * step + sigma * noise[:, j - 1] * np.sqrt(step))
    return path
