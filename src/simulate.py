"""Spot history, the correlated driving noise, and the local volatility projection."""

import numpy as np
import pandas as pd

from config import (CORRELATION_ORDER, INDICES, PATHS, SEEDS, SPOT_FILE, STEPS,
                    TICKER, TRADING_DAYS_PER_YEAR)
from surface import get_IV_M_T_Linear


def load_spot():
    """Daily closes of the three indices, columns renamed to the index names."""
    frame = pd.read_csv(SPOT_FILE, index_col=0, parse_dates=True).dropna()
    return frame.rename(columns={TICKER[name]: name for name in INDICES})[list(CORRELATION_ORDER)]


def cholesky_factor(spot):
    """Lower Cholesky factor of the daily log return correlation matrix."""
    returns = np.log(spot[list(CORRELATION_ORDER)] / spot[list(CORRELATION_ORDER)].shift(1)).dropna()
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
    """Index paths under the local volatility surface.

    The surface is a function of log forward moneyness, ln(K / F), the same
    argument the curves were fitted on, so the path level enters as the strike.

    Where the interpolated local variance is negative, it is replaced by the
    smallest non negative value seen so far, as the report states.
    """
    floor = 1.0
    path = np.full((PATHS, STEPS + 1), spot_0)
    forward = spot_0 * np.exp((rate(grid) - dividend(grid)) * grid)
    step = 1 / TRADING_DAYS_PER_YEAR
    for j in range(1, path.shape[1]):
        forward_rate = (grid[j] * rate(grid[j]) - grid[j - 1] * rate(grid[j - 1])) / step
        variance = get_IV_M_T_Linear(
            local_surface, np.log(path[:, j - 1] / forward[j - 1]), grid[j - 1])
        valid = variance >= 0
        if valid.any():
            floor = min(floor, variance[valid].min())
        variance = np.where(valid, variance, floor)
        sigma = np.sqrt(variance)
        path[:, j] = path[:, j - 1] * np.exp(
            (forward_rate - sigma ** 2 / 2) * step + sigma * noise[:, j - 1] * np.sqrt(step))
    return path
