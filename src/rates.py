"""Overnight index swap curves converted to a continuous rate function."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def get_continuous_r(r, t):
    """Continuously compounded rate of a rate r compounded daily on a basis of t days."""
    return t*np.log(1+r/t)


def get_r_func(file, t):
    """Rate curve r(T) of Section 3, a cubic spline through the continuous OIS rates."""
    temp = pd.read_excel(file, index_col=0)
    r = np.array([get_continuous_r(x, t)/100 for x in temp.iloc[0]])  # quotes in percent
    T = temp.columns
    return CubicSpline(T, r)
