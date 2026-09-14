"""Overnight index swap curves converted to a continuous rate function."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def get_continuous_r(r, t):
    """
    This function turns daily compounded r into continuously compounded r
    r: daily compounded r
    t: number of days compounded
    
    Return: continuously compounded r
    """
    return t*np.log(1+r/t)

def get_r_func(file, t):
    """
    This function returns a function that get r for each T
    file: file name containing all the daily compounded r
    t: number of days compounded
    
    Return: a function that get r for each T
    """
    temp = pd.read_excel(file, index_col=0)
    r = np.array([get_continuous_r(_, t)/100 for _ in temp.iloc[0]])
    T = temp.columns
    return CubicSpline(T, r)
