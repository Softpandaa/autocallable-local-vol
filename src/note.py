"""Note value V(c) of Section 5.1 and the coupon strike search of Section 6."""

import numpy as np

from config import (BRACKET_START, BRACKET_STEP, HORIZON, KNOCK_IN, KNOCK_OUT,
                    MAX_ITERATIONS, MAX_TRIALS, NOTIONAL, OBSERVATION_DAYS, PATHS,
                    TARGET_VALUE, TOLERANCE)


def structure(laggard, rate):
    """Parts of V(c), equation (6), that do not depend on the coupon terms."""
    observed = laggard[:, list(OBSERVATION_DAYS)]
    knocked_out = (observed >= KNOCK_OUT).astype(int)

    # Pi_m of Section 5.1: a note knocked out at observation i pays no coupon after i.
    alive = np.hstack((np.ones((PATHS, 1)), np.cumprod(1 - knocked_out, axis=1)[:, :-1]))
    times = np.arange(1, len(OBSERVATION_DAYS) + 1) / 2
    discount = np.exp(-rate(times) * times).reshape(-1, 1)

    ever = np.any(knocked_out, axis=1)
    first = times[knocked_out.argmax(axis=1)]
    knock_out_value = np.dot(ever, np.exp(-rate(first) * first)) * NOTIONAL

    knocked_in = (laggard.min(axis=1) <= KNOCK_IN).astype(int)
    final = np.dot(1 - ever, NOTIONAL * (np.minimum(1, laggard[:, -1]) * knocked_in
                                         + (1 - knocked_in)))
    final_value = final * np.exp(-rate(HORIZON) * HORIZON)
    return observed, alive, discount, knock_out_value, final_value


def value(parts, strike, max_interest, min_interest):
    """Note value V(c) of equation (6) at coupon strike c."""
    observed, alive, discount, knock_out_value, final_value = parts
    above = observed >= strike
    coupon = NOTIONAL * (max_interest / 2 * above + min_interest / 2 * (1 - above)) * alive
    return (np.sum(coupon.dot(discount)) + knock_out_value + final_value) / PATHS


def coupon_strike(parts, max_interest, min_interest):
    """Coupon strike c with |V(c) - TARGET_VALUE| <= TOLERANCE, Section 6; NaN if not found."""
    def at(strike):
        return value(parts, strike, max_interest, min_interest)

    strike, trials = BRACKET_START, 0
    if at(strike) > TARGET_VALUE:
        while at(strike) > TARGET_VALUE and trials < MAX_TRIALS:
            strike += BRACKET_STEP
            trials += 1
    else:
        while at(strike) <= TARGET_VALUE and trials < MAX_TRIALS:
            strike -= BRACKET_STEP
            trials += 1
    if trials >= MAX_TRIALS:
        return np.nan

    low, high = (BRACKET_START, strike) if strike > BRACKET_START else (strike, BRACKET_START)
    middle = (low + high) / 2
    for _ in range(MAX_ITERATIONS):
        if abs(at(middle) - TARGET_VALUE) <= TOLERANCE:
            return middle
        if at(middle) > TARGET_VALUE:
            low = middle
        else:
            high = middle
        middle = (low + high) / 2
    return np.nan
