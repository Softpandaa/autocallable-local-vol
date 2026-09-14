"""Valuation of the autocallable note and the coupon strike that prices it at par target."""

import numpy as np

from config import (BRACKET_START, BRACKET_STEP, HORIZON, KNOCK_IN, KNOCK_OUT,
                    MAX_ITERATIONS, MAX_TRIALS, NOTIONAL, OBSERVATION_DAYS, PATHS,
                    TARGET_VALUE, TOLERANCE)


def structure(laggard, rate):
    """The parts of the note that do not depend on the coupon terms.

    Returns the observed laggard levels, the indicator that the note is still
    alive at each observation, the discount factors, and the present values of
    the knock out and final redemptions.
    """
    observed = laggard[:, list(OBSERVATION_DAYS)]
    knocked_out = (observed >= KNOCK_OUT).astype(int)

    # A note knocked out at observation i pays no coupon after i.
    alive = np.hstack((np.ones((PATHS, 1)), np.cumprod(1 - knocked_out, axis=1)[:, :-1]))
    times = np.arange(1, len(OBSERVATION_DAYS) + 1) / 2
    discount = np.exp(-rate(times) * times).reshape(-1, 1)

    ever = np.any(knocked_out, axis=1)
    first = (np.where(ever, knocked_out.argmax(axis=1), 0) + 1) / 2
    knock_out_value = np.dot(ever, np.exp(-rate(first) * first)) * NOTIONAL

    knocked_in = (laggard.min(axis=1) <= KNOCK_IN).astype(int)
    final = np.dot(1 - ever, NOTIONAL * (np.minimum(1, laggard[:, -1]) * knocked_in
                                         + (1 - knocked_in)))
    final_value = final * np.exp(-rate(HORIZON) * HORIZON)
    return observed, alive, discount, knock_out_value, final_value


def value(parts, strike, max_interest, min_interest):
    """Present value of the note per unit of face, for one coupon strike."""
    observed, alive, discount, knock_out_value, final_value = parts
    above = observed >= strike
    coupon = NOTIONAL * (max_interest / 2 * above + min_interest / 2 * (1 - above)) * alive
    return (np.sum(coupon.dot(discount)) + knock_out_value + final_value) / PATHS


def coupon_strike(parts, max_interest, min_interest):
    """Coupon strike at which the note prices at the target.

    The bracket is opened by stepping away from a strike of one until the value
    crosses the target, then bisected. Exhausting either the trial count or the
    iteration count returns no result, which is what happens at the original
    coupon rates of two percent and one basis point.
    """
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
