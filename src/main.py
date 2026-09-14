"""Build the volatility surfaces, simulate the three indices, and report the grid."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import note
import rates
import simulate
import surface
from config import (CHAIN_FILE, DROP_IMPLIED, DROP_LOCAL, HORIZON, INDICES,
                    MATURITY_GRID, MAX_INTEREST, MIN_INTEREST, MONEYNESS_GRID,
                    MONEYNESS_WINDOW, OIS_BASIS, OIS_FILE, STEPS, STRIKE_STEP)


def _drop(fitted, table, maturities):
    """Remove the maturities whose curve was rejected, from both fit and table."""
    kept = [[fit, t] for fit, t in fitted if round(t, 3) not in maturities]
    return kept, table.loc[[t for _, t in kept]]


def _dividend(table):
    """Implied dividend interpolated across maturity, floored at zero."""
    spline = CubicSpline(table.index, table["q"])
    return lambda t: np.maximum(0.0, spline(t))


def fit_curves():
    """Per index, the rate curve, the fitted volatility curves, and the spot."""
    spot = simulate.load_spot()
    last = spot.iloc[-1]
    fitted = {}
    for name in INDICES:
        rate = rates.get_r_func(str(OIS_FILE[name]), OIS_BASIS[name])
        chains, table = surface.get_processed_data(
            str(CHAIN_FILE[name]), last[name], rate, STRIKE_STEP[name])
        implied_all = surface.get_vol_sur_M_all(chains, table)
        local_all = surface.get_loc_vol_sur_M_all(chains, table)
        implied, _ = _drop(implied_all, table, DROP_IMPLIED[name])
        local, local_table = _drop(local_all, table, DROP_LOCAL[name])
        fitted[name] = {"rate": rate, "spot": last[name],
                        "implied_all": implied_all, "local_all": local_all,
                        "implied": implied, "local": local,
                        "local_table": local_table}
        print(f"{name} curves fitted", flush=True)
    return spot, fitted


def surface_families(entry):
    """The six surfaces the report compares, as volatility on a moneyness grid.

    Two interpolations across maturity, cubic and linear, applied to the implied
    curves, to the Dupire local curves, and to Gatheral's mapping of the implied
    curves. Section 4.2 selects Dupire with linear interpolation.
    """
    moneyness = np.linspace(*MONEYNESS_GRID)
    maturity = np.linspace(*MATURITY_GRID)
    y = np.log(moneyness)
    raw = {
        "implied_cubic": surface.get_IV_M_T_Cubic(entry["implied"], y, maturity),
        "implied_linear": surface.get_IV_M_T_Linear(entry["implied"], y, maturity),
        "dupire_cubic": surface.get_IV_M_T_Cubic(entry["local"], y, maturity),
        "dupire_linear": surface.get_IV_M_T_Linear(entry["local"], y, maturity),
        "gatheral_cubic": surface.get_local_vol(entry["implied"], y, maturity, True),
        "gatheral_linear": surface.get_local_vol(entry["implied"], y, maturity, False),
    }
    with np.errstate(invalid="ignore"):
        return {k: pd.DataFrame(np.sqrt(v), index=maturity, columns=moneyness)
                for k, v in raw.items()}


def blank_share(frame):
    """Fraction of the traded moneyness window where the fit gives no real volatility.

    Restricted to MONEYNESS_WINDOW, so this is the quantity Table 1 reports.
    Outside that window the fitted curve is pure extrapolation.
    """
    low, high = MONEYNESS_WINDOW
    window = frame.loc[:, (frame.columns >= low) & (frame.columns <= high)]
    return float(np.isnan(window.to_numpy()).mean())


def project_paths(spot, fitted):
    """Index paths under the selected local volatility surface."""
    noise = simulate.driving_noise(simulate.cholesky_factor(spot))
    grid = np.linspace(0, HORIZON, STEPS + 1)
    return {name: simulate.project(entry["spot"], entry["rate"],
                                   _dividend(entry["local_table"]),
                                   entry["local"], grid, noise[name])
            for name, entry in fitted.items()}


def coupon_grid(paths, rate):
    relative = np.array([paths[name] / paths[name][0, 0] for name in INDICES])
    parts = note.structure(np.min(relative, axis=0), rate)
    rows = np.linspace(*MAX_INTEREST)
    columns = np.linspace(*MIN_INTEREST)
    return pd.DataFrame(
        {round(c, 6): [note.coupon_strike(parts, r, c) for r in rows] for c in columns},
        index=np.round(rows, 6))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    spot, fitted = fit_curves()
    for name, entry in fitted.items():
        families = surface_families(entry)
        print(name, "share of the moneyness window with no real volatility:",
              {k: round(blank_share(v), 3) for k, v in families.items()}, flush=True)
    paths = project_paths(spot, fitted)
    print(coupon_grid(paths, fitted["SPX"]["rate"]).round(3).to_string())
