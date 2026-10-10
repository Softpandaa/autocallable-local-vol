"""Option chain processing, the implied and local volatility surfaces, and Dupire."""

from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize

from blackscholes import BS_Euro_Pricer_F, IV_Bisect, get_ATMF, get_implied_div


def _first_derivative(f, x, dx):
    """First derivative by the three point central difference."""
    return (f(x + dx) - f(x - dx)) / (2 * dx)


def _second_derivative(f, x, dx):
    """Second derivative by the three point central difference."""
    return (f(x + dx) - 2 * f(x) + f(x - dx)) / dx ** 2


def get_dC_dT(F, K, r, T, sig):
    """dC/dT of equation (2), over one trading day."""
    f = partial(BS_Euro_Pricer_F, F, K, r, sig=sig)
    return _first_derivative(f, T, 1/252)


def get_d2C_dK2(F, K, r, T, sig, step):
    """d2C/dK2 of equation (2), over 1% of the strike step."""
    f = partial(BS_Euro_Pricer_F, F, r=r, T=T, sig=sig)
    return _second_derivative(f, K, step*.01)


def get_local_vol_Dupire(F, K, r, T, sig, step):
    """Dupire local variance, equation (2)."""
    dC_dT = get_dC_dT(F, K, r, T, sig)
    d2C_dK2 = get_d2C_dK2(F, K, r, T, sig, step)
    return dC_dT / ((K**2)*d2C_dK2/2)


def chain_processor(chain, S, r, q, T, K_ATMF, step):
    """Implied and Dupire local volatility at each strike of one chain, Sections 3.2 and 3.3."""
    Processed_Chain = pd.DataFrame()
    C = chain["Call"]
    P = chain["Put"]
    K = chain["Strike"]
    n = len(C)
    for i in range(1, n+1):
        try:
            y = np.log(K[i] / K_ATMF)  # forward moneyness
            X = np.log(K[i] / S)  # spot moneyness
            if y > 0:  # out-of-the-money quote: the call above the forward, the put below
                if C[i] == 0:
                    continue
                IV = IV_Bisect(C[i], .0001, True, S, K[i], r, q, T)
            else:
                if P[i] == 0:
                    continue
                IV = IV_Bisect(P[i], .0001, False, S, K[i], r, q, T)
            # A negative local variance has no square root; the strike is dropped below.
            with np.errstate(invalid="ignore"):
                LV = np.sqrt(get_local_vol_Dupire(K_ATMF, K[i], r, T, IV, step))
            if np.isnan(LV) or np.isinf(LV):
                continue
            row = pd.DataFrame({"Spot Moneyness": X, "IV": IV, "Call": C[i], "Put": P[i],
                                "Strike": K[i], "LV": LV}, index=(y,))
            Processed_Chain = pd.concat([Processed_Chain, row])
        except:  # noqa: E722
            continue
    Processed_Chain.index.name = "Forward Moneyness"
    return Processed_Chain


def get_processed_data(file, S, r_func, step):
    """Processed chain of every sheet, and a table of r, q and the ATMF values per maturity."""
    processed_chains = []
    processed_tables = pd.DataFrame()
    xl = pd.ExcelFile(file)
    for sheet_name in xl.sheet_names:
        raw_data = pd.read_excel(file, sheet_name=sheet_name)
        raw_chain = raw_data.loc[1:].copy()  # row 0 describes the chain
        raw_chain.rename(columns={"Last":"Call", "Last.1":"Put"}, inplace=True)
        chain_des = raw_data.loc[0].iloc[0].split()
        T = int(chain_des[1][1:-3])/365  # calendar days, read from e.g. "(20d);"
        r = r_func(T)
        C_ATMF, K_ATMF = get_ATMF(raw_chain)
        implied_div = get_implied_div(S, K_ATMF, r, T)
        sig_ATMF = IV_Bisect(C_ATMF, .00001, True, S, K_ATMF, r, implied_div, T)
        LV_ATMF = np.sqrt(get_local_vol_Dupire(K_ATMF, K_ATMF, r, T, sig_ATMF, step))
        processed_chains.append(chain_processor(raw_chain, S, r, implied_div, T, K_ATMF, step))
        row = pd.DataFrame({"rf": r, "q": implied_div, "C_ATMF": C_ATMF, "K_ATMF": K_ATMF,
                            "sig_ATMF": sig_ATMF, "M_ATMF": np.log(K_ATMF/S),
                            "LV_ATMF": LV_ATMF}, index=(T,))
        processed_tables = pd.concat([processed_tables, row])
        print(f"T={T} processed")
    processed_tables.index.name = "Time to maturity"
    return processed_chains, processed_tables


def vol_sur_M(M, sig_ATMF, params):
    """Equation (4) at log forward moneyness M; sig_ATMF is the ATMF variance."""
    delta, gamma, kappa = params
    z = np.tanh(M * kappa)
    return sig_ATMF + delta*z/kappa + gamma/2*(z/kappa)**2


def fit_vol_sur_M(M, Vol, sig_ATMF):
    """Equation (4) fitted to the variances Vol by least squares (SLSQP)."""
    def rss(params):
        error = vol_sur_M(M, sig_ATMF, params) - Vol
        return error.dot(error)

    res = minimize(rss, [-1, .25, .25], method="SLSQP")  # start: delta, gamma, kappa
    return partial(vol_sur_M, sig_ATMF=sig_ATMF, params=res.x)


def get_vol_sur_M_all(chain, table, column, atmf):
    """Equation (4) fitted at each maturity to the squared column, as [curve, T] pairs."""
    surface = []
    for i, T in enumerate(table.index):
        M = np.array(chain[i].index)
        Vol = np.square(np.array(chain[i][column]))
        surface.append([fit_vol_sur_M(M, Vol, table.loc[T, atmf]**2), T])
    return surface


def _knots(vol_surfaces, M):
    """Maturities and the fitted curves evaluated at M."""
    T = [t for _, t in vol_surfaces]
    Sig = [vol_sur(M) for vol_sur, _ in vol_surfaces]
    return T, Sig


def interp_cubic(vol_surfaces, M, T):
    """Fitted curves at M, cubic spline across maturity (Appendix B), evaluated at T."""
    return CubicSpline(*_knots(vol_surfaces, M))(T)


def interp_linear(vol_surfaces, M, T):
    """Fitted curves at M, linear across maturity, evaluated at T."""
    return make_interp_spline(*_knots(vol_surfaces, M), k=1, check_finite=False)(T)


def get_w(vol_surfaces, y, T, Cubic=True):
    """Interpolated implied variance times T, the total variance w of equation (3)."""
    if Cubic:
        return interp_cubic(vol_surfaces, y, T)*T
    return interp_linear(vol_surfaces, y, T)*T


def get_dw_dy(vol_surfaces, y, T, Cubic=True):
    f = partial(get_w, vol_surfaces, T=T, Cubic=Cubic)
    return _first_derivative(f, y, 1e-3)


def get_d2w_dy2(vol_surfaces, y, T, Cubic=True):
    f = partial(get_w, vol_surfaces, T=T, Cubic=Cubic)
    return _second_derivative(f, y, 1e-3)


def get_dw_dT(vol_surfaces, y, T, Cubic=True):
    f = partial(get_w, vol_surfaces, y, Cubic=Cubic)
    return _first_derivative(f, T, 1/252)


def get_local_vol(vol_surfaces, y, T, Cubic=True):
    """Gatheral's local variance, equation (3); Cubic picks the interpolation across maturity."""
    dw_dT = get_dw_dT(vol_surfaces, y, T, Cubic=Cubic)
    dw_dy = get_dw_dy(vol_surfaces, y, T, Cubic=Cubic)
    d2w_dy2 = get_d2w_dy2(vol_surfaces, y, T, Cubic=Cubic)
    w = get_w(vol_surfaces, y, T, Cubic=Cubic)
    return dw_dT / (1-y/w*dw_dy+1/4*(-1/4-1/w+(y**2)/(w**2))*(dw_dy)**2+d2w_dy2/2)
