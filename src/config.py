"""Paths and parameters, each declared here once."""

from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LATEX_DIR = ROOT / "latex"
FIGURE_DIR = LATEX_DIR / "figures"

# Indices and data ------------------------------------------------------------
INDICES = ("NKY", "HSI", "SPX")
TICKER = {"NKY": "^N225", "HSI": "^HSI", "SPX": "^GSPC"}
# Cholesky order of Section 5; the factor, and so the paths, depend on it.
CORRELATION_ORDER = ("SPX", "HSI", "NKY")
SPOT_FILE = DATA_DIR / "spot.csv"
CHAIN_FILE = {name: DATA_DIR / f"{name}.xlsx" for name in INDICES}
OIS_FILE = {"NKY": DATA_DIR / "JPY OIS (365).xlsx",
            "HSI": DATA_DIR / "HKD OIS (365).xlsx",
            "SPX": DATA_DIR / "USD OIS (360).xlsx"}
OIS_BASIS = {"NKY": 365, "HSI": 365, "SPX": 360}  # beta of Section 3
STRIKE_STEP = {"NKY": 250, "HSI": 100, "SPX": 25}  # dK of equation (2) is 1% of it

# Surface fit -----------------------------------------------------------------
# Maturities (years, rounded to 3 decimals) of the curves dropped in Section 4.1.
DROP_IMPLIED = {"NKY": (), "HSI": (0.879, 1.627), "SPX": (1.611,)}
DROP_LOCAL = {"NKY": (),
              "HSI": (0.219, 0.299, 0.627, 0.879, 1.627),
              "SPX": (0.381, 0.633, 0.89, 0.94, 1.016, 1.189, 1.611)}
SELECTED_FAMILY = "dupire_linear"   # Section 4.2; drives Figure 1, not the paths

# Surface grids ---------------------------------------------------------------
MONEYNESS_GRID = (0.001, 4.0, 200)  # K/F: start, stop, count
MATURITY_GRID = (1e-5, 2.0, 200)    # years
MONEYNESS_WINDOW = (0.5, 1.5)       # K/F; Table 1 and the surface figures
MATURITY_FLOOR = 0.07               # years; the surface figures start here

# Note terms ------------------------------------------------------------------
NOTIONAL = 10_000                   # N, USD
TARGET_VALUE = 0.98 * NOTIONAL
HORIZON = 2                         # years
KNOCK_OUT = 1.1                     # B+, on observation dates
KNOCK_IN = 0.5                      # B-, on any day
OBSERVATION_DAYS = (126, 252, 378, 504)  # t_m, trading days

# Monte Carlo -----------------------------------------------------------------
PATHS = 10_000
TRADING_DAYS_PER_YEAR = 252
STEPS = HORIZON * TRADING_DAYS_PER_YEAR
SEEDS = (1, 2, 3)

# Coupon strike search --------------------------------------------------------
BRACKET_START = 1.0
BRACKET_STEP = 0.01
MAX_TRIALS = 100                    # bracket steps
TOLERANCE = 0.1                     # on |V(c) - TARGET_VALUE|, USD
MAX_ITERATIONS = 100                # bisection steps
MAX_INTEREST = (0.02, 0.20, 15)     # r+ of Table 2: start, stop, count
MIN_INTEREST = (0.0001, 0.01, 15)   # r- of Table 2
