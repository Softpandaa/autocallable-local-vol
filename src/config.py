"""Every parameter used by this project, declared once."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LATEX_DIR = ROOT / "latex"
FIGURE_DIR = LATEX_DIR / "figures"

INDICES = ("NKY", "HSI", "SPX")
TICKER = {"NKY": "^N225", "HSI": "^HSI", "SPX": "^GSPC"}

# The correlation matrix is factorised in this order, which is the column order
# the original download returned. The Cholesky factor of a reordered matrix is
# not a reordering of the factor, so the order fixes the realised paths and
# reproduces the factor printed in the report.
CORRELATION_ORDER = ("SPX", "HSI", "NKY")

SPOT_FILE = DATA_DIR / "spot.csv"
CHAIN_FILE = {name: DATA_DIR / f"{name}.xlsx" for name in INDICES}
OIS_FILE = {"NKY": DATA_DIR / "JPY OIS (365).xlsx",
            "HSI": DATA_DIR / "HKD OIS (365).xlsx",
            "SPX": DATA_DIR / "USD OIS (360).xlsx"}

# Day count of each overnight index swap curve.
OIS_BASIS = {"NKY": 365, "HSI": 365, "SPX": 360}

# Spacing between adjacent strikes in each chain, used as the step of the
# second difference in the Dupire numerator.
STRIKE_STEP = {"NKY": 250, "HSI": 100, "SPX": 25}

# Maturities dropped because the surface fit fails on them. Matched on the
# maturity in years rounded to three decimals.
DROP_IMPLIED = {"NKY": (), "HSI": (0.879, 1.627), "SPX": (1.611,)}
DROP_LOCAL = {"NKY": (),
              "HSI": (0.219, 0.299, 0.627, 0.879, 1.627),
              "SPX": (0.381, 0.633, 0.89, 0.94, 1.016, 1.189, 1.611)}

# Note terms.
NOTIONAL = 10_000
TARGET_VALUE = 0.98 * NOTIONAL     # issue at 98 percent of face
HORIZON = 2                        # years
KNOCK_OUT = 1.1                    # laggard level at a semi annual date
KNOCK_IN = 0.5                     # laggard level on any day
OBSERVATION_DAYS = (126, 252, 378, 504)

# Monte Carlo.
PATHS = 10_000
STEPS = 504                        # 252 trading days a year over the horizon
TRADING_DAYS_PER_YEAR = 252
SEEDS = (1, 2, 3)

# Coupon strike search, as Section 6 of the report specifies it. The bracket is
# found by stepping from a strike of one, then bisected until the note value is
# within TOLERANCE of the target. Exhausting either count returns no result.
BRACKET_START = 1.0
BRACKET_STEP = 0.01
MAX_TRIALS = 100
TOLERANCE = 0.1
MAX_ITERATIONS = 100

# Grid on which the volatility surfaces are evaluated for the appendix figures.
MONEYNESS_GRID = (0.001, 4.0, 200)   # start, stop, count, in moneyness K/F

# The surfaces are only supported where options trade. Outside this window the
# fitted curve is pure extrapolation, so both the figures and the count of
# negative variance are restricted to it.
MONEYNESS_WINDOW = (0.5, 1.5)
MATURITY_GRID = (1e-5, 2.0, 200)

# The six surface families the report compares. Dupire with linear interpolation
# across maturity is the one selected in Section 4.2 and used for the paths.
SELECTED_FAMILY = "dupire_linear"

# Grid of coupon rates reported in the summary table.
MAX_INTEREST = (0.02, 0.20, 15)     # start, stop, count
MIN_INTEREST = (0.0001, 0.01, 15)
