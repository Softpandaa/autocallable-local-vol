# Calibrating the Coupon Strike of an Autocallable Note under Local Volatility

This study examines at which coupon strike, if any, a 2-year worst-of autocallable note on the Nikkei 225, the S&P 500 and the Hang Seng Index prices at 98% of issue. It is priced by Monte Carlo under a local volatility surface fitted to Bloomberg option chains, with a knock-out at 110% and a knock-in at 50% of initial spot.

## Findings

Six candidate surfaces are compared, the implied, the Dupire and the Gatheral variance under cubic and linear interpolation across maturity. Gatheral's mapping is the weakest, negative on 10.5% to 33.6% of the moneyness window. The Dupire local variance interpolated linearly, negative on at most 0.2% of the window, is carried into the simulation.

At the contractual coupon rates of 2% and 0.01%, no coupon strike prices the note at 98% of issue. The 2-year USD rate was 4.83%, so the 2% coupon falls well below the discount rate. When the two rates are treated as free, the coupon strike rises with both.

## Layout

```
data/     committed option chains, rate curves and spot history
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/NKY.xlsx`, `data/SPX.xlsx` and `data/HSI.xlsx` are option chains from Bloomberg. `data/JPY OIS (365).xlsx`, `data/USD OIS (360).xlsx` and `data/HKD OIS (365).xlsx` are overnight index swap curves quoted on a daily compounding basis. `data/spot.csv` is the daily close of the three indices over the 10 years to 11 November 2023, used for the correlation matrix and for the initial spot levels.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints the coupon strike grid of the report and the negative variance shares behind Table 2, and draws its five figures.
