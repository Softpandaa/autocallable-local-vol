# Calibrating the Coupon Strike of an Autocallable Note under Local Volatility

Calibrating the coupon strike of a two year multi-asset autocallable note on the Nikkei 225, the S&P 500 and the Hang Seng Index, priced by Monte Carlo under a local volatility surface fitted to Bloomberg option chains. The note settles on the weakest of the three indices, so the coupon, the knock-out at 110% of initial spot and the knock-in at 50% all depend on the worst performer.

## Findings

Six volatility surfaces were compared, being the implied, the Dupire and the Gatheral variance under cubic and linear interpolation across maturity. Gatheral's mapping is the weakest, leaving between 10.5% and 33.6% of the window with a negative variance. The Dupire local variance interpolated linearly leaves at most 0.2% of the traded moneyness window with a negative variance, and it is carried into the simulation.

No coupon strike prices the note at 98% of issue at the contractual rates of 2% maximum and 0.01% minimum interest, and the reason is the level of interest rates. The two year USD rate was 4.83% so that the coupon of 2% is well below the discount rate. Treating the two coupon rates as free variables, the coupon strike rises with both. 

## Layout

```
data/     committed option chains, rate curves and spot history
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/NKY.xlsx`, `data/SPX.xlsx` and `data/HSI.xlsx` are option chains from Bloomberg. `data/JPY OIS (365).xlsx`, `data/USD OIS (360).xlsx` and `data/HKD OIS (365).xlsx` are overnight index swap curves quoted on a daily compounding basis. `data/spot.csv` is the daily close of the three indices over the ten years to 11 November 2023, used for the correlation matrix and for the initial spot levels.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints the coupon strike grid of the report and the negative variance shares behind Table 1, and writes the five figures it uses to `latex/figures/`, which is created on first run and is not tracked.
