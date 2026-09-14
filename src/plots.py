"""Figures and the summary table for the report."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import (DROP_IMPLIED, DROP_LOCAL, FIGURE_DIR, MONEYNESS_WINDOW,
                    SELECTED_FAMILY)

# Okabe-Ito, paired with dash patterns so the figures survive greyscale.
BLUE, ORANGE = "#0072B2", "#E69F00"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

FAMILIES = (("implied_cubic", "Implied, cubic"), ("implied_linear", "Implied, linear"),
            ("dupire_cubic", "Dupire, cubic"), ("dupire_linear", "Dupire, linear"),
            ("gatheral_cubic", "Gatheral, cubic"), ("gatheral_linear", "Gatheral, linear"))


def _save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name)
    plt.close(fig)


def curves(fitted):
    """Fitted curves across forward moneyness, six panels on one page.

    One row per index, implied on the left and local on the right, with the
    rejected maturities marked.
    """
    moneyness = np.linspace(0.5, 1.8, 300)
    y = np.log(moneyness)
    fig, axes = plt.subplots(len(fitted), 2, figsize=(6.6, 2.5 * len(fitted)),
                             sharex=True, sharey="row")
    for row, (name, entry) in enumerate(fitted.items()):
        for column, (key, dropped, title) in enumerate(
                (("implied_all", DROP_IMPLIED[name], "Implied"),
                 ("local_all", DROP_LOCAL[name], "Local, Dupire"))):
            ax = axes[row, column]
            for fit, maturity in entry[key]:
                rejected = round(maturity, 3) in dropped
                colour, dash = (ORANGE, "--") if rejected else (BLUE, "-")
                with np.errstate(invalid="ignore"):
                    ax.plot(moneyness, np.sqrt(fit(y)), color=colour, linestyle=dash,
                            linewidth=0.9, alpha=0.9)
            ax.set_title(f"{name}, {title}", fontsize=9)
            if row == len(fitted) - 1:
                ax.set_xlabel("Forward moneyness")
        axes[row, 0].set_ylabel("Volatility")
    axes[0, 0].plot([], [], color=BLUE, linestyle="-", label="Retained")
    axes[0, 0].plot([], [], color=ORANGE, linestyle="--", label="Rejected")
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.subplots_adjust(hspace=0.28, wspace=0.12)
    _save(fig, "curves.png")


def surfaces(name, families, maturity_floor=0.07):
    """The six candidate surfaces, with the region of negative variance left open."""
    low, high = MONEYNESS_WINDOW
    fig = plt.figure(figsize=(7.4, 5.2))
    for position, (key, label) in enumerate(FAMILIES, start=1):
        frame = families[key]
        window = frame.loc[frame.index >= maturity_floor,
                           (frame.columns >= low) & (frame.columns <= high)]
        moneyness, maturity = np.meshgrid(window.columns.to_numpy(), window.index.to_numpy())
        ax = fig.add_subplot(2, 3, position, projection="3d")
        ax.plot_surface(moneyness, maturity, window.to_numpy(), cmap="viridis",
                        vmin=0, vmax=0.6, linewidth=0, antialiased=True,
                        rstride=4, cstride=4)
        ax.set_title(label, fontsize=8, pad=-4)
        ax.set_xlabel("Moneyness", fontsize=6, labelpad=-7)
        ax.set_ylabel("Maturity", fontsize=6, labelpad=-7)
        ax.tick_params(labelsize=5, pad=-3)
        ax.set_zlim(0, 0.6)
        if position % 3 != 0:
            ax.set_zticklabels([])
        ax.view_init(elev=24, azim=-60)
    fig.subplots_adjust(wspace=0.02, hspace=0.18, left=0.02, right=0.98)
    _save(fig, f"surfaces_{name.lower()}.png")


def selected_surface(chosen, maturity_floor=0.07):
    """The surface carried into the simulation, one three dimensional panel per index.

    Shown from the shortest traded maturity, below which the fitted local variance
    diverges and would dominate the vertical scale.
    """
    low, high = MONEYNESS_WINDOW
    fig = plt.figure(figsize=(7.4, 2.8))
    for position, (name, frame) in enumerate(chosen.items(), start=1):
        frame = frame.loc[frame.index >= maturity_floor,
                          (frame.columns >= low) & (frame.columns <= high)]
        moneyness, maturity = np.meshgrid(frame.columns.to_numpy(), frame.index.to_numpy())
        ax = fig.add_subplot(1, len(chosen), position, projection="3d")
        ax.plot_surface(moneyness, maturity, frame.to_numpy(), cmap="viridis",
                        linewidth=0, antialiased=True, rstride=4, cstride=4)
        ax.set_title(name, fontsize=9, pad=-2)
        ax.set_xlabel("Moneyness", fontsize=7, labelpad=-6)
        ax.set_ylabel("Maturity", fontsize=7, labelpad=-6)
        ax.tick_params(labelsize=6, pad=-3)
        ax.set_zlim(0, 0.6)
        if position != len(chosen):
            ax.set_zticklabels([])
        ax.view_init(elev=24, azim=-60)
    fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98)
    _save(fig, "selected_surface.png")


def summary_table(grid):
    """The grid as a booktabs table, printed for pasting into the report.

    Rates are shown as a percentage and as basis points so the fifteen columns
    fit the text width at a legible size. Both tables of the report are static
    LaTeX, so this prints the block rather than writing it to main.tex.
    """
    body = []
    for row, values in grid.iterrows():
        cells = ["--" if np.isnan(v) else f"{v:.3f}" for v in values]
        body.append(f"{100 * row:.2f} & " + " & ".join(cells) + r" \\")
    header = " & ".join(f"{10000 * c:.1f}" for c in grid.columns)
    n = len(grid.columns)
    tex = "\n".join([
        r"\begin{table}[thbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{l" + "r" * n + "}", r"\toprule",
        r"& \multicolumn{%d}{c}{Minimum interest, basis points} \\" % n,
        r"\cmidrule(lr){2-%d}" % (n + 1),
        r"Maximum interest, \% & " + header + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}",
        r"\caption{Coupon strike that prices the note at 98 percent of face. "
        r"A dash marks a pair of coupon rates for which the search returns no result.}",
        r"\label{tab:summary}", r"\end{table}", ""])
    print(tex)


if __name__ == "__main__":
    import main
    spot, fitted = main.fit_curves()
    curves(fitted)
    shares = {}
    chosen = {}
    for name, entry in fitted.items():
        families = main.surface_families(entry)
        surfaces(name, families)
        shares[name] = {k: main.blank_share(v) for k in families for v in [families[k]]}
        chosen[name] = families[SELECTED_FAMILY]
        print(name, {k: round(v, 3) for k, v in shares[name].items()}, flush=True)
    grid = main.coupon_grid(main.project_paths(spot, fitted), fitted["SPX"]["rate"])
    selected_surface(chosen)
    summary_table(grid)
    print("figures:", sorted(p.name for p in FIGURE_DIR.iterdir()))
