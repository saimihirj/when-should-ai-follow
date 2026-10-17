"""Build the simulation figures of "When Should AI Follow?" from the CSVs written by
nkc_models.py.

    python nkc_models.py --outdir results                 # simulate (see README)
    python make_figures.py --results results --figdir figures

Figures produced (numbering follows the current version of the paper):

    Fig 1   baseline payoff of each memory regime alone (from the modular CSV)
    Fig 3   modular structure: joint payoff vs N_AI/N_Human by K_Human/K_AI
    Fig 4   AI-to-Human structure: joint payoff vs N_AI/N_Human by C/K_AI
    Fig 5   Human-to-AI structure: joint payoff by AI update mode and human payoff
    Fig 6   side-by-side comparison of the two sequences
    Fig 8   joint payoff and local peaks in the AI-to-Human structure
    Fig 9   heptile heat map, modular structure
    Fig 11  heptile heat maps, Human-to-AI structure (one per AI update mode)

Fig 10 (heptiles of K_Human / C in the AI-to-Human structure) is not produced: in that
structure the human window equals C and K_Human is always 0, so K_Human / C is not a
variable of the simulation (see README).

Aggregation conventions (all options below are command-line flags):
* x axis: relative scope N_AI / N_Human, restricted to <= --max-ratio (10) for the
  modular and AI-to-Human panels (Figs 3, 4, 8 and Fig 6A); the Human-to-AI panels
  (Fig 5 and Fig 6B) use the full range.
* "Raw" panels: mean joint payoff at each distinct ratio, then a centred rolling mean
  over --roll neighbouring ratios. Fig 5 raw panel: means within integer ratio bins.
* "Smoothed" panels: least-squares polynomial fit to all rows (degree 2 for Figs 3, 4, 8;
  degree 3 for Figs 5, 6). These are polynomial fits, not splines, and can leave the
  [0, 1] range at the edges of the data.
* Human-to-AI "high-performing human": PO_Human > --h-threshold (0.42 by default).
* Heptiles: computed over parameter combinations. Rows are heptiles of N_AI/N_Human;
  columns are heptiles of the column variable by rank, so every column holds one
  seventh of the combinations. Cells show the mean over all runs in them. Rows
  with K_AI = 0 (undefined K_Human/K_AI) are excluded from Fig 9.

The CSVs are read in chunks and averaged per parameter combination, so memory use does
not grow with the number of iterations. Every figure is saved as vector PDF and 300-dpi
PNG, and the key numbers behind each figure are written to summary.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILES = {
    "modular": "modular_simulation_results.csv",
    "ath": "sequential_simulation_results_ath.csv",
    "hta": "simulation_results_hta.csv",
}
COLS = {
    "modular": ["N_AI", "K_AI", "N_Human", "K_Human", "LP_AI", "PO_AI", "LP_Human",
                "PO_Human", "APO_Modular"],
    "ath": ["N_AI", "K_AI", "N_Human", "C", "LP_AI", "PO_AI", "LP_Human", "PO_Human",
            "APO_Sequential_ATH"],
    "hta": ["N_AI", "N_Human", "K_Human", "C", "NLP_AI", "PO_AI", "NLP_Human",
            "PO_Human", "APO", "AI_Updation_Type"],
}

SINGLE_COL, DOUBLE_COL = 3.5, 7.2          # inches
CMAP = plt.get_cmap("viridis")
STYLES = [  # colour (viridis, colour-blind safe), line style, marker
    (CMAP(0.00), "-", "o"), (CMAP(0.35), "--", "s"),
    (CMAP(0.65), "-.", "^"), (CMAP(0.88), ":", "D"),
]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 12, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.linewidth": 0.8, "lines.linewidth": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def panel_label(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")


def save(fig, figdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


KEYS = {
    "modular": ["N_AI", "K_AI", "N_Human", "K_Human"],
    "ath": ["N_AI", "K_AI", "N_Human", "C", "hi"],
    "hta": ["N_AI", "N_Human", "K_Human", "C", "AI_Updation_Type", "hi"],
}


def load(results, case, h_threshold, chunksize=1_000_000):
    """Read a CSV in chunks and return one row per parameter combination (and, for the
    sequenced structures, per human-payoff group), holding mean values and the number of
    runs ``n``. With one iteration every row is a single run and ``n`` is 1."""
    path = os.path.join(results, FILES[case])
    values = [c for c in COLS[case] if c not in KEYS[case] and c != "AI_Updation_Type"]
    sums = None
    for chunk in pd.read_csv(path, usecols=COLS[case], chunksize=chunksize):
        if case != "modular":
            chunk["hi"] = chunk["PO_Human"] > h_threshold
        chunk["n"] = 1
        g = chunk.groupby(KEYS[case], sort=False)[values + ["n"]].sum()
        sums = g if sums is None else sums.add(g, fill_value=0)
    df = sums.reset_index()
    for c in values:
        df[c] = df[c] / df["n"]
    df["ratio"] = df["N_AI"] / df["N_Human"]
    return df


def wmean(d, col):
    return float(np.average(d[col], weights=d["n"]))


def collapse(d, keys, cols):
    """Weighted means of ``cols`` within ``keys``, keeping the run count ``n``."""
    t = d[keys + cols].copy()
    for c in cols:
        t[c] = t[c] * d["n"]
    t["n"] = d["n"]
    g = t.groupby(keys, sort=False).sum().reset_index()
    for c in cols:
        g[c] = g[c] / g["n"]
    return g


def raw_curve(d, ycol, roll):
    """Mean of ``ycol`` at each distinct ratio (weighted by runs), then a rolling mean."""
    t = pd.DataFrame({"x": d["ratio"], "yn": d[ycol] * d["n"], "n": d["n"]}).groupby("x").sum().sort_index()
    g = t["yn"] / t["n"]
    return g.index.values, g.rolling(roll, center=True, min_periods=1).mean().values


def poly_coef(d, ycol, deg):
    """Least-squares polynomial over all runs (weights sqrt(n) on the cell means give
    the same fit as using every run individually)."""
    return np.polyfit(d["ratio"].values, d[ycol].values, deg, w=np.sqrt(d["n"].values))


def poly_curve(d, ycol, deg, lo=None, hi=None, n=300):
    coef = poly_coef(d, ycol, deg)
    x = d["ratio"]
    xs = np.linspace(lo if lo is not None else x.min(), hi if hi is not None else x.max(), n)
    return xs, np.polyval(coef, xs)


def ratio_groups(df, num, den, defs):
    with np.errstate(divide="ignore", invalid="ignore"):
        q = df[num] / df[den]
    out = {}
    for label, op, thr in defs:
        if op is None:
            out[label] = df
        else:
            out[label] = df[op(q, thr)]
    return out


def two_panel_lines(groups, ycol, args, xlabel, ylabel, deg, lo=None, hi=None):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), sharey=False)
    for i, (label, d) in enumerate(groups.items()):
        c, ls, _ = STYLES[i % len(STYLES)]
        xr, yr = raw_curve(d, ycol, args.roll)
        axes[0].plot(xr, yr, color=c, ls=ls, label=label)
        xs, ys = poly_curve(d, ycol, deg, lo, hi)
        axes[1].plot(xs, ys, color=c, ls=ls, label=label)
    for ax, letter, title in zip(axes, "AB", ("Simulation means", f"Polynomial fit (degree {deg})")):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        panel_label(ax, letter)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, axes


def heatmap(ax, table, xlabel, ylabel, title, cbar_label):
    data = table.values.astype(float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", origin="upper")
    for (r, c), v in np.ndenumerate(data):
        txt = "empty" if np.isnan(v) else f"{v:.3f}"
        shade = im.norm(v) if not np.isnan(v) else 0.5
        ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                color="black" if shade > 0.55 else "white")
    ax.set_xticks(range(table.shape[1])); ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0])); ax.set_yticklabels(table.index)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cbar_label)


def weighted_grid(d, value):
    t = pd.DataFrame({"row": d["row"], "col": d["col"], "yn": d[value] * d["n"], "n": d["n"]})
    g = t.groupby(["row", "col"]).sum()
    return (g["yn"] / g["n"]).unstack("col")


def heptiles(series, by_rank):
    s = series.rank(method="first") if by_rank else series
    return pd.qcut(s, 7, labels=False, duplicates="drop") + 1


# --------------------------------------------------------------------------- figures
def fig1(mod, args, out):
    ai = collapse(mod, ["N_AI", "K_AI"], ["PO_AI"]).set_index(["N_AI", "K_AI"])["PO_AI"].sort_index()
    hu = collapse(mod, ["N_Human", "K_Human"], ["PO_Human"]).set_index(["N_Human", "K_Human"])["PO_Human"].sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))
    for i, k in enumerate((3, 10, 20)):
        s = ai.xs(k, level="K_AI"); s = s[s.index >= 3]
        c, ls, _ = STYLES[i]; axes[0].plot(s.index, s.values, color=c, ls=ls, label=f"K = {k}")
    for i, k in enumerate((2, 3, 4)):
        s = hu.xs(k, level="K_Human"); s = s[s.index >= 3]
        c, ls, _ = STYLES[i]; axes[1].plot(s.index, s.values, color=c, ls=ls, label=f"K = {k}")
    for ax, letter, t in zip(axes, "AB", ("Uniform-memory (AI archetype)", "Recency-weighted (human archetype)")):
        ax.set_xlabel("Task scope, N"); ax.set_ylabel("Mean payoff"); ax.set_title(t)
        ax.set_ylim(0, 0.8); ax.legend(frameon=False); panel_label(ax, letter)
    fig.tight_layout(); save(fig, args.figdir, "fig01_baseline_regimes")
    out["fig1_UM_payoff_N47_by_K"] = {int(k): round(float(ai.loc[(47, k)]), 3) for k in range(1, 11)}


def fig3(mod, args, out):
    d = mod[mod["ratio"] <= args.max_ratio]
    groups = ratio_groups(d, "K_Human", "K_AI", [
        ("No constraint", None, None), ("$K_H/K_{AI}$ < 0.5", np.less, 0.5),
        ("$K_H/K_{AI}$ > 0.5", np.greater, 0.5), (f"$K_H/K_{{AI}}$ > {args.kh_high:g}", np.greater, args.kh_high)])
    fig, axes = two_panel_lines(groups, "APO_Modular", args, "$N_{AI}/N_H$", "Average payoff (APO)", 2,
                                hi=args.max_ratio)
    save(fig, args.figdir, "fig03_modular")
    out["fig3_quadfit_peak"] = {}
    for label, g in groups.items():
        xs, ys = poly_curve(g, "APO_Modular", 2, hi=args.max_ratio)
        out["fig3_quadfit_peak"][label] = {"ratio": round(float(xs[ys.argmax()]), 2),
                                           "APO": round(float(ys.max()), 3),
                                           "quadratic_coef": round(float(poly_coef(g, "APO_Modular", 2)[0]), 5)}


def ath_groups(ath, args):
    d = ath[ath["ratio"] <= args.max_ratio]
    return ratio_groups(d, "C", "K_AI", [
        ("$C/K_{AI}$ < 0.5", np.less, 0.5), ("$C/K_{AI}$ > 0.5", np.greater, 0.5),
        ("$C/K_{AI}$ > 2", np.greater, 2)])


def fig4(ath, args, out):
    groups = ath_groups(ath, args)
    fig, _ = two_panel_lines(groups, "APO_Sequential_ATH", args, "$N_{AI}/N_H$", "Average payoff (APO)", 2,
                             hi=args.max_ratio)
    save(fig, args.figdir, "fig04_ai_to_human")
    out["fig4_quadfit_max"] = {}
    for label, g in groups.items():
        xs, ys = poly_curve(g, "APO_Sequential_ATH", 2, hi=args.max_ratio)
        out["fig4_quadfit_max"][label] = {"max": round(float(ys.max()), 3), "at_ratio_10": round(float(ys[-1]), 3)}


def hta_groups(hta, args):
    hi = hta["hi"]
    groups = {}
    for mode, name in (("rule-based", "UM"), ("hallucination", "Memoryless")):
        for flag, lab in ((False, "low-payoff H"), (True, "high-payoff H")):
            groups[f"{name}, {lab}"] = hta[(hta["AI_Updation_Type"] == mode) & (hi == flag)]
    return groups


def fig5(hta, args, out):
    groups = hta_groups(hta, args)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))
    for i, (label, d) in enumerate(groups.items()):
        c, ls, mk = STYLES[i]
        bins = np.ceil(d["ratio"]).clip(lower=1)
        t = pd.DataFrame({"b": bins, "yn": d["APO"] * d["n"], "n": d["n"]}).groupby("b").sum()
        b = t["yn"] / t["n"]
        b = b[b.index <= args.max_ratio]
        axes[0].plot(b.index, b.values, color=c, ls=ls, marker=mk, ms=4, label=label)
        xs, ys = poly_curve(d, "APO", 3)
        axes[1].plot(xs, ys, color=c, ls=ls, label=label)
    for ax, letter, t in zip(axes, "AB", ("Means by integer ratio bin", "Polynomial fit (degree 3)")):
        ax.set_xlabel("$N_{AI}/N_H$"); ax.set_ylabel("Average payoff (APO)"); ax.set_title(t); panel_label(ax, letter)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, args.figdir, "fig05_human_to_ai")
    out["fig5_group_means"] = {k: round(wmean(v, "APO"), 3) for k, v in groups.items()}
    out["fig5_group_sizes"] = {k: int(v["n"].sum()) for k, v in groups.items()}
    out["hta_mean_PO_Human"] = round(wmean(hta, "PO_Human"), 3)
    out["hta_mean_APO"] = round(wmean(hta, "APO"), 3)


def fig6(ath, hta, args, out):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))
    best = -1
    for i, (label, g) in enumerate(ath_groups(ath, args).items()):
        c, ls, _ = STYLES[i]
        xs, ys = poly_curve(g, "APO_Sequential_ATH", 2, hi=args.max_ratio)
        axes[0].plot(xs, ys, color=c, ls=ls, label=label); best = max(best, ys.max())
    groups = hta_groups(hta, args)
    for i, (label, d) in enumerate(groups.items()):
        c, ls, _ = STYLES[i]
        xs, ys = poly_curve(d, "APO", 3)
        axes[1].plot(xs, ys, color=c, ls=ls, label=label)
    avg = wmean(groups["UM, high-payoff H"], "APO")
    axes[0].axhline(best, color="0.4", lw=0.8, ls="--"); axes[1].axhline(avg, color="0.4", lw=0.8, ls="--")
    axes[0].text(0.98, best - 0.05, f"max APO = {best:.2f}", transform=axes[0].get_yaxis_transform(), fontsize=8, ha="right")
    axes[1].text(0.98, avg - 0.05, f"mean APO (UM | high-payoff H) = {avg:.2f}", transform=axes[1].get_yaxis_transform(), fontsize=8, ha="right")
    for ax, letter, t in zip(axes, "AB", ("AI-to-Human sequence", "Human-to-AI sequence")):
        ax.set_xlabel("$N_{AI}/N_H$"); ax.set_ylabel("Average payoff (APO)"); ax.set_ylim(0, 0.9)
        ax.set_title(t); ax.legend(frameon=False, fontsize=7); panel_label(ax, letter)
    fig.tight_layout(); save(fig, args.figdir, "fig06_sequence_comparison")
    hi_ath = ath["hi"]
    out["fig6"] = {
        "ATH_max_of_quadfit": round(float(best), 3),
        "HTA_mean_APO_UM_given_high_human": round(float(avg), 3),
        "like_for_like_ATH_mean_APO_given_high_human": round(wmean(ath[hi_ath], "APO_Sequential_ATH"), 3),
        "ATH_mean_APO_unconditional": round(wmean(ath, "APO_Sequential_ATH"), 3),
        "HTA_mean_APO_UM_unconditional": round(wmean(hta[hta["AI_Updation_Type"] == "rule-based"], "APO"), 3),
    }


def fig8(ath, args, out):
    groups = ath_groups(ath, args)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))
    for i, (label, g) in enumerate(groups.items()):
        c, ls, _ = STYLES[i]
        xs, ys = poly_curve(g, "APO_Sequential_ATH", 2, hi=args.max_ratio)
        axes[0].plot(xs, ys, color=c, ls=ls, label=label)
    sel = {k: v for k, v in groups.items() if "< 0.5" in k or "> 2" in k}
    for i, (label, g) in enumerate(sel.items()):
        g = g[g["ratio"] >= 1]
        for j, (col, who) in enumerate((("LP_AI", "AI"), ("LP_Human", "H"))):
            c, ls, _ = STYLES[(2 * i + j) % len(STYLES)]
            xs, ys = poly_curve(g, col, 2, lo=1, hi=args.max_ratio)
            axes[1].plot(xs, ys, color=c, ls=ls, label=f"{who}, {label}")
    for ax in axes:
        for v in (4, 6):
            ax.axvline(v, color="0.5", lw=0.8, ls="--")
    axes[0].set_ylabel("Average payoff (APO)"); axes[1].set_ylabel("Mean number of local peaks")
    for ax, letter in zip(axes, "AB"):
        ax.set_xlabel("$N_{AI}/N_H$"); ax.legend(frameon=False, fontsize=7); panel_label(ax, letter)
    fig.tight_layout(); save(fig, args.figdir, "fig08_local_peaks")
    band = ath[(ath["ratio"] > 4) & (ath["ratio"] < 6)]
    out["fig8"] = {
        "band_4_6_LP_AI_over_LP_Human": round(wmean(band, "LP_AI") / wmean(band, "LP_Human"), 2),
        "band_4_6_mean_N_AI_over_N_Human": round(wmean(band, "ratio"), 2),
    }


def fig9(mod, args, out):
    d = mod[mod["K_AI"] > 0].copy()
    d["kk"] = d["K_Human"] / d["K_AI"]
    d["row"] = heptiles(d["ratio"], by_rank=False)
    d["col"] = heptiles(d["kk"], by_rank=True)
    t = weighted_grid(d, "APO_Modular")
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.3))
    heatmap(ax, t, "Heptile of $K_H/K_{AI}$", "Heptile of $N_{AI}/N_H$", "Modular structure", "Mean joint payoff (APO)")
    fig.tight_layout(); save(fig, args.figdir, "fig09_heatmap_modular")
    out["fig9_grid"] = t.round(3).values.tolist()


def fig11(hta, args, out):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.4))
    out["fig11_grids"] = {}
    for ax, letter, (mode, name) in zip(axes, "AB", (("rule-based", "UM"), ("hallucination", "Memoryless"))):
        d = collapse(hta[hta["AI_Updation_Type"] == mode], ["N_AI", "N_Human", "K_Human", "C"], ["APO"])
        d["ratio"] = d["N_AI"] / d["N_Human"]
        d["row"] = heptiles(d["ratio"], by_rank=False)
        d["col"] = heptiles(d["C"], by_rank=True)
        t = weighted_grid(d, "APO")
        heatmap(ax, t, "Heptile of C", "Heptile of $N_{AI}/N_H$", f"Downstream AI: {name}", "Mean APO")
        panel_label(ax, letter)
        out["fig11_grids"][name] = t.round(3).values.tolist()
    fig.tight_layout(); save(fig, args.figdir, "fig11_heatmap_human_to_ai")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results", help="folder with the simulation CSVs")
    p.add_argument("--figdir", default="figures", help="output folder for figures")
    p.add_argument("--max-ratio", type=float, default=10.0, help="largest N_AI/N_Human shown in Figs 3, 4, 8")
    p.add_argument("--roll", type=int, default=15, help="rolling-mean width for raw panels (number of distinct ratios)")
    p.add_argument("--kh-high", type=float, default=2.0, help="threshold for the high K_Human/K_AI group in Fig 3")
    p.add_argument("--h-threshold", type=float, default=0.42, help="PO_Human above which the upstream human is high-performing")
    args = p.parse_args()
    os.makedirs(args.figdir, exist_ok=True); set_style()

    mod, ath, hta = (load(args.results, c, args.h_threshold) for c in ("modular", "ath", "hta"))
    out = {"iterations_detected": {"modular": int(mod["n"].sum()) / 70125, "ath": int(ath["n"].sum()) / 57375, "hta": int(hta["n"].sum()) / 18750},
           "settings": vars(args)}
    print("Building figures ...")
    fig1(mod, args, out); fig3(mod, args, out); fig4(ath, args, out); fig5(hta, args, out)
    fig6(ath, hta, args, out); fig8(ath, args, out); fig9(mod, args, out); fig11(hta, args, out)
    print("  Fig 10 skipped: K_Human is always 0 in the AI-to-Human structure (see README).")
    with open(os.path.join(args.figdir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote summary.json\nDone: {args.figdir}")


if __name__ == "__main__":
    main()
