import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})

res = pd.read_csv(f"{OUT}/results_all_designs.csv")
df = pd.read_csv(f"{OUT}/clean_with_oof.csv")
imp = pd.read_csv(f"{OUT}/permutation_importance.csv")

# Figure 1: R2 by evaluation design
designs = {"A. Random split + country ID (original)": "A. Random split\n+ country ID (original)",
           "B. Random split, no country ID": "B. Random split,\nno country ID",
           "D. Temporal holdout (train <=2010, test 2011-17)": "D. Future years\n(countries seen)",
           "E. Unseen countries + future years": "E. Unseen countries\n+ future years",
           "C. Country held out (5x5 GroupKFold)": "C. Unseen countries\n(5x5 grouped CV)"}
models = [("Ridge", BLUE, "o"), ("Random Forest", ORANGE, "s"), ("Gradient Boosting", AQUA, "D")]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ys = np.arange(len(designs))[::-1]
for (m, c, mk), off in zip(models, [0.18, 0, -0.18]):
    vals = [res[(res.design == d) & (res.model == m)].R2.iloc[0] for d in designs]
    sds = [res[(res.design == d) & (res.model == m)].get("R2_sd").iloc[0] for d in designs]
    ax.scatter(vals, ys + off, s=46, color=c, marker=mk, label=m, zorder=3, edgecolor="white", linewidth=1.2)
    for v, y0, sd in zip(vals, ys + off, sds):
        if not np.isnan(sd):
            ax.plot([v - sd, v + sd], [y0, y0], color=c, lw=2, zorder=2)
pers = res[res.model.str.startswith("Persistence")].R2.iloc[0]
ax.scatter([pers], [ys[2] - 0.36], marker="|", s=160, color=INK, zorder=3)
ax.annotate(f"Persistence baseline\n(last observed value) {pers:.3f}", (pers, ys[2] - 0.36),
            xytext=(0.55, ys[2] - 0.42), fontsize=8, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_yticks(ys); ax.set_yticklabels(designs.values(), fontsize=9)
ax.set_xlim(0.3, 1.02); ax.set_xlabel("Test R² (higher is better)")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("Accuracy depends on whether test countries were seen in training",
             loc="left", fontsize=11, color=INK)
fig.savefig(f"{OUT}/fig1_r2_by_design.png"); plt.close(fig)

# Figure 2: observed vs predicted, country-held-out, country means
cm = df.groupby("Entity")[["Schizophrenia (%)", "oof_gb"]].mean()
fig, ax = plt.subplots(figsize=(4.6, 4.4))
lo, hi = 0.14, 0.39
ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
ax.scatter(cm["Schizophrenia (%)"], cm.oof_gb, s=26, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.8, zorder=2)
for name in ["China", "Japan", "Australia", "United States", "Somalia", "Maldives", "Taiwan", "Netherlands"]:
    if name in cm.index:
        x, yv = cm.loc[name]; ax.annotate(name, (x, yv), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Observed prevalence (%)"); ax.set_ylabel("Predicted prevalence (%)")
ax.set_title("Unseen-country predictions\n(gradient boosting, country means)", loc="left", fontsize=11, color=INK)
fig.savefig(f"{OUT}/fig2_observed_vs_predicted.png"); plt.close(fig)

# Figure 3: permutation importance (gradient boosting)
g = imp[imp.model == "Gradient Boosting"].sort_values("mean")
fig, ax = plt.subplots(figsize=(6, 3.8))
lab = g.feature.str.replace(" (%)", "", regex=False).str.replace("log_population", "Population (log)")
ax.barh(lab, g["mean"], xerr=g["std"], color=BLUE, height=0.6, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Drop in R² when feature is shuffled (unseen countries)")
ax.grid(axis="y", visible=False)
ax.set_title("Eating and bipolar disorder prevalence carry the signal", loc="left", fontsize=11, color=INK)
fig.savefig(f"{OUT}/fig3_permutation_importance.png"); plt.close(fig)

# Figure 4: country trajectories
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for e, s in df.groupby("Entity"):
    ax.plot(s.Year, s["Schizophrenia (%)"], color="#b9b8b2", lw=0.6, alpha=0.7)
w = df.groupby("Year")["Schizophrenia (%)"].mean()
ax.plot(w.index, w.values, color=BLUE, lw=2.2)
ax.annotate("Unweighted mean\nof 192 countries", (2017, w.iloc[-1]), xytext=(2009.5, 0.33), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Year"); ax.set_ylabel("Schizophrenia prevalence (%)")
ax.set_title("Countries differ a lot; each country barely changes", loc="left", fontsize=11, color=INK)
fig.savefig(f"{OUT}/fig4_country_trajectories.png"); plt.close(fig)
print("ok")
