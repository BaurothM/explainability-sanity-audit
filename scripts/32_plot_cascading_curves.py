# -*- coding: utf-8 -*-
"""
Cascading parameter-randomization curves (Experiment 1), parallel to
Adebayo et al. (2018), Model Parameter Randomization Test (cascading variant).

Reads the cascading metrics Parquet from scripts/31 and produces a 3x3 grid
(rows = Spearman / SSIM / HOG, columns = the three architectures). Each
curve shows median + IQR of (cascading-vs-trained) similarity across the
1000 samples, per method family, over the cascading stages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import TABLES_DIR, FIGURES_DIR

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --- 1. Configuration -----------------------------------------------------
MANIFEST_STEM = "subset_500x2_seed42"
METRICS_PATH  = TABLES_DIR / f"cascading_metrics__{MANIFEST_STEM}.parquet"

METRICS      = ["spearman", "ssim", "hog"]
# Row labels: match scripts/28 wording for cross-figure consistency.
METRIC_LABEL = {"spearman": "Spearman \u03c1",
                "ssim":     "SSIM",
                "hog":      "HOG-Pearson"}
# Red y=0 reference line: meaningful for signed correlations, not for SSIM.
METRIC_HAS_ZEROLINE = {"spearman": True, "ssim": False, "hog": True}

MODELS       = ["resnet50", "vit_base", "mobilevit_s"]
MODEL_LABEL  = {"resnet50":    "ResNet50",
                "vit_base":    "ViT-B/16",
                "mobilevit_s": "MobileViT-S"}

FAMILY_ORDER  = ["IntegratedGradients", "GradCAM", "LRP", "Random"]
FAMILY_COLORS = {
    "IntegratedGradients": "#1f77b4",   # blue
    "GradCAM":             "#ff7f0e",   # orange
    "LRP":                 "#2ca02c",   # green
    "Random":              "#7f7f7f",   # grey
}

# Cosmetic toggle. True=clean plot; False=self-describing.
SHOW_SUPTITLE = False


# --- 2. Load and aggregate -------------------------------------------------
df = pd.read_parquet(METRICS_PATH)

def quantile_summary(g):
    out = {}
    for m in METRICS:
        out[f"{m}_q25"] = g[m].quantile(0.25)
        out[f"{m}_q50"] = g[m].quantile(0.50)
        out[f"{m}_q75"] = g[m].quantile(0.75)
    return pd.Series(out)

agg = (df
       .groupby(["model", "method_family", "stage_idx", "stage_name"])[METRICS]
       .apply(quantile_summary)
       .reset_index())


# --- 3. Inject the 'original' anchor (stage_idx = -1, all metrics = 1.0) ---
# Baseline-vs-baseline: every metric equals 1.0 for every sample, zero spread.
# Every grid cell therefore starts at 1.0 on the left and decays rightwards,
# parallel to Adebayo's 'original' x-tick.
anchor_rows = []
for row in (agg[["model", "method_family"]]
            .drop_duplicates()
            .itertuples(index=False)):
    r = {"model": row.model, "method_family": row.method_family,
         "stage_idx": -1, "stage_name": "original"}
    for m in METRICS:
        r[f"{m}_q25"] = 1.0
        r[f"{m}_q50"] = 1.0
        r[f"{m}_q75"] = 1.0
    anchor_rows.append(r)

agg = pd.concat([pd.DataFrame(anchor_rows), agg], ignore_index=True)
agg = agg.sort_values(["model", "method_family", "stage_idx"]).reset_index(drop=True)


# --- 4. Per-model ordered stage labels for the x-axis ----------------------
def stage_axis_for(model):
    sub = (agg[agg["model"] == model][["stage_idx", "stage_name"]]
           .drop_duplicates()
           .sort_values("stage_idx"))
    return sub["stage_idx"].tolist(), sub["stage_name"].tolist()


# --- 5. Subplot drawer -----------------------------------------------------
def _draw_subplot(ax, model, metric, *, show_xticklabels):
    """Render one (model, metric) cell of the 3x3 grid."""
    stage_idxs, stage_names = stage_axis_for(model)
    x_pos = {s: i for i, s in enumerate(stage_idxs)}

    for family in FAMILY_ORDER:
        sub = (agg[(agg["model"] == model) &
                   (agg["method_family"] == family)]
               .sort_values("stage_idx"))
        if sub.empty:
            continue   # e.g. LRP absent on mobilevit_s -> simply no line
        x   = [x_pos[s] for s in sub["stage_idx"]]
        med = sub[f"{metric}_q50"].to_numpy()
        lo  = sub[f"{metric}_q25"].to_numpy()
        hi  = sub[f"{metric}_q75"].to_numpy()
        color = FAMILY_COLORS[family]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, med, color=color, marker="o", markersize=4,
                linestyle="-.", linewidth=1.5)

    if METRIC_HAS_ZEROLINE[metric]:
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    # Onset of randomization: at x_pos[0] ('fc' / 'head'), i.e. between
    # 'original' and the first randomized stage.
    ax.axvline(x_pos[0], color="black", linestyle="--", linewidth=1.0)

    ax.set_xticks(list(x_pos.values()))
    if show_xticklabels:
        ax.set_xticklabels(stage_names, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=8)


# --- 6. Assemble 3x3 grid --------------------------------------------------
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(11, 9),
    sharex="col", sharey="row",
)

for r, metric in enumerate(METRICS):
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        _draw_subplot(ax, model, metric, show_xticklabels=(r == 2))
        if r == 0:
            ax.set_title(MODEL_LABEL[model], fontsize=11, fontweight="bold")
        if c == 0:
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=10)


# --- 7. Shared legend below the grid (Adebayo-style) -----------------------
legend_handles = [
    Line2D([0], [0], color=FAMILY_COLORS[f], linestyle="-.",
           marker="o", markersize=5, linewidth=1.5, label=f)
    for f in FAMILY_ORDER
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(FAMILY_ORDER),
    frameon=True,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.01),
)

if SHOW_SUPTITLE:
    fig.suptitle("Cascading parameter randomization", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
else:
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))

OUT = FIGURES_DIR / "32_cascading_curves_3x3.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved 3x3 grid to: {OUT}")
plt.show()


# --- 8. Exact median tables for all models x metrics ---
for metric in METRICS:
    print(f"\n{'='*70}\nMedian {metric} per (model, stage)\n{'='*70}")
    piv = agg.pivot_table(index=["model", "method_family"],
                          columns="stage_name",
                          values=f"{metric}_q50",
                          sort=False)
    print(piv.round(3).to_string())


