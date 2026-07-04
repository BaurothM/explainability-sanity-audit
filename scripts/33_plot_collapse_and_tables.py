# -*- coding: utf-8 -*-
"""
Diagnostic companion to scripts/32:
  1. exact median tables   (model x method_family x stage_name) for all metrics
  2. exact IQR-width tables (q75 - q25) for all metrics
  3. collapse plot: fraction of samples with exact-zero Spearman per
     (model, method_family, stage), as a 1x3 grid mirroring scripts/32.

The fraction-of-exact-zero is a *proxy* for the constant-heatmap fraction:
  - spearmanr returns NaN iff one of the inputs has zero variance (constant).
  - scripts/31 maps that NaN -> 0.0 (documented convention).
  - A non-constant heatmap producing spearman == 0.0 *exactly* is, in
    practice, vanishingly rare among 1000 samples.
  - But: an *almost* constant heatmap (e.g. only a handful of nonzero
    pixels) gives some small non-zero Spearman and is not counted by this
    proxy, even though it is informationally dead. The proxy is therefore
    a lower bound on the true collapse fraction.
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

METRICS = ["spearman", "ssim", "hog"]

MODELS       = ["resnet50", "vit_base", "mobilevit_s"]
MODEL_LABEL  = {"resnet50":    "ResNet50",
                "vit_base":    "ViT-B/16",
                "mobilevit_s": "MobileViT-S"}

FAMILY_ORDER  = ["IntegratedGradients", "GradCAM", "LRP", "Random"]
FAMILY_COLORS = {
    "IntegratedGradients": "#1f77b4",
    "GradCAM":             "#ff7f0e",
    "LRP":                 "#2ca02c",
    "Random":              "#7f7f7f",
}


# --- 2. Load ---------------------------------------------------------------
df = pd.read_parquet(METRICS_PATH)
print(f"Loaded {METRICS_PATH.name}  shape={df.shape}\n")


# --- 3. Exact median + IQR tables -----------------------------------------
# Pivoted by stage_name (the human-readable label). sort=False preserves the
# per-model stage ordering present in the dataframe. Since the dataframe
# was sorted by (model, family, stage_idx) upstream, stages appear in
# cascading order *within each model*. The NaNs in the cross-product cells
# (e.g. ResNet stages for ViT rows) are expected and visible at a glance.
def print_pivot(values, title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    piv = df.pivot_table(
        index=["model", "method_family"],
        columns="stage_name",
        values=values,
        aggfunc=lambda s: s.quantile(0.50) if title.startswith("Median")
                          else s.quantile(0.75) - s.quantile(0.25),
        sort=False,
    )
    print(piv.round(3).to_string())

for metric in METRICS:
    print_pivot(metric, f"Median {metric}  (q50)")
for metric in METRICS:
    print_pivot(metric, f"IQR width {metric}  (q75 - q25)")


# --- 4. Collapse fraction --------------------------------------------------
# Fraction of samples with spearman == 0.0 exactly, per cell.
# (Proxy for constant-heatmap fraction; see module docstring.)
df["is_zero"] = (df["spearman"] == 0.0)
collapse = (df
            .groupby(["model", "method_family", "stage_idx", "stage_name"])
            ["is_zero"]
            .mean()
            .reset_index()
            .rename(columns={"is_zero": "collapse_frac"}))

# Anchor: at stage_idx = -1 ("original") no randomization has happened,
# so by construction zero samples collapse. We add this so the collapse
# plot starts at 0 on the left and rises rightwards, mirroring the
# 1.0-anchored decay of scripts/32.
anchor_rows = []
for row in (collapse[["model", "method_family"]]
            .drop_duplicates()
            .itertuples(index=False)):
    anchor_rows.append({
        "model": row.model, "method_family": row.method_family,
        "stage_idx": -1, "stage_name": "original", "collapse_frac": 0.0,
    })
collapse = pd.concat([pd.DataFrame(anchor_rows), collapse], ignore_index=True)
collapse = collapse.sort_values(["model", "method_family", "stage_idx"]).reset_index(drop=True)

# Print the collapse table too, since the plot won't show fine differences
# at the high end (e.g. 0.987 vs 0.994).
print(f"\n{'='*78}")
print("Collapse fraction  (samples with spearman == 0.0 exactly)")
print(f"{'='*78}")
piv = collapse.pivot_table(
    index=["model", "method_family"], columns="stage_name",
    values="collapse_frac", sort=False,
)
print(piv.round(3).to_string())


# --- 5. Per-model stage axis (reuses the same idea as scripts/32) ----------
def stage_axis_for(model):
    sub = (collapse[collapse["model"] == model][["stage_idx", "stage_name"]]
           .drop_duplicates()
           .sort_values("stage_idx"))
    return sub["stage_idx"].tolist(), sub["stage_name"].tolist()


# --- 6. Plot: 1 row x 3 columns, mirroring scripts/32 ----------------------
# y-axis fixed at [0, 1]: showing only the populated range would visually 
# exaggerate the collapse, which is in fact a minority phenomenon even 
# where it occurs.
fig, axes = plt.subplots(
    nrows=1, ncols=3,
    figsize=(12, 4.0),
    sharey=True,
)

for c, model in enumerate(MODELS):
    ax = axes[c]
    stage_idxs, stage_names = stage_axis_for(model)
    x_pos = {s: i for i, s in enumerate(stage_idxs)}

    for family in FAMILY_ORDER:
        sub = (collapse[(collapse["model"] == model) &
                        (collapse["method_family"] == family)]
               .sort_values("stage_idx"))
        if sub.empty:
            continue
        x = [x_pos[s] for s in sub["stage_idx"]]
        y = sub["collapse_frac"].to_numpy()
        ax.plot(x, y, color=FAMILY_COLORS[family],
                marker="o", markersize=4,
                linestyle="-.", linewidth=1.5)

    ax.axvline(x_pos[0], color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(stage_names, rotation=45, ha="right", fontsize=8)
    ax.set_title(MODEL_LABEL[model], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(axis="y", labelsize=8)
    if c == 0:
        ax.set_ylabel("Fraction Spearman = 0", fontsize=10)

legend_handles = [
    Line2D([0], [0], color=FAMILY_COLORS[f], linestyle="-.",
           marker="o", markersize=5, linewidth=1.5, label=f)
    for f in FAMILY_ORDER
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=len(FAMILY_ORDER), frameon=True, fontsize=10,
           bbox_to_anchor=(0.5, -0.05))

fig.tight_layout(rect=(0, 0.06, 1, 1.0))
OUT = FIGURES_DIR / "33_collapse_fraction_1x3.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved collapse plot to: {OUT}")
plt.show()