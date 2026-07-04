# -*- coding: utf-8 -*-
"""
Analysis of Experiment 2 (pixel flipping) results.

Reads the curves and AUC parquets written by scripts/34 and produces:

    Tables (printed + saved to results/tables/):
        T1: Median AUC per (model, method, order)
        T2: Per-sample (LeRF - MoRF) diff distribution per (model, method),
            including sign-flip count and rate
        T3: Sign-flip raw counts (the underlying data of T2's flip columns,
            kept separate for clarity)

    Figures (PNG, saved to results/figures/):
        F1: Confidence-vs-fraction-masked curves
            (2 rows = MoRF/LeRF, 3 cols = models, median line + dashed IQR)
        F2: Box plot of per-sample (LeRF - MoRF) diff distribution
            per (model, method)
        F3: Two heatmaps side-by-side, MoRF | LeRF, of median AUC

Notes
-----
- All AUCs are trapezoidal over fraction in [0, 0.5], so the theoretical
  maximum is 0.5 (constant confidence = 1.0 over the interval).
- Method order in plots and tables follows MODEL_METHODS for consistency
  with scripts/27 and /31.
- Random is included as the noise floor, same convention as the cascading
  analysis (scripts/31).
- Figures have no suptitle by design. Subplot headers (model/order) are
  retained as small subplot titles since they identify the panel content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import TABLES_DIR, FIGURES_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Configuration ----------------------------------------------------
MANIFEST_STEM = "subset_500x2_seed42"

CURVES_PATH = TABLES_DIR / f"pixel_flipping__{MANIFEST_STEM}.parquet"
AUC_PATH    = TABLES_DIR / f"pixel_flipping__{MANIFEST_STEM}__auc.parquet"

T1_PATH = TABLES_DIR / f"pixel_flipping_T1_median_auc__{MANIFEST_STEM}.csv"
T2_PATH = TABLES_DIR / f"pixel_flipping_T2_diff_distribution__{MANIFEST_STEM}.csv"
T3_PATH = TABLES_DIR / f"pixel_flipping_T3_sign_flips__{MANIFEST_STEM}.csv"

F1_PATH = FIGURES_DIR / f"35_pixel_flipping_F1_curves__{MANIFEST_STEM}.png"
F2_PATH = FIGURES_DIR / f"35_pixel_flipping_F2_diff_boxplot__{MANIFEST_STEM}.png"
F3_PATH = FIGURES_DIR / f"35_pixel_flipping_F3_heatmap__{MANIFEST_STEM}.png"

MODEL_ORDER = ["resnet50", "vit_base", "mobilevit_s"]
MODEL_DISPLAY = {
    "resnet50":    "ResNet50",
    "vit_base":    "ViT-B/16",
    "mobilevit_s": "MobileViT-S",
}
METHOD_ORDER = ["IntegratedGradients", "GradCAM", "LRP", "Chefer-LRP", "Random"]
METHOD_DISPLAY = {
    "IntegratedGradients": "IG",
    "GradCAM":              "GradCAM",
    "LRP":                  "LRP",
    "Chefer-LRP":           "Chefer-LRP",
    "Random":               "Random",
}
ORDER_ORDER = ["MoRF", "LeRF"]

METHOD_COLORS = {
    "IntegratedGradients": "#1f77b4",
    "GradCAM":              "#ff7f0e",
    "LRP":                  "#2ca02c",
    "Chefer-LRP":           "#2ca02c",
    "Random":               "#888888",
}

# F1 IQR display
F1_IQR_STYLE = "band"     # "band" with custom percentiles below
F1_BAND_LOWER = 0.40
F1_BAND_UPPER = 0.60
F1_BAND_ALPHA = 0.20


# --- 2. Load data --------------------------------------------------------
print(f"Loading curves from: {CURVES_PATH}")
curves = pd.read_parquet(CURVES_PATH)
print(f"  shape: {curves.shape}, "
      f"unique samples: {curves['sample_id'].nunique()}")

print(f"\nLoading AUCs from:  {AUC_PATH}")
auc = pd.read_parquet(AUC_PATH)
print(f"  shape: {auc.shape}")


# --- 3. T1: Median AUC per (model, method, order) -----------------------
print(f"\n{'='*70}")
print("T1: Median AUC per (model, method, order)")
print(f"{'='*70}")

t1 = (auc
      .groupby(["model", "method", "order"])["auc"]
      .median()
      .reset_index()
      .pivot(index=["model", "method"], columns="order", values="auc")
      .reset_index())

t1["_model_rank"]  = t1["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
t1["_method_rank"] = t1["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
t1 = t1.sort_values(["_model_rank", "_method_rank"]).drop(
    columns=["_model_rank", "_method_rank"]
).reset_index(drop=True)

t1["gap_LeRF_minus_MoRF"] = t1["LeRF"] - t1["MoRF"]

print(t1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
t1.to_csv(T1_PATH, index=False)
print(f"\nSaved to: {T1_PATH}")


# --- 4. T2: Per-sample diff distribution + T3: Sign flips ----------------
auc_wide = auc.pivot_table(
    index=["sample_id", "model", "method"],
    columns="order",
    values="auc",
).reset_index()
auc_wide["diff"] = auc_wide["LeRF"] - auc_wide["MoRF"]
auc_wide["is_flipped"] = auc_wide["diff"] < 0

print(f"\n{'='*70}")
print("T2: Per-sample (LeRF - MoRF) diff distribution per (model, method)")
print(f"{'='*70}")

t2_rows = []
for (model, method), g in auc_wide.groupby(["model", "method"]):
    n = len(g)
    n_flipped = int(g["is_flipped"].sum())
    t2_rows.append({
        "model":        model,
        "method":       method,
        "n":            n,
        "median_diff":  g["diff"].median(),
        "iqr_lower":    g["diff"].quantile(0.25),
        "iqr_upper":    g["diff"].quantile(0.75),
        "min_diff":     g["diff"].min(),
        "max_diff":     g["diff"].max(),
        "n_flipped":    n_flipped,
        "pct_flipped":  100.0 * n_flipped / n,
    })
t2 = pd.DataFrame(t2_rows)

t2["_model_rank"]  = t2["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
t2["_method_rank"] = t2["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
t2 = t2.sort_values(["_model_rank", "_method_rank"]).drop(
    columns=["_model_rank", "_method_rank"]
).reset_index(drop=True)

print(t2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
t2.to_csv(T2_PATH, index=False)
print(f"\nSaved to: {T2_PATH}")

t3 = auc_wide[["sample_id", "model", "method", "MoRF", "LeRF",
               "diff", "is_flipped"]].copy()
t3.to_csv(T3_PATH, index=False)
print(f"T3 (raw per-sample diffs) saved to: {T3_PATH}")


# --- 5. F1: Confidence-vs-fraction curves -------------------------------
# 2 rows (MoRF, LeRF) x 3 cols (models). Median solid line + dashed IQR
# bounds per method. Extra vertical space between rows for breathing room.

print(f"\n{'='*70}")
print("F1: Building confidence-vs-fraction curves...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True,
                         gridspec_kw={"hspace": 0.35, "wspace": 0.10})

for row_idx, order in enumerate(ORDER_ORDER):
    for col_idx, model in enumerate(MODEL_ORDER):
        ax = axes[row_idx, col_idx]
        subset = curves[
            (curves["order"] == order) &
            (curves["model"] == model)
        ]

        methods_present = [
            m for m in METHOD_ORDER
            if m in subset["method"].unique()
        ]

        for method in methods_present:
            method_data = subset[subset["method"] == method]
            grouped = method_data.groupby("fraction")["confidence"]
            median = grouped.median()
            color  = METHOD_COLORS.get(method, "#000000")
            label  = METHOD_DISPLAY.get(method, method)

            # Median line: solid, full weight
            ax.plot(median.index, median.values, label=label,
                    color=color, linewidth=2.0, zorder=3)

            # IQR display            
            lower = grouped.quantile(F1_BAND_LOWER)
            upper = grouped.quantile(F1_BAND_UPPER)
            ax.fill_between(median.index, lower.values, upper.values,
                            color=color, alpha=F1_BAND_ALPHA, zorder=1)

        ax.set_title(f"{MODEL_DISPLAY[model]} · {order}",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Fraction masked")
        if col_idx == 0:
            ax.set_ylabel("GT softmax confidence")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

plt.savefig(F1_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {F1_PATH}")


# --- 6. F2: Box plot of per-sample diff distribution --------------------
print(f"\n{'='*70}")
print("F2: Building per-sample diff box plot...")
print(f"{'='*70}")

boxplot_data = auc_wide[["sample_id", "model", "method", "diff"]].copy()
boxplot_data = boxplot_data.dropna(subset=["diff"])

present_combos = []
for model in MODEL_ORDER:
    for method in METHOD_ORDER:
        if ((boxplot_data["model"] == model) &
            (boxplot_data["method"] == method)).any():
            present_combos.append((model, method))

boxplot_data["combo"] = (
    boxplot_data["model"] + " | " + boxplot_data["method"]
)
combo_order   = [f"{m} | {meth}" for (m, meth) in present_combos]
display_order = [
    f"{MODEL_DISPLAY[m]} | {METHOD_DISPLAY[meth]}"
    for (m, meth) in present_combos
]

fig, ax = plt.subplots(figsize=(13, 6))

sns.boxplot(
    data=boxplot_data,
    x="combo",
    y="diff",
    order=combo_order,
    hue="combo",
    hue_order=combo_order,
    palette=[METHOD_COLORS.get(meth, "#000000")
             for (_, meth) in present_combos],
    legend=False,
    showfliers=True,
    fliersize=2,
    ax=ax,
)

ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7,
           label="zero")
ax.set_xlabel("")
ax.set_ylabel("AUC(LeRF) - AUC(MoRF)")
ax.set_xticks(range(len(display_order)))
ax.set_xticklabels(display_order, rotation=45, ha="right")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(F2_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {F2_PATH}")


# --- 7. F3: Heatmaps of median AUC --------------------------------------
# Two panels (MoRF, LeRF), each a (model x method) heatmap of median AUC.
# Shared color scale [0, 0.5]. We use a 3-column GridSpec so that both
# heatmap panels get exactly the same width, placing the colorbar in its
# own dedicated column (not stolen from the right panel) ensures that the
# cells in both heatmaps render at identical sizes.

print(f"\n{'='*70}")
print("F3: Building median AUC heatmaps...")
print(f"{'='*70}")

fig = plt.figure(figsize=(13, 4.8))
gs = fig.add_gridspec(
    nrows=1, ncols=3,
    width_ratios=[1.0, 1.0, 0.04],   # two equal heatmaps + thin colorbar
    wspace=0.10,
)
ax_morf = fig.add_subplot(gs[0, 0])
ax_lerf = fig.add_subplot(gs[0, 1], sharey=ax_morf)
ax_cbar = fig.add_subplot(gs[0, 2])

def build_display_pivot(order_value: str) -> pd.DataFrame:
    pivot = (auc[auc["order"] == order_value]
             .groupby(["model", "method"])["auc"]
             .median()
             .reset_index()
             .pivot(index="model", columns="method", values="auc"))
    pivot = pivot.reindex(index=MODEL_ORDER, columns=METHOD_ORDER)
    pivot.index   = [MODEL_DISPLAY[m]  for m in pivot.index]
    pivot.columns = [METHOD_DISPLAY[m] for m in pivot.columns]
    return pivot

for ax, order, is_rightmost in [
    (ax_morf, "MoRF", False),
    (ax_lerf, "LeRF", True),
]:
    pivot = build_display_pivot(order)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
        # Attach the colorbar to our dedicated cbar axes only once
        cbar=is_rightmost,
        cbar_ax=ax_cbar if is_rightmost else None,
        cbar_kws={"label": "Median AUC"} if is_rightmost else None,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 11},
    )

    ax.set_title(order, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Method", labelpad=10)
    ax.set_ylabel("Model" if not is_rightmost else "")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Hide y-tick labels on the right panel (it shares y with the left).
    if is_rightmost:
        ax.tick_params(axis="y", labelleft=False)

# Colorbar styling: padding between tick numbers and "Median AUC" label
ax_cbar.yaxis.set_tick_params(pad=6)
ax_cbar.yaxis.label.set_size(11)
ax_cbar.yaxis.labelpad = 12

plt.savefig(F3_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {F3_PATH}")


# --- 8. Final summary ----------------------------------------------------
print(f"\n{'='*70}")
print("All outputs written.")
print(f"{'='*70}")
print(f"Tables:  {T1_PATH.name}")
print(f"         {T2_PATH.name}")
print(f"         {T3_PATH.name}")
print(f"Figures: {F1_PATH.name}")
print(f"         {F2_PATH.name}")
print(f"         {F3_PATH.name}")