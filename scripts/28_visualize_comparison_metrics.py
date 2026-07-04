# -*- coding: utf-8 -*-
"""
Aggregate and visualize the pairwise comparison metrics produced by
script 27 (comparison_metrics__<manifest_stem>.parquet).

Pipeline:
    1. Load Parquet, sanity-check structure (this stage).
    2. Aggregate to median + IQR per (axis, item_a, item_b, metric).
    3. Produce boxplots and violinplots per axis.

Aggregation output:
    results/tables/comparison_metrics_aggregated__<manifest_stem>.parquet

Figure output:
    results/figures/28_<axis>_<plotkind>.png   (six files)

Both boxplots and violinplots are generated, since each highlights different 
aspects of the distribution (boxplots: quartiles and outliers; violins: 
density shape).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import TABLES_DIR, RESULTS_DIR

import pandas as pd
import re

# --- 1. Configuration ---
MANIFEST_STEM = "subset_500x2_seed42"
PARQUET_PATH  = TABLES_DIR / f"comparison_metrics__{MANIFEST_STEM}.parquet"

# --- 2. Load Parquet ---
print(f"Loading: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
print(f"  shape:  {df.shape}")
print(f"  dtypes:\n{df.dtypes}")

# --- 3. First rows ---
print("\n--- head(8) ---")
print(df.head(8).to_string(index=False))

# --- 4. Structural overview ---
print("\n--- counts per axis ---")
print(df["axis"].value_counts().to_string())

print("\n--- metrics present ---")
print(df["metric"].value_counts().to_string())

print("\n--- pairs per axis (unique item_a/item_b combinations) ---")
pairs_per_axis = (df.groupby("axis")[["item_a", "item_b"]]
                    .apply(lambda g: g.drop_duplicates().shape[0]))
print(pairs_per_axis.to_string())

print("\n--- samples per pair (should be 1000 for every pair) ---")
samples_per_pair = (df.groupby(["axis", "item_a", "item_b", "metric"])
                      .size())
print(f"  min:    {samples_per_pair.min()}")
print(f"  max:    {samples_per_pair.max()}")
print(f"  median: {samples_per_pair.median()}")
if samples_per_pair.min() != samples_per_pair.max():
    print("  WARNING: unequal sample counts across pairs - investigate.")
else:
    print(f"  OK: every pair has {samples_per_pair.min()} samples.")

# --- 5. Sanity: exact zeros per (axis, metric) ---
# compute_spearman in script 27 returns 0.0 (not NaN) when an input is
# constant. compute_hog_corr does the same for low-std HOG descriptors.
# If many "synthetic zeros" are in the data, they would bias median and
# violinplot shape. Pre-flight check.
print("\n--- exact zeros per (axis, metric) ---")
exact_zero_mask = df["value"] == 0.0
zero_counts = (df.assign(is_zero=exact_zero_mask)
                 .groupby(["axis", "metric"])["is_zero"]
                 .agg(["sum", "count"]))
zero_counts["pct"] = 100 * zero_counts["sum"] / zero_counts["count"]
print(zero_counts.to_string())

# --- 6. Sanity: Random-pair medians ---
# Random pairs should land near 0 on cross_method and cross_model. 
# Display medians of all pairs where "Random" appears in either item label.
print("\n--- Random-involving pairs: median per (axis, item_a, item_b, metric) ---")
random_mask = df["item_a"].str.contains("Random") | df["item_b"].str.contains("Random")
random_subset = df[random_mask]
print(f"  rows involving Random: {len(random_subset)} of {len(df)}")
random_medians = (random_subset.groupby(["axis", "item_a", "item_b", "metric"])
                                ["value"].median()
                                .reset_index())
print(random_medians.to_string(index=False))

# --- 7. Aggregate to median + IQR per (axis, item_a, item_b, metric) ---
# Long-format input -> wide-ish aggregated table. Each row represents one
# pair-metric combination, columns hold the summary statistics over the
# 1000 samples. n_samples is included as a built-in consistency check
# (should be 1000 for every row, if not, something is off upstream).
print("\n--- Aggregating to median + IQR per (axis, item_a, item_b, metric) ---")

grouped = df.groupby(["axis", "item_a", "item_b", "metric"])["value"]

agg = grouped.agg(
    median="median",
    q25=lambda s: s.quantile(0.25),
    q75=lambda s: s.quantile(0.75),
    n_samples="count",
).reset_index()

agg["iqr"] = agg["q75"] - agg["q25"]

# Column order: identifiers first, then stats in a natural reading order.
agg = agg[["axis", "item_a", "item_b", "metric",
           "median", "q25", "q75", "iqr", "n_samples"]]

print(f"  shape: {agg.shape}  (expected: 36 pairs * 3 metrics = 108 rows)")
print(f"  n_samples range: {agg['n_samples'].min()} to {agg['n_samples'].max()}")

# --- 8. Save aggregated table + preview ---
AGG_OUTPUT_PATH = TABLES_DIR / f"comparison_metrics_aggregated__{MANIFEST_STEM}.parquet"
print(f"\nSaving aggregated table to: {AGG_OUTPUT_PATH}")
agg.to_parquet(AGG_OUTPUT_PATH, index=False)
print(f"  size: {AGG_OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

# Preview: show the highest and lowest median per axis (Spearman only,
# the most directly interpretable metric). This is a quick first look at
# which pairs are most/least similar in each axis -- not a finding yet,
# just a sanity preview before we plot.
print("\n--- Spearman median preview (top 3 and bottom 3 per axis) ---")
for axis_name in ["cross_method", "cross_model", "cross_target"]:
    sub = agg[(agg["axis"] == axis_name) & (agg["metric"] == "spearman")]
    sub = sub.sort_values("median", ascending=False)
    print(f"\n[{axis_name}] top 3 by median Spearman:")
    print(sub.head(3)[["item_a", "item_b", "median", "iqr"]].to_string(index=False))
    print(f"[{axis_name}] bottom 3 by median Spearman:")
    print(sub.tail(3)[["item_a", "item_b", "median", "iqr"]].to_string(index=False))
    
# --- 9. Plotting setup ---
import matplotlib.pyplot as plt
import numpy as np

from src.paths import FIGURES_DIR

# False for publication-style figures (caption carries context).
# True for self-describing standalone figures.
SHOW_SUPTITLE = False

# Metric display configuration. Y-ranges chosen to fit the empirically
# observed value range across all six plots (cross_method, cross_model,
# cross_target, plus their misclassified-only variants). We use one
# config for all plots so that all six are directly visually comparable
# on the same y-axis. This means the cross_method plot has some
# whitespace at the top of SSIM/HOG, which we accept in exchange for
# cross-axis comparability.
METRIC_CONFIG = {
    "spearman": {"label": "Spearman ρ", "ylim": (-1.0, 1.0), "zero_line": True},
    "ssim":     {"label": "SSIM",         "ylim": (-0.2, 1.0), "zero_line": True},
    "hog":      {"label": "HOG-Pearson",  "ylim": (-0.1, 1.0), "zero_line": True},
}

# Color coding by architecture. Labels of cross_method pairs are drawn
# in the architecture's color.
ARCH_COLORS = {
    "resnet50":    "#1f77b4",   # matplotlib default blue
    "vit_base":    "#ff7f0e",   # default orange
    "mobilevit_s": "#2ca02c",   # default green
}

# Display names for the legend. The dict keys above (resnet50, vit_base,
# mobilevit_s) are the internal identifiers used to match against the
# DataFrame's item_a/item_b strings. The display names below are the
# human-readable forms that appear in the legend of every figure.
ARCH_DISPLAY = {
    "resnet50":    "ResNet50",
    "vit_base":    "ViT-B/16",
    "mobilevit_s": "MobileViT-S",
}

def architectures_in_pair(item_a: str, item_b: str) -> list[str]:
    """Return the unique architectures referenced by a pair, in label order."""
    archs = []
    for item in (item_a, item_b):
        for arch in ARCH_COLORS:
            if item.startswith(arch):
                if arch not in archs:
                    archs.append(arch)
                break
    return archs

# --- 10. Plotting function ---
def plot_axis(df_long: pd.DataFrame,
              axis_name: str,
              plot_kind: str,
              output_path: Path,
              pair_order: list[tuple[str, str]] | None = None,
              subtitle_extra: str = "",
              n_per_arch: dict[str, int] | None = None) -> None:
    """
    One figure per (axis, plot_kind), with three subplots (one per metric).

    df_long      : the original long-format dataframe (not the aggregated one).
                   We need the per-sample values to draw box/violin shapes.
    axis_name    : "cross_method", "cross_model", or "cross_target".
    plot_kind    : "box" or "violin".
    output_path  : where the .png is saved.
    pair_order   : list of (item_a, item_b) tuples in display order.
                   If None, we use the natural pandas groupby order.
    subtitle_extra : extra string inserted into the suptitle (only used
                     if SHOW_SUPTITLE is True).
    n_per_arch   : optional dict mapping arch key to per-architecture sample
                   count, shown in the legend (used by the per-model
                   misclassified variant).
    """
    sub = df_long[df_long["axis"] == axis_name]
    if sub.empty:
        raise ValueError(f"No rows for axis={axis_name}")

    # Determine pair order if not provided.
    # Strategy: non-Random pairs first, sorted by median Spearman descending
    # (most-similar pairs leftmost). Random pairs at the end, also sorted
    # by median Spearman descending (cosmetic, all are near zero).
    if pair_order is None:
        all_pairs = (sub[["item_a", "item_b"]]
                     .drop_duplicates()
                     .apply(tuple, axis=1)
                     .tolist())

        # Compute median Spearman per pair for sorting
        spearman_sub = sub[sub["metric"] == "spearman"]
        medians = (spearman_sub.groupby(["item_a", "item_b"])["value"]
                                .median()
                                .to_dict())

        def is_random_pair(pair):
            return "Random" in pair[0] or "Random" in pair[1]

        non_random = [p for p in all_pairs if not is_random_pair(p)]
        random_pairs = [p for p in all_pairs if is_random_pair(p)]

        non_random.sort(key=lambda p: medians[p], reverse=True)
        random_pairs.sort(key=lambda p: medians[p], reverse=True)

        pair_order = non_random + random_pairs

    # Build display labels: shorter, multi-line for legibility.
    # "resnet50_IntegratedGradients" vs "resnet50_GradCAM" -> "resnet50\nIG vs GradCAM"
    # When models differ (cross_model), put both models in the label.
    def make_label(item_a: str, item_b: str) -> str:
        """Three-line label: methodA / vs / methodB. Architecture is shown via color."""
        def strip_arch(item):
            for arch in ARCH_COLORS:
                if item.startswith(arch + "_"):
                    return item[len(arch) + 1:]
            return item

        a = strip_arch(item_a).replace("_gt", "").replace("_pred", "")
        b = strip_arch(item_b).replace("_gt", "").replace("_pred", "")

        replacements = {
            "IntegratedGradients": "IG",
            "Chefer-LRP": "CheferLRP",
        }
        for old, new in replacements.items():
            a = a.replace(old, new)
            b = b.replace(old, new)
        return f"{a}\nvs\n{b}"

    labels = [make_label(a, b) for (a, b) in pair_order]
    n_pairs = len(pair_order)

    # Figure: 3 subplots horizontal, one per metric.
    # Width scales with number of pairs. Height fixed.
    fig_width = max(8, 0.8 * n_pairs)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width * 3, 6), sharex=True)

    for ax_idx, (metric_name, cfg) in enumerate(METRIC_CONFIG.items()):
        ax = axes[ax_idx]

        # Build a list of 1D arrays, one per pair, of all 1000 values.
        data_per_pair = []
        for (item_a, item_b) in pair_order:
            mask = ((sub["item_a"] == item_a) &
                    (sub["item_b"] == item_b) &
                    (sub["metric"] == metric_name))
            values = sub.loc[mask, "value"].to_numpy()
            data_per_pair.append(values)

        positions = np.arange(n_pairs) + 1

        if plot_kind == "box":
            ax.boxplot(data_per_pair, positions=positions,
                       widths=0.6, showfliers=True,
                       flierprops={"marker": ".", "markersize": 3,
                                   "markerfacecolor": "gray",
                                   "markeredgecolor": "gray", "alpha": 0.3})
        elif plot_kind == "violin":
            parts = ax.violinplot(data_per_pair, positions=positions,
                                  widths=0.8, showmedians=True,
                                  showextrema=False,
                                  quantiles=[[0.25, 0.75]] * len(data_per_pair))
            # Light grey violin bodies for visual calm
            for body in parts["bodies"]:
                body.set_facecolor("#a0a0a0")
                body.set_edgecolor("#404040")
                body.set_alpha(0.7)
            # Quartile lines (Q25, Q75) dashed to distinguish from median
            if "cquantiles" in parts:
                parts["cquantiles"].set_linestyle("--")
                parts["cquantiles"].set_linewidth(0.8)
        else:
            raise ValueError(f"Unknown plot_kind: {plot_kind}")

        # Zero reference line (Random baseline)
        if cfg["zero_line"]:
            ax.axhline(0.0, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6, zorder=0)

        ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["label"])
        ax.set_xticks(positions)
        # Labels are set in a loop so each label can take its architecture's
        # color. For cross_method, both methods share an architecture, so
        # the whole label gets one color.
        ax.set_xticklabels([""] * n_pairs)  # placeholder so set_xticks takes effect
        for pos, label_text, (item_a, item_b) in zip(positions, labels, pair_order):
            archs = architectures_in_pair(item_a, item_b)

            # In both cases (single and two architectures), render the
            # three lines as separate ax.text calls so that "vs" can be
            # kept black regardless of the architecture(s) involved.
            # This unifies the visual treatment of the middle line
            # across cross_method, cross_model, and cross_target plots.
            lines = label_text.split("\n")  # ["methodA", "vs", "methodB"]
            line_offsets = [-0.035, -0.075, -0.115]

            if len(archs) == 1:
                color_a = color_b = ARCH_COLORS[archs[0]]
            else:
                color_a = ARCH_COLORS[archs[0]]
                color_b = ARCH_COLORS[archs[1]]

            line_colors = [color_a, "black", color_b]
            for line, y_off, c in zip(lines, line_offsets, line_colors):
                ax.text(pos, y_off, line,
                        transform=ax.get_xaxis_transform(),
                        rotation=0, ha="center", va="top",
                        fontsize=8, color=c)
        ax.grid(axis="y", alpha=0.3)
        # Display names: Spearman with Greek rho, others uppercase
        display_names = {
            "spearman": "Spearman ρ",
            "ssim":     "SSIM",
            "hog":      "HOG-Pearson",
        }
        ax.set_title(display_names[metric_name], fontsize=12, fontweight="bold")
        
        # Architecture color legend, repeated per subplot for zoom robustness.
        # If n_per_arch is provided (currently only for the per-model
        # misclassified variant), each label gets a " (n=X)" suffix so
        # the per-architecture sample count is visible exactly where the
        # architecture color is read.
        # ARCH_DISPLAY translates the internal arch key (used to match the
        # DataFrame) into the human-readable form shown to the reader.
        if n_per_arch is not None:
            label_fn = lambda name: f"{ARCH_DISPLAY[name]} (n={n_per_arch[name]})"
        else:
            label_fn = lambda name: ARCH_DISPLAY[name]

        legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=color, markersize=8,
                                     label=label_fn(name))
                          for name, color in ARCH_COLORS.items()]
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.32),
                  ncol=len(ARCH_COLORS), frameon=False, fontsize=9)
    
    # Suptitle: prettier formulation with hyphenated axis name. If a
    # subtitle_extra is provided (e.g. for filtered variants), it is
    # inserted between the axis name and the plot-kind parenthetical.
    # Only rendered if SHOW_SUPTITLE is True (toggle at the top of Block 9).
    axis_display = axis_name.replace("_", "-")
    n_samples = len(data_per_pair[0])

    if subtitle_extra and n_per_arch is None:
        title = (f"{axis_display.capitalize()} comparison - {subtitle_extra} "
                 f"({plot_kind}plot, n={n_samples} per pair)")
    elif subtitle_extra and n_per_arch is not None:
        title = (f"{axis_display.capitalize()} comparison - {subtitle_extra} "
                 f"({plot_kind}plot)")
    else:
        title = (f"{axis_display.capitalize()} comparison "
                 f"({plot_kind}plot, n={n_samples} per pair)")

    if SHOW_SUPTITLE:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()
    plt.close(fig)

# --- 11. Generate plots for all three axes ---
print("\n--- Generating plots: cross_method ---")
for plot_kind in ["box", "violin"]:
    out_path = FIGURES_DIR / f"28_cross_method_{plot_kind}plot.png"
    plot_axis(df, "cross_method", plot_kind, out_path)

print("\n--- Generating plots: cross_model ---")
for plot_kind in ["box", "violin"]:
    out_path = FIGURES_DIR / f"28_cross_model_{plot_kind}plot.png"
    plot_axis(df, "cross_model", plot_kind, out_path)

print("\n--- Generating plots: cross_target ---")
for plot_kind in ["box", "violin"]:
    out_path = FIGURES_DIR / f"28_cross_target_{plot_kind}plot.png"
    plot_axis(df, "cross_target", plot_kind, out_path)
    
# --- 12. Supplementary analysis: misclassified-only cross_target ---
# The standard cross_target plots (Block 11) include all 1000 samples.
# For correctly classified samples, _gt and _pred targets coincide,
# producing Spearman = 1.0 spikes that dominate the distribution.
#
# Two supplementary filtered variants:
#   - all-three-wrong - only samples that ALL three models
#     misclassified (~101 samples). Same sample pool across plots,
#     so cross-model comparisons within cross_target are
#     strictly valid.
#   - per-model misclassified - each model's cross_target pair is
#     filtered to the sample subset that THIS model misclassified.
#     Different sample sizes per pair (132-206). Highest sample
#     count, but pools differ across pairs.
#
# Both variants reuse the same plot_axis function, with subtitle_extra
# annotation and (where appropriate) filtered dataframes.

print("\n--- Computing misclassified-sample pools ---")

# Read class_idx_gt and class_idx_pred from .npz files to find which
# samples each model misclassified. 

HEATMAPS_DIR_FOR_META = RESULTS_DIR / "heatmaps" / MANIFEST_STEM

misclass_per_model = {}
for model in ["resnet50", "vit_base", "mobilevit_s"]:
    misclassified = set()
    files = sorted(HEATMAPS_DIR_FOR_META.glob(f"sample_*_{model}.npz"))
    for f in files:
        with np.load(f) as data:
            if int(data["class_idx_gt"]) != int(data["class_idx_pred"]):
                sid = int(re.match(r"sample_(\d+)_", f.name).group(1))
                misclassified.add(sid)
    misclass_per_model[model] = misclassified
    print(f"  {model}: {len(misclassified)} misclassified samples")

# All-three-wrong set: intersection over all three models
misclass_all_three = (misclass_per_model["resnet50"]
                      & misclass_per_model["vit_base"]
                      & misclass_per_model["mobilevit_s"])
print(f"  misclassified by all three: {len(misclass_all_three)} samples")

# --- 12a: all-three-wrong filter ---
print("\n--- Generating plots: cross_target, misclassified by all three models ---")
df_b3 = df[df["sample_id"].isin(misclass_all_three)]
print(f"  filtered df shape: {df_b3.shape}")

for plot_kind in ["box", "violin"]:
    out_path = FIGURES_DIR / f"28_cross_target_misclass_all_three_{plot_kind}plot.png"
    plot_axis(df_b3, "cross_target", plot_kind, out_path,
              subtitle_extra="misclassified by all three models")

# --- 12b: per-model misclassified filter ---
# Each cross_target pair uses the misclassified-sample pool of the
# specific model it belongs to. Sample counts differ per model
# (132-206). They are shown in the architecture color legend.
# Random pairs use the same per-model pool as the other methods,
# so all pairs on one architecture share the same data basis.
print("\n--- Generating plots: cross_target, per-model misclassified ---")

model_to_misclass = {
    "resnet50":    misclass_per_model["resnet50"],
    "vit_base":    misclass_per_model["vit_base"],
    "mobilevit_s": misclass_per_model["mobilevit_s"],
}

# For each cross_target pair, identify the underlying model and keep
# only rows whose sample_id is in that model's misclassified pool.
cross_target_rows = df[df["axis"] == "cross_target"]
filtered_chunks = []
for (item_a, item_b), pair_df in cross_target_rows.groupby(["item_a", "item_b"]):
    # Both items in a cross_target pair share the same model. Extract it
    # from item_a (e.g. "resnet50_LRP_gt" -> "resnet50").
    model = None
    for m in ["resnet50", "vit_base", "mobilevit_s"]:
        if item_a.startswith(m):
            model = m
            break
    if model is None:
        raise ValueError(f"Could not infer model from item_a={item_a}")

    pool = model_to_misclass[model]
    filtered_chunks.append(pair_df[pair_df["sample_id"].isin(pool)])

df_b1 = pd.concat(filtered_chunks, ignore_index=True)
print(f"  filtered df shape: {df_b1.shape}")

# Sanity: confirm per-pair row counts. Each pair should now contain
# n_model * 3 metrics rows. (3 metrics: spearman, ssim, hog)
sanity = (df_b1.groupby(["item_a", "item_b"])
                .size()
                .reset_index(name="n_rows"))
print("\n  Per-pair row counts (should be n_model * 3 metrics):")
print(sanity.to_string(index=False))

# Build the per-architecture N dict for the legend
n_per_arch_b1 = {
    "resnet50":    len(model_to_misclass["resnet50"]),
    "vit_base":    len(model_to_misclass["vit_base"]),
    "mobilevit_s": len(model_to_misclass["mobilevit_s"]),
}

for plot_kind in ["box", "violin"]:
    out_path = FIGURES_DIR / f"28_cross_target_misclass_per_model_{plot_kind}plot.png"
    plot_axis(df_b1, "cross_target", plot_kind, out_path,
              subtitle_extra="per-model misclassified",
              n_per_arch=n_per_arch_b1)