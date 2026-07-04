# -*- coding: utf-8 -*-
"""
Multi-seed validation for the cascading parameter randomization experiment.

The full cascading run (scripts/30) uses 1 random seed per stage on 1000
samples. This is methodologically defensible only if inter-seed variance
is small compared to inter-sample variance, otherwise a single seed is
an unreliable representative of the random-init distribution.

This script tests that claim empirically. From the 10-sample subset where
we kept 3 seeds (42, 43, 44), we compute for each (model, stage, method):
    Var[Spearman | fix sample, vary seed]    (inter-seed)
    Var[Spearman | fix seed, vary sample]    (inter-sample)
The Spearman is computed between the cascading heatmap and its trained-
baseline counterpart from scripts/22. If the inter-seed variance is 
consistently smaller than the inter-sample variance, the single-seed 
choice is empirically backed.

Output: console diagnostics + one figure
    results/figures/31a_multiseed_validation.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS_DIR, FIGURES_DIR

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict


# --- 1. Configuration -------------------------------------------------------
MANIFEST_STEM = "subset_500x2_seed42"
CASCADING_DIR = RESULTS_DIR / "heatmaps_cascading" / MANIFEST_STEM
BASELINE_DIR  = RESULTS_DIR / "heatmaps" / MANIFEST_STEM

# We use the 10-sample subset where we have 3 seeds.
SAMPLE_IDS = list(range(10))         # samples 0..9
SEEDS      = [42, 43, 44]
MODELS     = ["resnet50", "vit_base", "mobilevit_s"]

# Methods to validate per architecture (must match what's in the .npz files).
# LRP only on resnet50, Chefer-LRP only on vit_base, IG/GradCAM/Random everywhere.
METHODS_PER_MODEL = {
    "resnet50":    ["IntegratedGradients", "GradCAM", "LRP",        "Random"],
    "vit_base":    ["IntegratedGradients", "GradCAM", "Chefer-LRP", "Random"],
    "mobilevit_s": ["IntegratedGradients", "GradCAM",               "Random"],
}

# Number of cascading stages per model (matches src/randomization_schedule.py).
N_STAGES = 5


# --- 2. Helper: load + Spearman --------------------------------------------
def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman between two flattened heatmaps. If either input is constant,
    scipy returns nan. We map that to 0.0.
    """
    rho, _ = spearmanr(a.flatten(), b.flatten())
    if np.isnan(rho):
        return 0.0
    return float(rho)


def load_cascading_heatmap(sample_id, model, stage, seed, method):
    """Return the (H, W) heatmap, or None if file/method missing."""
    path = (CASCADING_DIR
            / f"sample_{sample_id:05d}_{model}_stage{stage}_seed{seed}.npz")
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if method not in data:
        return None
    return data[method]


def load_baseline_heatmap(sample_id, model, method):
    """
    Return the GT-target heatmap from the trained baseline (scripts/22),
    using the '_gt' suffix convention. Returns None if missing.
    """
    path = BASELINE_DIR / f"sample_{sample_id:05d}_{model}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    key = f"{method}_gt"
    if key not in data:
        return None
    return data[key]


# --- 3. Build the Spearman matrix ------------------------------------------
# spearman_values[model][method][stage] -> dict {(sample_id, seed): rho}
# from which we can compute both variances.

print(f"{'='*70}")
print("Loading heatmaps and computing Spearman vs trained baseline")
print(f"{'='*70}")
print(f"Samples: {SAMPLE_IDS}")
print(f"Seeds:   {SEEDS}")
print(f"Models:  {MODELS}")
print()

spearman_values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
n_missing = 0

for model in MODELS:
    methods = METHODS_PER_MODEL[model]
    print(f"  {model}: ", end="", flush=True)
    for sample_id in SAMPLE_IDS:
        for method in methods:
            baseline_hm = load_baseline_heatmap(sample_id, model, method)
            if baseline_hm is None:
                n_missing += 1
                continue
            for stage in range(N_STAGES):
                for seed in SEEDS:
                    cas_hm = load_cascading_heatmap(
                        sample_id, model, stage, seed, method
                    )
                    if cas_hm is None:
                        n_missing += 1
                        continue
                    rho = spearman_safe(cas_hm, baseline_hm)
                    spearman_values[model][method][stage][(sample_id, seed)] = rho
    print("done")

print(f"\nMissing entries: {n_missing}")


# --- 4. Compute inter-seed and inter-sample variance per (model, stage, method)
# Per (model, stage, method):
#   Inter-seed-variance: for each sample_id, take the 3 Spearman values
#     across seeds. Compute std over seeds. Then average those stds across
#     samples (so we have ONE number for the "typical inter-seed spread").
#   Inter-sample-variance: for each seed, take the 10 Spearman values
#     across samples. Compute std over samples. Then average those stds
#     across seeds.

print(f"\n{'='*70}")
print("Variance comparison per (model, stage, method)")
print(f"{'='*70}")
print(f"{'model':<14s} {'method':<22s} {'stage':<6s} "
      f"{'mean Spearman':>14s} {'inter-seed std':>16s} "
      f"{'inter-sample std':>18s} {'ratio (samp/seed)':>18s}")

# Store the summary numbers for later visualization
summary = []   # list of dicts

for model in MODELS:
    for method in METHODS_PER_MODEL[model]:
        for stage in range(N_STAGES):
            entries = spearman_values[model][method][stage]
            if not entries:
                continue

            # Build a 10x3 matrix: rows = samples, cols = seeds
            mat = np.full((len(SAMPLE_IDS), len(SEEDS)), np.nan)
            for (sid, sd), rho in entries.items():
                i = SAMPLE_IDS.index(sid)
                j = SEEDS.index(sd)
                mat[i, j] = rho

            # Skip if any rows/cols entirely missing (shouldn't happen, but defensive)
            if np.isnan(mat).any():
                continue

            # Inter-seed std: std along the seed axis, per row. Then mean over rows
            inter_seed_std = np.std(mat, axis=1, ddof=1).mean()
            # Inter-sample std: std along the sample axis, per col. Then mean over cols
            inter_sample_std = np.std(mat, axis=0, ddof=1).mean()
            mean_spearman = mat.mean()
            ratio = inter_sample_std / inter_seed_std if inter_seed_std > 0 else np.inf

            summary.append({
                "model": model,
                "method": method,
                "stage": stage,
                "mean_spearman": mean_spearman,
                "inter_seed_std": inter_seed_std,
                "inter_sample_std": inter_sample_std,
                "ratio": ratio,
            })

            print(f"{model:<14s} {method:<22s} {stage:<6d} "
                  f"{mean_spearman:>+14.4f} {inter_seed_std:>16.4f} "
                  f"{inter_sample_std:>18.4f} {ratio:>18.2f}")


# --- 5. Aggregate ratio: is inter-sample > inter-seed in general? -----------
ratios = np.array([s["ratio"] for s in summary if np.isfinite(s["ratio"])])
print(f"\n{'='*70}")
print("Summary of (inter-sample std) / (inter-seed std) ratios")
print(f"{'='*70}")
print(f"  N (model,stage,method) cells: {len(ratios)}")
print(f"  ratios > 1 (inter-sample dominant): "
      f"{(ratios > 1).sum()} / {len(ratios)} "
      f"({(ratios > 1).mean()*100:.0f}%)")
print(f"  median ratio: {np.median(ratios):.2f}")
print(f"  min / max:    {ratios.min():.2f} / {ratios.max():.2f}")
print()
print("Interpretation:")
print("  ratio > 1 means inter-sample variance dominates inter-seed variance,")
print("  i.e., choosing a single seed is methodologically safe at n=1000")
print("  because two random samples will differ more than two random seeds.")
print("  A ratio < 1 in any cell flags a (model, stage, method) where the")
print("  single-seed choice should be qualified.")


# --- 6. Visualization ------------------------------------------------------
# Scatter: x = inter-seed std, y = inter-sample std, one point per cell.
# Color by model, marker by method. The diagonal y=x is the equal-variance
# line. Points ABOVE it support the single-seed decision.

print("\nGenerating figure 31a_multiseed_validation.png...")

fig, ax = plt.subplots(figsize=(9, 8))

model_colors = {"resnet50": "C0", "vit_base": "C1", "mobilevit_s": "C2"}
method_markers = {
    "IntegratedGradients": "o",
    "GradCAM": "s",
    "LRP": "D",
    "Chefer-LRP": "D",       # same marker as LRP, they are the LRP family
    "Random": "x",
}

# Plot a y=x reference line first so it sits behind the points
lim_max = max(
    max(s["inter_seed_std"] for s in summary),
    max(s["inter_sample_std"] for s in summary),
) * 1.1
ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.4, lw=1,
        label="y = x  (equal variance)")

for s in summary:
    ax.scatter(
        s["inter_seed_std"], s["inter_sample_std"],
        color=model_colors[s["model"]],
        marker=method_markers[s["method"]],
        s=70, alpha=0.7, edgecolor="black", linewidth=0.5,
    )

ax.set_xlim(0, lim_max)
ax.set_ylim(0, lim_max)
ax.set_xlabel("Inter-seed std of Spearman\n"
              "(how much one image's heatmap changes across random seeds)")
ax.set_ylabel("Inter-sample std of Spearman\n"
              "(how much heatmaps differ across images at one seed)")
ax.set_title(
    f"Multi-seed validation - n={len(SAMPLE_IDS)} samples, {len(SEEDS)} seeds\n"
    f"Points above the diagonal: inter-sample variance dominates "
    f"(single-seed defensible).\n"
    f"{(ratios > 1).sum()}/{len(ratios)} cells above the line "
    f"(median ratio {np.median(ratios):.2f})."
)

# Manual two-part legend: colors for models, markers for methods
from matplotlib.lines import Line2D
color_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
           markeredgecolor="black", markersize=10, label=m)
    for m, c in model_colors.items()
]
marker_legend = [
    Line2D([0], [0], marker=mk, color="black", markerfacecolor="lightgray",
           markersize=10, label=meth)
    for meth, mk in method_markers.items()
]
leg1 = ax.legend(handles=color_legend, loc="upper left",
                 title="model", fontsize=9, title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=marker_legend, loc="lower right",
          title="method", fontsize=9, title_fontsize=9)

ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = FIGURES_DIR / "31a_multiseed_validation.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved figure to {out_path}")


# --- 7. Per-cell seed drift of the MEDIAN (not of individual samples) -----
# Cell-median drift across seeds: for each (model, method, stage) cell,
# how much does the median Spearman over the 10 samples shift between
# seeds 42 → 43 and 42 → 44?
#
# Per-sample drift is not the right quantity here, because the NaN→0.0
# convention for constant heatmaps can produce large per-sample jumps
# that reflect different failure modes rather than seed noise. The
# median over the 10 samples absorbs those failure-mode switches
# robustly.
#
# Caveat: with n=10 samples, the medians themselves are noisy. The
# thresholds below are heuristic guides for which result claims are
# supportable, not significance bounds.


print(f"\n{'='*70}")
print("Cell-median drift across seeds (the quantity that matters for findings)")
print(f"{'='*70}")
print(f"{'model':<14s} {'method':<22s} {'stage':<6s} "
      f"{'med@42':>8s} {'med@43':>8s} {'med@44':>8s} "
      f"{'drift_43':>9s} {'drift_44':>9s} {'drift_max':>10s}")

median_drift_records = []

for model in MODELS:
    for method in METHODS_PER_MODEL[model]:
        for stage in range(N_STAGES):
            entries = spearman_values[model][method][stage]
            if not entries:
                continue

            # 10x3 matrix: rows = samples, cols = seeds. Same as block 4.
            mat = np.full((len(SAMPLE_IDS), len(SEEDS)), np.nan)
            for (sid, sd), rho in entries.items():
                mat[SAMPLE_IDS.index(sid), SEEDS.index(sd)] = rho
            if np.isnan(mat).any():
                continue

            i42, i43, i44 = SEEDS.index(42), SEEDS.index(43), SEEDS.index(44)
            med_42 = float(np.median(mat[:, i42]))
            med_43 = float(np.median(mat[:, i43]))
            med_44 = float(np.median(mat[:, i44]))
            drift_43 = abs(med_43 - med_42)
            drift_44 = abs(med_44 - med_42)
            drift_max = max(drift_43, drift_44)

            rec = {"model": model, "method": method, "stage": stage,
                   "med_42": med_42, "med_43": med_43, "med_44": med_44,
                   "drift_43": drift_43, "drift_44": drift_44,
                   "drift_max": drift_max}
            median_drift_records.append(rec)
            print(f"{model:<14s} {method:<22s} {stage:<6d} "
                  f"{med_42:>+8.4f} {med_43:>+8.4f} {med_44:>+8.4f} "
                  f"{drift_43:>9.4f} {drift_44:>9.4f} {drift_max:>10.4f}")

drifts = np.array([r["drift_max"] for r in median_drift_records])

print(f"\n{'='*70}")
print("Rule-of-thumb thresholds for findings claims (cell-median drift)")
print(f"{'='*70}")
print(f"  Median of cell-median drifts:     {np.median(drifts):.4f}")
print(f"  75th percentile:                  {np.percentile(drifts, 75):.4f}")
print(f"  90th percentile:                  {np.percentile(drifts, 90):.4f}")
print(f"  Max:                              {drifts.max():.4f}  "
      f"(in: {max(median_drift_records, key=lambda r: r['drift_max'])['model']}/"
      f"{max(median_drift_records, key=lambda r: r['drift_max'])['method']}/"
      f"stage {max(median_drift_records, key=lambda r: r['drift_max'])['stage']})")
print()
print("Median over n=10 samples is noisy. These numbers are a")
print("rule-of-thumb threshold for the findings stage, not a significance test.")
print()
print("Interpretation:")
print("  - If our claimed difference D > 75th-percentile drift: claim is safe.")
print("  - If D > median drift but < 75th: mention as tentative.")
print("  - If D < median drift: do not claim.")