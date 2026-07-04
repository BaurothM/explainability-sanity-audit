# -*- coding: utf-8 -*-
"""
Compute similarity metrics (Spearman, SSIM, HOG-correlation) between each
cascading-randomized heatmap (from scripts/30) and its trained-baseline
counterpart (from scripts/22).

For every (sample, model, stage, method) tuple, we load the cascading
heatmap and the matching trained baseline (same sample, same model, same
method, GT-target), then compute the three similarity metrics.

Cascading randomization is GT-target-only (see scripts/30), so all
comparisons here are against the GT-target baseline.

Each row of the output corresponds to ONE cascading heatmap. The Spearman,
SSIM, and HOG metrics live in three columns (wide format), making
per-method cascading curves straightforward to plot directly.

Output: results/tables/cascading_metrics__<manifest_stem>.parquet

Notes:
- No resampling: all comparisons are within the same model, so cascading
  and baseline heatmaps already have identical shapes.
- nan -> 0 convention for Spearman, matching scripts/27 (constant heatmaps
  produce nan from spearmanr. We treat them as zero correlation, since a
  constant heatmap has no rank structure to correlate).
- HOG receives the same defensive treatment if the descriptor is constant.
- method_family column lets downstream scripts group ResNet's LRP and
  ViT's Chefer-LRP under the family label "LRP" for cross-architecture
  plots, as scripts/27 already does for the agreement axes.
- Multi-seed heatmaps (seeds 43, 44 from the multi-seed validation subset
  on samples 0-9) are NOT processed here. They are exclusively for
  scripts/31a's variance comparison. This script processes seed 42 only.
"""

import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS_DIR, TABLES_DIR

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog


# --- 1. Configuration -----------------------------------------------------
MANIFEST_STEM = "subset_500x2_seed42"
CASCADING_DIR = RESULTS_DIR / "heatmaps_cascading" / MANIFEST_STEM
BASELINE_DIR  = RESULTS_DIR / "heatmaps" / MANIFEST_STEM
OUTPUT_PATH   = TABLES_DIR  / f"cascading_metrics__{MANIFEST_STEM}.parquet"

PRIMARY_SEED = 42       # only this seed enters the main cascading analysis

# Models with their methods. (method_key, method_family) pairs:
#   method_key matches the .npz key (e.g. "Chefer-LRP")
#   method_family is the family-level label for cross-arch grouping
# This matches the MODEL_METHODS structure of scripts/27.
MODEL_METHODS = {
    "resnet50": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
        ("LRP",                  "LRP"),
    ],
    "vit_base": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
        ("Chefer-LRP",           "LRP"),
    ],
    "mobilevit_s": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
    ],
}


# --- 2. Metric functions (identical to scripts/27) ------------------------
def compute_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a.flatten(), b.flatten())
    return float(rho) if np.isfinite(rho) else 0.0

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, data_range=1.0))

def compute_hog_corr(a: np.ndarray, b: np.ndarray) -> float:
    hog_a = hog(a, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    hog_b = hog(b, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    if np.std(hog_a) < 1e-12 or np.std(hog_b) < 1e-12:
        return 0.0
    r, _ = pearsonr(hog_a, hog_b)
    return float(r) if np.isfinite(r) else 0.0


# --- 3. Heatmap loaders ---------------------------------------------------
def load_cascading_heatmap(sample_id: int, model: str,
                           stage: int, seed: int, method: str):
    """Return the cascading heatmap, or None if file/method missing."""
    path = (CASCADING_DIR
            / f"sample_{sample_id:05d}_{model}_stage{stage}_seed{seed}.npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if method not in data.files:
            return None
        return data[method].copy()

def load_baseline_heatmap(sample_id: int, model: str, method: str):
    """Return the GT-target trained baseline heatmap, or None if missing."""
    path = BASELINE_DIR / f"sample_{sample_id:05d}_{model}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        key = f"{method}_gt"
        if key not in data.files:
            return None
        return data[key].copy()


# --- 4. Stage names from filenames (1 lookup) ------------------------------
# We need the stage_name (human label like "+layer4") for the output table.
# Rather than re-reading the schedule, we extract it from one .npz per
# (model, stage), which carries `stage_name` as metadata. This avoids
# coupling the analysis script to src/randomization_schedule.py.
def get_stage_names(model: str) -> dict:
    """Return {stage_idx: stage_name} for the given model, from one .npz."""
    out = {}
    for stage in range(5):
        # any sample works. Pick sample 0 with the primary seed
        path = (CASCADING_DIR
                / f"sample_00000_{model}_stage{stage}_seed{PRIMARY_SEED}.npz")
        if path.exists():
            with np.load(path, allow_pickle=True) as data:
                out[stage] = str(data["stage_name"])
        else:
            out[stage] = f"stage_{stage}"
    return out

STAGE_NAMES_PER_MODEL = {m: get_stage_names(m) for m in MODEL_METHODS}
print("Stage names per model:")
for m, sn in STAGE_NAMES_PER_MODEL.items():
    print(f"  {m}: {sn}")


# --- 5. Discover sample ids -----------------------------------------------
# Use ResNet50 stage 0 with primary seed as the canonical sample list.
# (Every (model, stage, seed=42) combination has the same 1000 samples.)
pattern = f"sample_*_resnet50_stage0_seed{PRIMARY_SEED}.npz"
sample_ids = sorted({
    int(re.match(r"sample_(\d+)_", p.name).group(1))
    for p in CASCADING_DIR.glob(pattern)
})
print(f"\nFound {len(sample_ids)} samples to process "
      f"(seed={PRIMARY_SEED}, primary analysis only).")


# --- 6. Main loop ---------------------------------------------------------
rows = []
n_missing_baseline = 0
n_missing_cascading = 0
overall_start = time.time()

for sample_id in tqdm(sample_ids, unit="sample"):
    for model, methods in MODEL_METHODS.items():
        # Cache baselines per (model) to avoid re-loading 5x per sample
        # (once per cascading stage).
        baseline_cache = {}
        for method_key, _ in methods:
            hm = load_baseline_heatmap(sample_id, model, method_key)
            if hm is None:
                n_missing_baseline += 1
                continue
            baseline_cache[method_key] = hm

        for stage in range(5):
            stage_name = STAGE_NAMES_PER_MODEL[model][stage]
            for method_key, method_family in methods:
                if method_key not in baseline_cache:
                    continue  # baseline missing already counted above

                cascading_hm = load_cascading_heatmap(
                    sample_id, model, stage, PRIMARY_SEED, method_key
                )
                if cascading_hm is None:
                    n_missing_cascading += 1
                    continue

                baseline_hm = baseline_cache[method_key]

                rows.append({
                    "sample_id":    sample_id,
                    "model":        model,
                    "stage_idx":    stage,
                    "stage_name":   stage_name,
                    "method":       method_key,
                    "method_family": method_family,
                    "spearman":     compute_spearman(cascading_hm, baseline_hm),
                    "ssim":         compute_ssim(cascading_hm, baseline_hm),
                    "hog":          compute_hog_corr(cascading_hm, baseline_hm),
                })

elapsed = time.time() - overall_start
print(f"\nDone. Computed {len(rows)} (sample, model, stage, method) tuples "
      f"in {elapsed:.1f}s ({len(rows)/elapsed:.0f}/s).")
if n_missing_baseline > 0:
    print(f"  Skipped {n_missing_baseline} (sample, model, method) tuples "
          f"with missing baseline heatmaps.")
if n_missing_cascading > 0:
    print(f"  Skipped {n_missing_cascading} (sample, model, stage, method) "
          f"tuples with missing cascading heatmaps.")


# --- 7. Write Parquet -----------------------------------------------------
df = pd.DataFrame(rows)
print(f"\nResult dataframe shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nSaving to: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False)
print("Saved.")


# --- 8. Quick sanity preview ----------------------------------------------
print(f"\n{'='*70}")
print("Sanity preview: median metric per (model, stage, method)")
print(f"{'='*70}")
summary = (df
           .groupby(["model", "stage_idx", "stage_name", "method"])
           [["spearman", "ssim", "hog"]]
           .median()
           .reset_index())
# Print one model at a time for readability
for model in MODEL_METHODS:
    sub = summary[summary["model"] == model]
    print(f"\n  {model}:")
    print(sub.drop(columns=["model"]).to_string(index=False))