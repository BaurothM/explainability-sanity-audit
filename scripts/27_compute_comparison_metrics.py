# -*- coding: utf-8 -*-
"""
Compute pairwise similarity metrics (Spearman, SSIM, HOG-correlation)
between heatmaps along three axes:

    - cross_method  : same model, different methods (e.g. IG vs GradCAM on ResNet50)
    - cross_model   : same method, different models (e.g. GradCAM on ResNet vs ViT)
    - cross_target  : same model and method, GT-target vs Pred-target

For the cross_method and cross_model axes we use GT-target heatmaps only.
cross_target is by definition the GT-vs-Pred comparison. Pred-based
cross_method/cross_model comparisons are out of scope for the main analysis.
Could be added later.

For cross_model with MobileViT involved, MobileViT heatmaps (256x256) are
downsampled to 224x224 with cv2.INTER_AREA to match ResNet50/ViT-B/16.
Other comparisons need no resampling (shapes already match).

Output: long-format Parquet table with one row per (sample, axis, item_a,
item_b, metric). Aggregation (median, IQR, ...) happens in a downstream
visualization script, not here.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS_DIR, TABLES_DIR

import time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog

# --- 1. Configuration ---
MANIFEST_STEM = "subset_500x2_seed42"
HEATMAPS_DIR  = RESULTS_DIR / "heatmaps" / MANIFEST_STEM
OUTPUT_PATH   = TABLES_DIR  / f"comparison_metrics__{MANIFEST_STEM}.parquet"

# Common comparison size: smallest model input size, downsample others to match.
COMPARE_SIZE = 224

# Models and the methods each one carries.
# Each entry is (heatmap_key, display_label):
#   - heatmap_key matches the key inside the .npz files (e.g. "LRP_gt" -> "LRP")
#   - display_label is the family-level label used in the Parquet output
# This decoupling lets us treat ResNet's EpsilonGammaBox-LRP and ViT's
# Chefer-LRP as members of the same "LRP" family for cross_model
# comparisons. The .npz files are not touched: heatmap_key remains the
# concrete implementation identifier.
MODEL_METHODS = {
    "resnet50": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
        ("LRP",                  "LRP"),          # EpsilonGammaBox-LRP
    ],
    "vit_base": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
        ("Chefer-LRP",           "LRP"),          # Chefer-style LRP, same family
    ],
    "mobilevit_s": [
        ("IntegratedGradients", "IntegratedGradients"),
        ("GradCAM",              "GradCAM"),
        ("Random",               "Random"),
    ],
}

# Native shapes per model (for the resampling decision)
MODEL_SHAPES = {
    "resnet50":    224,
    "vit_base":    224,
    "mobilevit_s": 256,
}

# --- 2. Resample helper ---
# We resample MobileViT heatmaps (256x256) down to ResNet/ViT size (224x224)
# rather than upsampling 224 -> 256. Downsampling aggregates neighboring
# pixels (INTER_AREA = local averaging). Upsampling would have to invent
# pixel values between existing ones, which artificially smooths the heatmap
# and inflates SSIM/HOG similarity scores. Choosing the smaller common
# size is the conservative choice: we lose ~14% linear resolution from
# MobileViT heatmaps, but we add no information that wasn't in the source.
def to_compare_size(heatmap: np.ndarray) -> np.ndarray:
    """
    Resample a heatmap to (COMPARE_SIZE, COMPARE_SIZE). For downsampling
    (256 -> 224) INTER_AREA is the correct choice, for same size it's a no-op.
    """
    if heatmap.shape == (COMPARE_SIZE, COMPARE_SIZE):
        return heatmap
    return cv2.resize(heatmap, (COMPARE_SIZE, COMPARE_SIZE),
                      interpolation=cv2.INTER_AREA)

# --- 3. The three metrics ---
def compute_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation over flattened heatmaps."""
    rho, _ = spearmanr(a.flatten(), b.flatten())
    # spearmanr returns nan if one input is constant (e.g. zero heatmap).
    return float(rho) if np.isfinite(rho) else 0.0

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity, treating heatmaps as [0,1] greyscale images."""
    return float(ssim(a, b, data_range=1.0))

def compute_hog_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation of HOG descriptors. Standard parameters from
    the Adebayo-et-al. follow-up literature.
    """
    hog_a = hog(a, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    hog_b = hog(b, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    if np.std(hog_a) < 1e-12 or np.std(hog_b) < 1e-12:
        return 0.0
    r, _ = pearsonr(hog_a, hog_b)
    return float(r) if np.isfinite(r) else 0.0

METRICS = {
    "spearman": compute_spearman,
    "ssim":     compute_ssim,
    "hog":      compute_hog_corr,
}

# --- 4. Build the list of pairs per axis ---
# Each pair is (axis, item_a_label, item_b_label, fetch_a_fn, fetch_b_fn)
# where fetch_*_fn takes the per-sample heatmaps dict and returns a heatmap.
# We define the labels and fetchers up front so the per-sample loop is clean.

def make_pair_list():
    """
    Returns a list of dicts describing each comparison to perform.
    The dicts carry display labels (for the Parquet) and key-tuples for
    fetching heatmaps from the per-sample {model: {heatmap_key: array}} dict.
    """
    pairs = []

    # Axis 1: cross_method (same model, different methods, GT target)
    # We compare on the display-label level (which is identical to
    # heatmap_key for non-LRP methods, and "LRP" -> "LRP" for ResNet's
    # LRP and ViT's Chefer-LRP). Within one model, each method appears
    # exactly once, so the display labels are also unique within a model.
    for model, methods in MODEL_METHODS.items():
        for i, (key_a, label_a) in enumerate(methods):
            for (key_b, label_b) in methods[i+1:]:
                pairs.append({
                    "axis":   "cross_method",
                    "item_a": f"{model}_{label_a}",
                    "item_b": f"{model}_{label_b}",
                    "fetch_a": (model, f"{key_a}_gt"),
                    "fetch_b": (model, f"{key_b}_gt"),
                    "needs_resample_a": False,
                    "needs_resample_b": False,
                })

    # Axis 2: cross_model (same method family, different models, GT target)
    # We iterate over display labels rather than heatmap keys, so that
    # ResNet's LRP and ViT's Chefer-LRP are correctly paired under the
    # shared family label "LRP". The heatmap_key per model is looked up
    # for the actual .npz fetch.
    model_list = list(MODEL_METHODS.keys())

    # Build {label: {model: heatmap_key}} from MODEL_METHODS, so we can
    # look up which heatmap_key represents this label inside each model.
    label_to_model_key = {}
    for model, methods in MODEL_METHODS.items():
        for key, label in methods:
            label_to_model_key.setdefault(label, {})[model] = key

    # Family-level methods present in >=2 models
    family_methods = [label for label, model_keys in label_to_model_key.items()
                      if len(model_keys) >= 2]

    for label in family_methods:
        model_keys = label_to_model_key[label]
        present_models = [m for m in model_list if m in model_keys]
        for i, mod_a in enumerate(present_models):
            for mod_b in present_models[i+1:]:
                key_a = model_keys[mod_a]
                key_b = model_keys[mod_b]
                pairs.append({
                    "axis":   "cross_model",
                    "item_a": f"{mod_a}_{label}_gt",
                    "item_b": f"{mod_b}_{label}_gt",
                    "fetch_a": (mod_a, f"{key_a}_gt"),
                    "fetch_b": (mod_b, f"{key_b}_gt"),
                    "needs_resample_a": MODEL_SHAPES[mod_a] != COMPARE_SIZE,
                    "needs_resample_b": MODEL_SHAPES[mod_b] != COMPARE_SIZE,
                })

    # Axis 3: cross_target (same model and method, GT vs Pred)
    for model, methods in MODEL_METHODS.items():
        for (key, label) in methods:
            pairs.append({
                "axis":   "cross_target",
                "item_a": f"{model}_{label}_gt",
                "item_b": f"{model}_{label}_pred",
                "fetch_a": (model, f"{key}_gt"),
                "fetch_b": (model, f"{key}_pred"),
                "needs_resample_a": False,
                "needs_resample_b": False,
            })

    return pairs

PAIRS = make_pair_list()
print(f"Total pairs to compute per sample: {len(PAIRS)}")
print(f"  cross_method: {sum(1 for p in PAIRS if p['axis']=='cross_method')}")
print(f"  cross_model:  {sum(1 for p in PAIRS if p['axis']=='cross_model')}")
print(f"  cross_target: {sum(1 for p in PAIRS if p['axis']=='cross_target')}")

# --- 5. Per-sample heatmap loader ---
def load_sample_heatmaps(sample_id: int) -> dict:
    """
    Load all three model .npz files for a given sample. Returns:
        {model_name: {key: heatmap_array, ...}}
    Only loads heatmap arrays (filters out metadata 0-dim entries).
    """
    out = {}
    for model in MODEL_METHODS.keys():
        npz_path = HEATMAPS_DIR / f"sample_{sample_id:05d}_{model}.npz"
        with np.load(npz_path) as data:
            out[model] = {
                key: data[key].copy()
                for key in data.files
                if data[key].ndim == 2   # heatmaps are 2D, metadata is 0-dim
            }
    return out

# --- 6. Main loop ---
# Discover all sample ids from the heatmap directory
all_files = sorted(HEATMAPS_DIR.glob("sample_*_resnet50.npz"))
sample_ids = sorted({
    int(re.match(r"sample_(\d+)_", p.name).group(1))
    for p in all_files
})
print(f"\nFound {len(sample_ids)} samples to process.")

rows = []
overall_start = time.time()

for sample_id in tqdm(sample_ids, unit="sample"):
    heatmaps = load_sample_heatmaps(sample_id)

    for pair in PAIRS:
        mod_a, key_a = pair["fetch_a"]
        mod_b, key_b = pair["fetch_b"]
        a = heatmaps[mod_a][key_a]
        b = heatmaps[mod_b][key_b]

        # Resample if cross-model comparison straddles different shapes
        if pair["needs_resample_a"]:
            a = to_compare_size(a)
        if pair["needs_resample_b"]:
            b = to_compare_size(b)

        for metric_name, metric_fn in METRICS.items():
            value = metric_fn(a, b)
            rows.append({
                "sample_id": sample_id,
                "axis":      pair["axis"],
                "item_a":    pair["item_a"],
                "item_b":    pair["item_b"],
                "metric":    metric_name,
                "value":     value,
            })

elapsed = time.time() - overall_start
print(f"\nDone. Computed {len(rows)} values in {elapsed:.1f}s "
      f"({len(rows)/elapsed:.0f} values/s).")

# --- 7. Write Parquet ---
df = pd.DataFrame(rows)
print(f"\nResult dataframe shape: {df.shape}")
print(f"Saving to: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False)
print("Saved.")

# --- 8. Quick sanity preview ---
print("\n--- Sanity preview: median per (axis, item_a, item_b, metric) ---")
summary = (df
           .groupby(["axis", "item_a", "item_b", "metric"])["value"]
           .median()
           .reset_index())
print(summary.head(20).to_string(index=False))
print(f"... ({len(summary)} total summary rows)")