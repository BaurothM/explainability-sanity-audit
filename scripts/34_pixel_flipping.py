# -*- coding: utf-8 -*-
"""
Experiment 2: Pixel Flipping / Deletion faithfulness.

For every (sample, model, method) combination, this script measures how
quickly the model's GT-class softmax confidence drops as we mask blocks
of the input image in heatmap-ranked order. We do this in two orderings:

    MoRF (Most Relevant First): mask highest-ranked blocks first.
                                Good heatmap -> steep early drop.
    LeRF (Least Relevant First): mask lowest-ranked blocks first.
                                 Good heatmap -> confidence holds high.

The summary statistic is the area under the confidence-vs-fraction-masked
curve (AUC), computed by the trapezoidal rule over [0, 0.5]. Low AUC under
MoRF and high AUC under LeRF both indicate a faithful heatmap.

Design decisions:

- Block size 8x8 pixels
- Heatmap-to-block aggregation: mean within block.
- Masking baseline: per-channel image mean of the (preprocessed) input.
- Fraction schedule: 0%, 5%, 10%, ..., 50% (11 measurement points).
- Same-fraction-per-model convention: at step k we mask 5k% of blocks
  in EACH model's native grid (28x28 for ResNet/ViT, 32x32 for MobileViT).
  This preserves the faithfulness comparison ("how fast does THIS model
  lose confidence when X% of THIS image is masked").
- Methods: GT-target heatmaps only, reused from results/heatmaps/
  (not recomputed). MobileViT carries IG/GradCAM/Random only. ResNet50
  and ViT additionally carry their respective LRP family member.

Why GT-target only: cascading randomization is GT-only (script 30) and we
keep the same axis here for consistency. Pred-target faithfulness could
be added later without re-running heatmap generation.

Output: results/tables/pixel_flipping__<manifest_stem>.parquet
in long format. One row per (sample, model, method, order, fraction).
"""

import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TABLES_DIR
from src.models import load_model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm import tqdm
import gc


# --- 1. Configuration ----------------------------------------------------
RUN_MODE = "full"          # "smoke" (10 samples), "full" (all samples)
SMOKE_N  = 10

MANIFEST_NAME = "subset_500x2_seed42.csv"
MANIFEST_PATH = DATA_DIR / "manifests" / MANIFEST_NAME
MANIFEST_STEM = MANIFEST_PATH.stem

HEATMAPS_DIR = RESULTS_DIR / "heatmaps" / MANIFEST_STEM
OUTPUT_PATH  = TABLES_DIR / f"pixel_flipping__{MANIFEST_STEM}.parquet"
if RUN_MODE == "smoke":
    OUTPUT_PATH = TABLES_DIR / f"pixel_flipping_smoke__{MANIFEST_STEM}.parquet"

MODEL_NAMES = ["resnet50", "vit_base", "mobilevit_s"]

# Method keys per model (GT-target only). Mirrors scripts/27, /31.
# MobileViT has no LRP family member (Hybrid-LRP is a documented negative
# finding).
MODEL_METHODS = {
    "resnet50":    ["IntegratedGradients", "GradCAM", "Random", "LRP"],
    "vit_base":    ["IntegratedGradients", "GradCAM", "Random", "Chefer-LRP"],
    "mobilevit_s": ["IntegratedGradients", "GradCAM", "Random"],
}

# method_family for cross-architecture grouping (same convention as 27/31)
METHOD_FAMILY = {
    "IntegratedGradients": "IntegratedGradients",
    "GradCAM":              "GradCAM",
    "Random":               "Random",
    "LRP":                  "LRP",
    "Chefer-LRP":           "LRP",
}

BLOCK_SIZE = 8
FRACTIONS = np.arange(0.0, 0.51, 0.05)   # 0%, 5%, ..., 50% -> 11 points
ORDERS = ["MoRF", "LeRF"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:  {device}")
print(f"Run mode:      {RUN_MODE}")
print(f"Manifest:      {MANIFEST_PATH}")
print(f"Heatmaps dir:  {HEATMAPS_DIR}")
print(f"Output path:   {OUTPUT_PATH}")
print(f"Block size:    {BLOCK_SIZE}x{BLOCK_SIZE}")
print(f"Fractions:     {[f'{f:.2f}' for f in FRACTIONS]}")


# --- 2. Read manifest ----------------------------------------------------
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    manifest_rows = list(reader)

if RUN_MODE == "smoke":
    manifest_rows = manifest_rows[:SMOKE_N]

n_samples = len(manifest_rows)
print(f"Samples to process: {n_samples}")


# --- 3. Helpers ----------------------------------------------------------
def aggregate_heatmap_to_blocks(heatmap: np.ndarray, block: int) -> np.ndarray:
    """
    Mean-aggregate an (H, W) heatmap into ((H/b), (W/b)) blocks.
    Assumes H and W are both divisible by block size. ResNet/ViT heatmaps
    are 224x224 -> 28x28 blocks; MobileViT heatmaps are 256x256 -> 32x32.
    """
    H, W = heatmap.shape
    assert H % block == 0 and W % block == 0, \
        f"Heatmap shape {heatmap.shape} not divisible by block size {block}"
    Hb, Wb = H // block, W // block
    return heatmap.reshape(Hb, block, Wb, block).mean(axis=(1, 3))


def get_block_order(block_heatmap: np.ndarray, order: str) -> np.ndarray:
    """
    Return a 1D array of flat block indices, sorted by block-heatmap value.
    MoRF: descending (highest first). LeRF: ascending (lowest first).
    Ties broken by index order (deterministic).
    """
    flat = block_heatmap.flatten()
    if order == "MoRF":
        return np.argsort(-flat, kind="stable")   # descending
    elif order == "LeRF":
        return np.argsort(flat,  kind="stable")   # ascending
    else:
        raise ValueError(f"Unknown order: {order}")


def build_mask_at_fraction(block_grid_shape: tuple, sorted_idx: np.ndarray,
                           fraction: float, block: int,
                           image_hw: tuple) -> torch.Tensor:
    """
    Build a (1, 1, H, W) float mask in {0, 1}: 1 where the pixel is MASKED,
    0 where it is kept. Used to blend image with the mean baseline.

    fraction: fraction of total blocks to mask (e.g. 0.05 = 5%).
    """
    Hb, Wb = block_grid_shape
    n_blocks_total = Hb * Wb
    n_blocks_mask = int(round(fraction * n_blocks_total))

    if n_blocks_mask == 0:
        return torch.zeros((1, 1, image_hw[0], image_hw[1]),
                           dtype=torch.float32)

    block_mask_flat = np.zeros(n_blocks_total, dtype=np.float32)
    block_mask_flat[sorted_idx[:n_blocks_mask]] = 1.0
    block_mask = block_mask_flat.reshape(Hb, Wb)

    # Upsample block_mask (Hb, Wb) to pixel mask (H, W) via Kronecker-style
    # nearest-neighbor repeat. Each block becomes a (block x block) patch.
    pixel_mask = np.repeat(np.repeat(block_mask, block, axis=0), block, axis=1)
    return torch.from_numpy(pixel_mask).unsqueeze(0).unsqueeze(0)


def apply_mask(x: torch.Tensor, mask: torch.Tensor,
               baseline_per_channel: torch.Tensor) -> torch.Tensor:
    """
    Replace masked pixels with the per-channel baseline value.

    x: (1, C, H, W). mask: (1, 1, H, W) with 1.0 = mask, 0.0 = keep.
    baseline_per_channel: (C,) tensor.

    Result: x where mask=0, baseline where mask=1.
    """
    C = x.shape[1]
    bl = baseline_per_channel.view(1, C, 1, 1).to(x.device)
    mask = mask.to(x.device)
    return x * (1.0 - mask) + bl * mask


def load_gt_heatmap(sample_id: int, model_name: str, method_key: str):
    """Load the GT-target heatmap from results/heatmaps/. None if missing."""
    path = HEATMAPS_DIR / f"sample_{sample_id:05d}_{model_name}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        key = f"{method_key}_gt"
        if key not in data.files:
            return None
        return data[key].copy()


# --- 4. Main loop: models outer, samples inner ---------------------------
rows = []
overall_start = time.time()
n_missing_heatmap = 0

try:
    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}\nModel: {model_name}\n{'='*70}")

        bundle = load_model(model_name, device)
        data_config = timm.data.resolve_model_data_config(bundle.model)
        transform = timm.data.create_transform(**data_config, is_training=False)

        methods_for_model = MODEL_METHODS[model_name]
        print(f"Active methods: {methods_for_model}")

        model_start = time.time()
        n_done_model = 0

        pbar = tqdm(manifest_rows, desc=model_name, unit="img")
        for row in pbar:
            sample_id = int(row["sample_id"])
            class_idx_gt = int(row["class_idx"])
            img_rel_path = row["image_path"]

            # Load + preprocess image
            img = Image.open(PROJECT_ROOT / img_rel_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
            _, C, H, W = x.shape

            # Per-channel mean baseline of THIS preprocessed image.
            # Spatial-mean per channel, so we replace masked pixels with a
            # constant per-channel gray that matches the image's color cast.
            baseline = x.view(1, C, -1).mean(dim=2).squeeze(0)  # (C,)

            # Confidence at fraction=0 (unmasked), identical for all
            # methods/orders on this (sample, model), so compute once.
            with torch.no_grad():
                logits0 = bundle.model(x)
                conf0 = F.softmax(logits0, dim=1)[0, class_idx_gt].item()

            for method_key in methods_for_model:
                hm = load_gt_heatmap(sample_id, model_name, method_key)
                if hm is None:
                    n_missing_heatmap += 1
                    continue

                # Sanity: heatmap shape must match image spatial shape.
                # ResNet/ViT: 224x224. MobileViT: 256x256.
                assert hm.shape == (H, W), \
                    f"Heatmap {hm.shape} != image {(H, W)} for " \
                    f"{model_name}/{method_key} sample {sample_id}"

                # Aggregate heatmap to blocks once per method
                block_hm = aggregate_heatmap_to_blocks(hm, BLOCK_SIZE)
                Hb, Wb = block_hm.shape

                for order in ORDERS:
                    sorted_idx = get_block_order(block_hm, order)

                    for frac in FRACTIONS:
                        if frac == 0.0:
                            conf = conf0
                        else:
                            mask = build_mask_at_fraction(
                                (Hb, Wb), sorted_idx, frac,
                                BLOCK_SIZE, (H, W)
                            )
                            x_masked = apply_mask(x, mask, baseline)
                            with torch.no_grad():
                                logits = bundle.model(x_masked)
                                conf = F.softmax(logits, dim=1)[
                                    0, class_idx_gt
                                ].item()

                        rows.append({
                            "sample_id":     sample_id,
                            "model":         model_name,
                            "method":        method_key,
                            "method_family": METHOD_FAMILY[method_key],
                            "order":         order,
                            "fraction":      float(frac),
                            "confidence":    float(conf),
                        })

            n_done_model += 1

        model_elapsed = time.time() - model_start
        print(f"\n{model_name} done: {n_done_model} samples, "
              f"{model_elapsed:.1f}s total "
              f"({model_elapsed / max(n_done_model, 1):.2f}s/sample)")

        # Free VRAM before next model
        del bundle
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Partial results not saved.")
    sys.exit(1)


# --- 5. Build dataframe and compute AUCs ---------------------------------
df = pd.DataFrame(rows)
print(f"\nResult dataframe shape: {df.shape}")

# Compute AUC per (sample, model, method, order) via trapezoidal rule
# over fraction in [0, 0.5]. With 11 equally-spaced points at spacing 0.05,
# this gives a value in [0, 0.5] roughly (since confidence is in [0,1]).
print("\nComputing AUCs (trapezoidal rule over [0, 0.5])...")
auc_rows = []
for (sample_id, model, method, order), group in df.groupby(
    ["sample_id", "model", "method", "order"]
):
    g = group.sort_values("fraction")
    auc = np.trapezoid(g["confidence"].values, g["fraction"].values)
    auc_rows.append({
        "sample_id":     sample_id,
        "model":         model,
        "method":        method,
        "method_family": METHOD_FAMILY[method],
        "order":         order,
        "auc":           float(auc),
    })
auc_df = pd.DataFrame(auc_rows)
print(f"AUC dataframe shape: {auc_df.shape}")


# --- 6. Write outputs ----------------------------------------------------
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved curves to: {OUTPUT_PATH}")

auc_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "__auc.parquet")
auc_df.to_parquet(auc_path, index=False)
print(f"Saved AUCs to:   {auc_path}")


# --- 7. Smoke-test summary ----------------------------------------------
overall_elapsed = time.time() - overall_start
print(f"\n{'='*70}")
print("Run summary")
print(f"{'='*70}")
print(f"Wall-clock: {overall_elapsed:.1f}s ({overall_elapsed/60:.2f} min)")
print(f"Samples:    {n_samples}")
print(f"Per sample: {overall_elapsed/max(n_samples,1):.2f}s")
if n_missing_heatmap:
    print(f"Missing heatmaps skipped: {n_missing_heatmap}")

if RUN_MODE == "smoke":
    # Projection: how long would all 1000 samples take?
    full_n = 1000
    projected = overall_elapsed * (full_n / n_samples)
    print(f"\nProjection for full run (n={full_n}):")
    print(f"  Estimated wall-clock: {projected:.0f}s "
          f"({projected/60:.1f} min, {projected/3600:.2f} h)")

# --- 8. Quick sanity preview --------------------------------------------
print(f"\n{'='*70}")
print("Sanity preview: median AUC per (model, method, order)")
print(f"{'='*70}")
summary = (auc_df
           .groupby(["model", "method", "order"])["auc"]
           .median()
           .reset_index()
           .pivot(index=["model", "method"], columns="order", values="auc")
           .reset_index())
print(summary.to_string(index=False))

print("\nExpected qualitative pattern:")
print("  - For faithful heatmaps: MoRF-AUC < LeRF-AUC (steep drop under MoRF,")
print("    confidence preserved under LeRF).")
print("  - For Random: MoRF-AUC approx LeRF-AUC (no ordering structure).")
print("  - Disagreement direction/size between MoRF and LeRF is itself a")
print("    signal - partly heatmap quality, partly OOD imputation artifact.")