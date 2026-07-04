# -*- coding: utf-8 -*-
"""
Diagnostic: load the ViT cascading heatmaps from the smoke run and check
for numerical pathologies (NaN/Inf, degenerate ranges) and basic visual
sanity. Also report per-method stats across stages.

This runs on the output of scripts/30 (smoke mode). It does not produce
.npz output, only console diagnostics and one inspection figure.

The figure includes the original input image as a reference row and
marks degenerate (constant) heatmaps explicitly.

Checks whether Chefer-LRP remains numerically stable on heavily 
randomized transformer blocks.
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR
from src.models import load_model

import numpy as np
import torch
import timm
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 1. Configuration ---
MANIFEST_STEM = "subset_500x2_seed42"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps_cascading" / MANIFEST_STEM
MODEL_NAME = "vit_base"
SEED_FOR_FIGURE = 42
SAMPLE_FOR_FIGURE = None       # None = first sample we find

# --- 2. Discover sample IDs ---
all_files = sorted(HEATMAPS_DIR.glob(f"sample_*_{MODEL_NAME}_stage*_seed*.npz"))
if not all_files:
    raise FileNotFoundError(f"No cascading heatmaps found in {HEATMAPS_DIR}")
sample_ids = sorted({int(f.name.split("_")[1]) for f in all_files})
print(f"Found cascading heatmaps for {MODEL_NAME}.")
print(f"Unique sample IDs in smoke run: {sample_ids}")
if SAMPLE_FOR_FIGURE is None:
    SAMPLE_FOR_FIGURE = sample_ids[0]
print(f"Using sample_id={SAMPLE_FOR_FIGURE} for the inspection figure.\n")


# --- 3. Numerical sanity across ALL ViT cascading heatmaps ---
print(f"{'='*70}")
print(f"Numerical sanity check across all ViT cascading heatmaps")
print(f"{'='*70}")

methods_to_check = ["IntegratedGradients", "GradCAM", "Chefer-LRP", "Random"]

n_total = 0
n_with_nan = 0
n_with_inf = 0
n_constant = 0
stats = defaultdict(lambda: defaultdict(list))

for f in all_files:
    data = np.load(f, allow_pickle=True)
    stage_idx = int(data["stage_idx"])
    for method_name in methods_to_check:
        if method_name not in data:
            continue
        hm = data[method_name]
        n_total += 1
        has_nan = np.isnan(hm).any()
        has_inf = np.isinf(hm).any()
        if has_nan:
            n_with_nan += 1
            print(f"  NaN in {f.name} :: {method_name}")
        if has_inf:
            n_with_inf += 1
            print(f"  Inf in {f.name} :: {method_name}")
        if hm.min() == hm.max():
            n_constant += 1
            print(f"  CONSTANT (min==max=={hm.min():.4f}) in {f.name} :: {method_name}")
        stats[method_name][stage_idx].append(
            (float(hm.min()), float(hm.max()),
             float(hm.mean()), float(hm.std()))
        )

print(f"\nTotal heatmaps scanned:  {n_total}")
print(f"  with NaN:              {n_with_nan}")
print(f"  with Inf:              {n_with_inf}")
print(f"  constant (degenerate): {n_constant}")
print()


# --- 4. Per-method, per-stage summary table ---
print(f"{'='*70}")
print(f"Per-method × per-stage stats (averaged across seeds and samples)")
print(f"{'='*70}")
print(f"{'method':<22s} {'stage':<6s} {'mean min':>12s} {'mean max':>12s} "
      f"{'mean(hm)':>12s} {'mean std':>12s}")
for method_name in methods_to_check:
    if method_name not in stats:
        print(f"{method_name}: NOT FOUND in any file")
        continue
    for stage_idx in sorted(stats[method_name].keys()):
        records = np.array(stats[method_name][stage_idx])
        mn, mx, mean, std = records.mean(axis=0)
        print(f"{method_name:<22s} {stage_idx:<6d} {mn:>12.4f} {mx:>12.4f} "
              f"{mean:>12.4f} {std:>12.4f}")
    print()


# --- 5. Load the original input image (as the ViT model would see it) ---
print(f"{'='*70}")
print(f"Loading original image for sample {SAMPLE_FOR_FIGURE}")
print(f"{'='*70}")

manifest_path = DATA_DIR / "manifests" / f"{MANIFEST_STEM}.csv"
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    manifest_rows = list(reader)

target_row = next(r for r in manifest_rows
                  if int(r["sample_id"]) == SAMPLE_FOR_FIGURE)
img_rel_path = target_row["image_path"]
img_pil = Image.open(PROJECT_ROOT / img_rel_path).convert("RGB")

# Get the model's transform so the image we display matches the heatmap geometry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model(MODEL_NAME, device)
data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x_transformed = transform(img_pil)             # (3, H, W) tensor
img_display = x_transformed.cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
del bundle
if device.type == "cuda":
    torch.cuda.empty_cache()
print(f"Image loaded, transformed shape: {img_display.shape}")


# --- 6. Visualize: input row + (method × seed) rows × stages columns ---
print(f"{'='*70}")
print(f"Generating inspection figure for sample {SAMPLE_FOR_FIGURE}, "
      f"seed {SEED_FOR_FIGURE}")
print(f"{'='*70}")

file_pattern = (
    f"sample_{SAMPLE_FOR_FIGURE:05d}_{MODEL_NAME}_stage*_seed{SEED_FOR_FIGURE}.npz"
)
files_for_fig = sorted(HEATMAPS_DIR.glob(file_pattern))
if not files_for_fig:
    raise FileNotFoundError(
        f"No files matching {file_pattern} in {HEATMAPS_DIR}"
    )

n_stages = len(files_for_fig)
n_methods = len(methods_to_check)
n_rows = 1 + n_methods             # 1 input row + 4 method rows

fig, axes = plt.subplots(
    n_rows, n_stages,
    figsize=(2.5 * n_stages, 2.5 * n_rows),
    squeeze=False,
)

# Row 0: the input image, repeated across columns for visual alignment
for col in range(n_stages):
    ax = axes[0][col]
    ax.imshow(img_display)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("input", fontsize=9, rotation=90, labelpad=8, va="center")

# Rows 1..n_methods: heatmaps
stage_names = []
for col, f in enumerate(files_for_fig):
    data = np.load(f, allow_pickle=True)
    stage_idx = int(data["stage_idx"])
    stage_name = str(data["stage_name"])
    stage_names.append(f"st{stage_idx}\n{stage_name}")

    for row_offset, method_name in enumerate(methods_to_check):
        row = row_offset + 1   # heatmap rows start at row 1
        ax = axes[row][col]
        if method_name not in data:
            ax.text(0.5, 0.5, "n/a",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        hm = data[method_name]
        is_constant = (hm.min() == hm.max())

        ax.imshow(hm, cmap="hot")
        ax.set_xticks([]); ax.set_yticks([])

        title = f"[{hm.min():.2f}, {hm.max():.2f}]"
        if is_constant:
            title = "CONSTANT\n" + title
            ax.set_title(title, fontsize=7, color="red")
        else:
            ax.set_title(title, fontsize=7)

        if col == 0:
            ax.set_ylabel(method_name, fontsize=9, rotation=90,
                          labelpad=8, va="center")

# Top-row column labels (the stage names) go above the input row
for col in range(n_stages):
    axes[0][col].set_xlabel(stage_names[col], fontsize=8)
    axes[0][col].xaxis.set_label_position("top")

fig.suptitle(
    f"ViT cascading heatmaps - sample {SAMPLE_FOR_FIGURE}, seed {SEED_FOR_FIGURE}\n"
    f"(top row = input; method rows = heatmaps; "
    f"red 'CONSTANT' = collapsed to a single value)",
    fontsize=10,
)
plt.tight_layout()

out_path = FIGURES_DIR / "30b_vit_cascading_inspection.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"\nSaved figure to {out_path}")