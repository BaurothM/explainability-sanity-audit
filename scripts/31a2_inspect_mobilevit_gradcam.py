# -*- coding: utf-8 -*-
"""
Check whether MobileViT-S GradCAM cascading heatmaps are pixel-identical
across stages 1–4, or whether the heatmaps differ in detail despite
producing the same summary statistics in the multi-seed variance
analysis.

Compares all five cascading stages for sample 0, seed 42, pixel by pixel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS_DIR

import numpy as np

MANIFEST_STEM = "subset_500x2_seed42"
CASCADING_DIR = RESULTS_DIR / "heatmaps_cascading" / MANIFEST_STEM
MODEL = "mobilevit_s"
METHOD = "GradCAM"
SAMPLE_ID = 0
SEED = 42

# Load all five stages
heatmaps = {}
for stage in range(5):
    path = (CASCADING_DIR
            / f"sample_{SAMPLE_ID:05d}_{MODEL}_stage{stage}_seed{SEED}.npz")
    data = np.load(path, allow_pickle=True)
    hm = data[METHOD]
    heatmaps[stage] = hm
    print(f"stage {stage}: shape={hm.shape}, "
          f"min={hm.min():.6f}, max={hm.max():.6f}, "
          f"mean={hm.mean():.6f}, sum={hm.sum():.6f}")

print()

# Pairwise pixel-identity check (st1 vs st2 vs st3 vs st4)
print("Pairwise comparisons (stages 1-4):")
for i in range(1, 5):
    for j in range(i + 1, 5):
        a = heatmaps[i]
        b = heatmaps[j]
        identical = np.array_equal(a, b)
        max_abs_diff = np.abs(a - b).max()
        print(f"  st{i} vs st{j}: identical={identical}, "
              f"max abs diff={max_abs_diff:.6e}")

print()

# Also compare stage 0 to the others for reference
print("Stage 0 vs stages 1-4 (reference, expected to differ):")
for j in range(1, 5):
    a = heatmaps[0]
    b = heatmaps[j]
    identical = np.array_equal(a, b)
    max_abs_diff = np.abs(a - b).max()
    print(f"  st0 vs st{j}: identical={identical}, "
          f"max abs diff={max_abs_diff:.6e}")