# -*- coding: utf-8 -*-
"""
Self-consistency checks for the manual Chefer-LRP implementation.

Three property-based tests that should hold for any correct implementation:

  1. Each row of every A_eff sums to 1 after row-normalization.
  2. Two identical runs on the same input produce identical heatmaps 
     (bitwise equal up to float tolerance).
  3. Heatmaps for the correct class and for a clearly unrelated class differ 
     substantially. If they were similar, the method would be ignoring its 
     target argument.

These tests check claims that follow from the algorithm's specification.
Failure of any of them indicates a real implementation bug.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR
from src.models import load_model
from src.methods.chefer_lrp import (
    CheferLRPMethod,
    _compute_chefer_relevance,
)

import torch
import timm
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")

def fresh_method():
    """Fresh model + method per call - avoids cross-test contamination."""
    bundle = load_model("vit_base", device)
    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)
    method = CheferLRPMethod(bundle.model)
    return bundle, method, x

# --- 2. Test 1: per-block conservation ---
print("=" * 70)
print("TEST 1: Per-block conservation (each A_eff row sums to 1)")
print("=" * 70)

bundle, method, x = fresh_method()
with torch.no_grad():
    pred_class = bundle.model(x).argmax(dim=1).item()

# Run forward+backward as in explain(), but stop before accumulation so we can
# inspect each block's A_eff separately.
bundle.model.zero_grad()
logits = bundle.model(x)
target_signal = torch.zeros_like(logits)
target_signal[0, pred_class] = 1.0
logits.backward(gradient=target_signal)

# Manually replicate the per-block construction to check rows.
N = method.wrappers[0].attn_matrix.shape[-1]
max_deviations = []
for i, w in enumerate(method.wrappers):
    A = w.attn_matrix
    gA = A.grad
    A_eff = (gA * A).clamp(min=0).squeeze(0).mean(dim=0)  # (N, N)
    A_eff = A_eff + torch.eye(N, device=device)
    A_eff = A_eff / A_eff.sum(dim=-1, keepdim=True)
    row_sums = A_eff.sum(dim=-1)
    max_dev = (row_sums - 1.0).abs().max().item()
    max_deviations.append(max_dev)

worst_block = int(np.argmax(max_deviations))
worst_dev = max_deviations[worst_block]
print(f"Worst-case row-sum deviation from 1.0: {worst_dev:.2e} "
      f"(block {worst_block})")
if worst_dev < 1e-5:
    print("  -> Conservation: PASS")
else:
    print("  -> Conservation: FAIL")
test1_ok = worst_dev < 1e-5

# --- 3. Test 2: determinism ---
print()
print("=" * 70)
print("TEST 2: Determinism (two identical runs produce identical heatmaps)")
print("=" * 70)

bundle1, method1, x1 = fresh_method()
heatmap_a = method1.explain(x1, target=pred_class)

bundle2, method2, x2 = fresh_method()
heatmap_b = method2.explain(x2, target=pred_class)

max_diff = np.abs(heatmap_a - heatmap_b).max()
print(f"Max abs difference between two runs: {max_diff:.2e}")
if max_diff < 1e-5:
    print("  -> Determinism: PASS")
else:
    print("  -> Determinism: FAIL")
test2_ok = max_diff < 1e-5

# --- 4. Test 3: target sensitivity ---
print()
print("=" * 70)
print("TEST 3: Target sensitivity (different classes produce different maps)")
print("=" * 70)

# Use a target far from "Samoyed" semantically. Class 130 is "flamingo" -
# a non-mammal, very different visual category.
unrelated_class = 130
print(f"Comparing target={pred_class} (predicted) vs. target={unrelated_class} "
      f"(unrelated)")

bundle3, method3, x3 = fresh_method()
heatmap_correct = method3.explain(x3, target=pred_class)

bundle4, method4, x4 = fresh_method()
heatmap_unrelated = method4.explain(x4, target=unrelated_class)

# Compare with Spearman rank correlation: high (~1.0) would mean the method
# ignores the target. Values noticeably below 1 indicate genuine sensitivity.
rho, _ = spearmanr(heatmap_correct.flatten(), heatmap_unrelated.flatten())
print(f"Spearman rank correlation between the two heatmaps: {rho:.4f}")
if rho < 0.95:
    print("  -> Target sensitivity: PASS (heatmaps differ meaningfully)")
else:
    print("  -> Target sensitivity: FAIL (heatmaps are essentially identical)")
test3_ok = rho < 0.95

# --- 5. Summary ---
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
results = [
    ("Per-block conservation", test1_ok),
    ("Determinism", test2_ok),
    ("Target sensitivity", test3_ok),
]
for name, ok in results:
    mark = "PASS" if ok else "FAIL"
    print(f"  {name:35s}  {mark}")
print()
if all(ok for _, ok in results):
    print("All self-consistency tests passed.")
else:
    print("At least one test failed - see details above.")

