# -*- coding: utf-8 -*-
"""
Verify that AttentionWithCapture is a faithful replacement:
1. Wrapped model produces numerically identical output to the original.
2. Captured matrices have the expected shape and are valid softmax outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model
from src.methods.chefer_lrp import install_attention_capture

import torch
import timm
from PIL import Image
from src.paths import DATA_DIR

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load two fresh copies of the same model: one original, one to be wrapped.
bundle_original = load_model("vit_base", device)
bundle_wrapped = load_model("vit_base", device)

# --- 2. Install capture on one copy ---
wrappers = install_attention_capture(bundle_wrapped.model)
print(f"Installed capture on {len(wrappers)} attention modules")

# --- 3. Prepare a real input (the sample dog) ---
img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
data_config = timm.data.resolve_model_data_config(bundle_original.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)
print(f"Input shape: {tuple(x.shape)}")

# --- 4. Numerical identity check ---
with torch.no_grad():
    out_original = bundle_original.model(x)
    out_wrapped = bundle_wrapped.model(x)

max_diff = (out_original - out_wrapped).abs().max().item()
print(f"\nMax abs difference between original and wrapped output: {max_diff:.2e}")
if max_diff < 1e-5:
    print("  -> Numerical identity: PASS")
else:
    print("  -> Numerical identity: FAIL (something is wrong in the wrapper)")

# --- 5. Capture check ---
print("\nCaptured attention matrices:")
for i, w in enumerate(wrappers):
    A = w.attn_matrix
    if A is None:
        print(f"  Block {i:2d}: NO MATRIX captured")
        continue
    # Expected shape: (B, num_heads, N, N) = (1, 12, 197, 197)
    shape = tuple(A.shape)
    row_sums = A.sum(dim=-1)  # should all be ~1.0 (softmax property)
    rsum_min = row_sums.min().item()
    rsum_max = row_sums.max().item()
    print(f"  Block {i:2d}: shape={shape}, row sums in [{rsum_min:.4f}, {rsum_max:.4f}]")

# --- 6. Predicted class for reference ---
with torch.no_grad():
    pred_class = bundle_wrapped.model(x).argmax(dim=1).item()
print(f"\nPredicted class (wrapped model): {pred_class}")
print("(Expected: 258 - Samoyed)")

