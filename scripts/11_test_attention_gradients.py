# -*- coding: utf-8 -*-
"""
Verify that AttentionWithCapture correctly retains gradients of the attention
matrices after a backward pass from the target-class logit.

Steps:
1. Forward pass with capture (produces attn_matrix per block).
2. Backward pass from the predicted class logit.
3. Check that .attn_matrix.grad is now populated on every block, with the
   expected shape and finite values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model
from src.methods.chefer_lrp import install_attention_capture
from src.paths import DATA_DIR

import torch
import timm
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bundle = load_model("vit_base", device)
wrappers = install_attention_capture(bundle.model)
print(f"Installed capture on {len(wrappers)} attention modules")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

# --- 2. Forward pass WITHOUT no_grad (we need gradients) ---
# Note: do NOT wrap in torch.no_grad() this time.
logits = bundle.model(x)
pred_class = logits.argmax(dim=1).item()
print(f"Predicted class: {pred_class}")

# --- 3. Backward pass from the target-class logit ---
# Build a one-hot vector at the predicted class, of the same shape as logits.
# Then backward() propagates the gradient of logits[0, pred_class] through the graph.
target = torch.zeros_like(logits)
target[0, pred_class] = 1.0

# Clear any stray gradients on parameters (not strictly needed but tidy)
bundle.model.zero_grad()

# Backward: this populates .grad on every retained-grad tensor along the way
logits.backward(gradient=target)
print("Backward pass complete")

# --- 4. Check captured gradients on each block ---
print("\nCaptured attention matrix gradients:")
all_ok = True
for i, w in enumerate(wrappers):
    A = w.attn_matrix
    if A is None:
        print(f"  Block {i:2d}: NO MATRIX")
        all_ok = False
        continue
    if A.grad is None:
        print(f"  Block {i:2d}: matrix present but NO GRADIENT")
        all_ok = False
        continue
    g = A.grad
    shape_ok = tuple(g.shape) == (1, 12, 197, 197)
    finite_ok = torch.isfinite(g).all().item()
    print(
        f"  Block {i:2d}: grad shape={tuple(g.shape)}, "
        f"min={g.min().item():+.4e}, max={g.max().item():+.4e}, "
        f"finite={finite_ok}"
    )
    if not (shape_ok and finite_ok):
        all_ok = False

print()
if all_ok:
    print("All blocks: gradients captured with correct shape and finite values. PASS")
else:
    print("FAIL: at least one block had a problem (see above).")


