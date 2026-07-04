# -*- coding: utf-8 -*-
"""
LRP on MobileViT-S, single image.

MobileViT mixes conv blocks with transformer blocks. Our EpsilonGammaBox
composite is designed for conv-style architectures and does not have specific
rules for attention. zennit will apply default behavior to unknown modules.
This script tests whether the result is plausible or pathological.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods import LRPMethod

import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bundle = load_model("mobilevit_s", device)
img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")

data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred_class = bundle.model(x).argmax(dim=1).item()
print(f"Predicted class index: {pred_class}")

# --- 2. LRP ---
print("\nComputing LRP on MobileViT")
try:
    method = LRPMethod(bundle.model)
    heatmap = method.explain(x, target=pred_class)
    print(f"Succeeded. Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    # Quick sanity stats: how much of the heatmap is non-trivial?
    nonzero_fraction = (heatmap > 0.01).mean()
    print(f"Fraction of pixels with relevance > 0.01: {nonzero_fraction:.3f}")
    success = True
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    success = False
    heatmap = None

if not success:
    print("\nNo heatmap produced. Stopping here.")
    sys.exit(0)

# --- 3. Visualize ---
img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_display)
axes[0].set_title("Input")
axes[0].axis("off")

im = axes[1].imshow(heatmap, cmap="hot")
axes[1].set_title("LRP on MobileViT-S\n(EpsilonGammaBox, no transformer rules)")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(img_display)
axes[2].imshow(heatmap, cmap="hot", alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

# --- 4. Save ---
plt.tight_layout()
out_path = FIGURES_DIR / "08_lrp_mobilevit_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()

