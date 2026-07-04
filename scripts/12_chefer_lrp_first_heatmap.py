# -*- coding: utf-8 -*-
"""
First end-to-end run of manual Chefer-LRP on ViT-B/16 with the sample image.
Generates a heatmap and visualizes it alongside the input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods.chefer_lrp import CheferLRPMethod

import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bundle = load_model("vit_base", device)
img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

# Predicted class
with torch.no_grad():
    pred_class = bundle.model(x).argmax(dim=1).item()
print(f"Predicted class: {pred_class}")

# --- 2. Build method (wraps attention modules in-place) ---
method = CheferLRPMethod(bundle.model)

# --- 3. Compute heatmap ---
print("\nComputing Chefer-LRP...")
heatmap = method.explain(x, target=pred_class)
print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

# --- 4. Visualize ---
img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_display)
axes[0].set_title("Input")
axes[0].axis("off")

im = axes[1].imshow(heatmap, cmap="hot")
axes[1].set_title("Chefer-LRP on ViT-B/16")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(img_display)
axes[2].imshow(heatmap, cmap="hot", alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

# --- 5. Save ---
plt.tight_layout()
out_path = FIGURES_DIR / "12_chefer_lrp_vit_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()

