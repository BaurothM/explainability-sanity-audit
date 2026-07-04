# -*- coding: utf-8 -*-
"""
Diagnostic: are the bright spots in Chefer-LRP's heatmap on background
locations register-token artifacts? Compute the L2 norm of each patch token
after the final transformer block and visualize as a 14x14 grid.

If the spatial pattern of high-norm tokens matches the off-object hotspots in
the Chefer-LRP heatmap, the artifacts are a known ViT phenomenon, not a bug.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model

import torch
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model("vit_base", device)
img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

# --- 2. Capture output of the final transformer block via a hook ---
final_block_output = {}

def hook(module, inputs, output):
    final_block_output["x"] = output.detach()

handle = bundle.model.blocks[-1].register_forward_hook(hook)

with torch.no_grad():
    bundle.model(x)

handle.remove()

# Shape: (1, 197, 768)
tokens = final_block_output["x"].squeeze(0)             # (197, 768)
print(f"Token tensor shape after last block: {tuple(tokens.shape)}")

# --- 3. Compute L2 norms of patch tokens (drop CLS) ---
patch_norms = tokens[1:].norm(dim=-1).cpu().numpy()     # (196,)
print(f"Patch token norms: min={patch_norms.min():.2f}, "
      f"max={patch_norms.max():.2f}, mean={patch_norms.mean():.2f}")

# How many tokens have norm > mean + 3*std (a rough "outlier" threshold)?
threshold = patch_norms.mean() + 3 * patch_norms.std()
n_outliers = (patch_norms > threshold).sum()
print(f"Outlier tokens (>mean+3sigma, i.e. >{threshold:.2f}): {n_outliers}")

# --- 4. Visualize the 14x14 grid of token norms next to the input ---
grid = patch_norms.reshape(14, 14)

img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_display)
axes[0].set_title("Input")
axes[0].axis("off")

im = axes[1].imshow(grid, cmap="viridis")
axes[1].set_title("Patch token L2 norm\n(after final block)")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

# Upsample for overlay
grid_up = F.interpolate(
    torch.tensor(grid)[None, None, :, :], size=img_display.shape[:2],
    mode="bilinear", align_corners=False,
).squeeze().numpy()

axes[2].imshow(img_display)
axes[2].imshow(grid_up, cmap="viridis", alpha=0.5)
axes[2].set_title("Norms overlaid on input")
axes[2].axis("off")

plt.tight_layout()
out_path = FIGURES_DIR / "13_token_norms_vit_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")
plt.show()

