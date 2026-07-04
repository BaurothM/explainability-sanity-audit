# -*- coding: utf-8 -*-
"""
Diagnostic visualization: show the per-MobileVitBlock heatmaps of the F.2
hybrid Chefer-LRP method, alongside the final aggregated heatmap.

Lets us see whether individual blocks produce meaningful spatial signals
(and the aggregation destroys them), or whether each block is already
chaotic (in which case the aggregation is not the source of the problem).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods.chefer_lrp import CheferLRPMobileViTMethod

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
print(f"Predicted class: {pred_class}")

# --- 2. Run method with per-block output ---
method = CheferLRPMobileViTMethod(bundle.model)
print(f"\nComputing Chefer-LRP F.2 (per-block diagnostic)...")
final, per_block = method.explain(x, target=pred_class, return_per_block_maps=True)
print(f"Number of per-block heatmaps: {len(per_block)}")
for i, hm in enumerate(per_block):
    print(f"  Block {i}: shape={hm.shape}, range=[{hm.min():.3f}, {hm.max():.3f}]")
print(f"Final heatmap: shape={final.shape}, range=[{final.min():.3f}, {final.max():.3f}]")

# --- 3. Visualize: 2 rows x 4 cols ---
# Row 0: pure heatmaps. Row 1: overlays on input.
img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

panels = [
    (f"Block {i + 1}\n(stage {i + 2})", hm) for i, hm in enumerate(per_block)
] + [("Final (mean)", final)]

fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 10))

for col, (title, hm) in enumerate(panels):
    axes[0, col].imshow(hm, cmap="hot")
    axes[0, col].set_title(title, fontsize=12)
    axes[0, col].axis("off")

    axes[1, col].imshow(img_display)
    axes[1, col].imshow(hm, cmap="hot", alpha=0.5)
    axes[1, col].axis("off")

axes[0, 0].text(
    -0.08, 0.5, "Heatmaps",
    transform=axes[0, 0].transAxes,
    fontsize=14, rotation=90,
    va="center", ha="center",
)
axes[1, 0].text(
    -0.08, 0.5, "Overlays",
    transform=axes[1, 0].transAxes,
    fontsize=14, rotation=90,
    va="center", ha="center",
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
out_path = FIGURES_DIR / "19_chefer_lrp_mobilevit_per_block.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")
plt.show()

