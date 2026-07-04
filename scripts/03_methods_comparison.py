# -*- coding: utf-8 -*-
"""
Comparison: run multiple explanation methods on the same model and image.
Each method produces a heatmap. We plot them side by side.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.methods import IntegratedGradientsMethod, GradCAMMethod

import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = timm.create_model("resnet50", pretrained=True)
model.eval()
model.to(device)

img_path = DATA_DIR / "smoke_test_dog.jpg"
img = Image.open(img_path).convert("RGB")
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred_class = model(x).argmax(dim=1).item()
print(f"Predicted class index: {pred_class}")

# --- 2. Instantiate methods ---
# Each method is configured once here.
methods = [
    IntegratedGradientsMethod(model, n_steps=50),
    GradCAMMethod(model, target_layer=model.layer4),
]
print(f"\nMethods: {[m.name for m in methods]}")

# --- 3. Compute heatmaps ---
heatmaps = {}
for method in methods:
    print(f"Computing {method.name}...")
    heatmaps[method.name] = method.explain(x, target=pred_class)

# --- 4. Visualize ---
img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

n_methods = len(methods)
fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))

# Top row: input + each heatmap
axes[0, 0].imshow(img_display)
axes[0, 0].set_title("Input")
axes[0, 0].axis("off")

for i, method in enumerate(methods):
    im = axes[0, i + 1].imshow(heatmaps[method.name], cmap="hot")
    axes[0, i + 1].set_title(method.name)
    axes[0, i + 1].axis("off")
    fig.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

# Bottom row: overlays
axes[1, 0].axis("off")  # blank cell under "Input"

for i, method in enumerate(methods):
    axes[1, i + 1].imshow(img_display)
    axes[1, i + 1].imshow(heatmaps[method.name], cmap="hot", alpha=0.5)
    axes[1, i + 1].set_title(f"{method.name} (overlay)")
    axes[1, i + 1].axis("off")

plt.tight_layout()
out_path = FIGURES_DIR / "03_methods_comparison_resnet50_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()

