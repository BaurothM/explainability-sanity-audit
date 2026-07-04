# -*- coding: utf-8 -*-
"""
Run GradCAM on all three architectures (CNN, ViT, hybrid) on the sample image.
Tests that the reshape_transform mechanism works for the transformer model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods import GradCAMMethod

import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")

model_names = ["resnet50", "vit_base", "mobilevit_s"]
results = {}

for name in model_names:
    print(f"\n--- {name} ---")
    bundle = load_model(name, device)

    # Each model has its own preprocessing
    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_class = bundle.model(x).argmax(dim=1).item()
    print(f"  Predicted class index: {pred_class}")

    method = GradCAMMethod(
        bundle.model,
        target_layer=bundle.target_layer,
        reshape_transform=bundle.reshape_transform,
    )
    print("  Computing GradCAM...")
    heatmap = method.explain(x, target=pred_class)

    # Store the displayable input alongside the heatmap
    img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    results[name] = (img_display, heatmap, bundle.family)

# --- Visualize: one column per model, input on top, overlay below ---
fig, axes = plt.subplots(2, len(model_names), figsize=(5 * len(model_names), 10))

for i, name in enumerate(model_names):
    img_display, heatmap, family = results[name]

    axes[0, i].imshow(img_display)
    axes[0, i].imshow(heatmap, cmap="hot", alpha=0.5)
    axes[0, i].set_title(f"{name} ({family})\nGradCAM overlay")
    axes[0, i].axis("off")

    im = axes[1, i].imshow(heatmap, cmap="hot")
    axes[1, i].set_title("heatmap")
    axes[1, i].axis("off")
    fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = FIGURES_DIR / "05_gradcam_all_models_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()

