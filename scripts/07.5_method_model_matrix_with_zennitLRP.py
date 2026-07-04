# -*- coding: utf-8 -*-
"""
Extended matrix: adds LRP to the methods×models view from script 06.

LRP is shown as a single row covering both LRP variants:
  - ResNet50:  zennit EpsilonGammaBox composite (classical LRP).
  - ViT-B/16:  manual Chefer-style LRP.
  - MobileViT-S: n/a.

Model display names are pretty-printed (ResNet50, ViT-B/16, MobileViT-S).
Internal model bundles are still loaded by their timm-style names.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods import (
    IntegratedGradientsMethod,
    GradCAMMethod,
    RandomBaselineMethod,
    LRPMethod,
    CheferLRPMethod,
)

import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")

# Internal names (used by load_model) and display names (used in the figure).
model_names = ["resnet50", "vit_base", "mobilevit_s"]
model_display = {
    "resnet50": "ResNet50",
    "vit_base": "ViT-B/16",
    "mobilevit_s": "MobileViT-S",
}
method_names = ["IntegratedGradients", "GradCAM", "Random", "LRP"]

# --- 2. Build methods per model ---
def build_methods(bundle):
    """
    Instantiate methods applicable to a given model bundle.

    For the LRP row, the concrete variant depends on the architecture:
      - resnet family -> classical LRP (zennit EpsilonGammaBox).
      - vit family    -> manual Chefer-style LRP.
      - other (mobilevit / hybrid) -> None, the cell is rendered as n/a.

    Methods that do not apply return None. The caller and the visualization
    must handle missing entries.
    """
    if bundle.family == "cnn":
        lrp_method = LRPMethod(bundle.model)
    elif bundle.family == "vit":
        lrp_method = CheferLRPMethod(bundle.model)
    else:
        lrp_method = None

    methods = {
        "IntegratedGradients": IntegratedGradientsMethod(bundle.model, n_steps=50),
        "GradCAM": GradCAMMethod(
            bundle.model,
            target_layer=bundle.target_layer,
            reshape_transform=bundle.reshape_transform,
        ),
        "Random": RandomBaselineMethod(bundle.model, seed=42),
        "LRP": lrp_method,
    }
    return methods

# --- 3. Compute heatmaps for every (model, method) pair ---
# Nested dict: results[model_name][method_name] = (img_display, heatmap)
results = {}

for model_name in model_names:
    print(f"\n--- {model_display[model_name]} ---")
    bundle = load_model(model_name, device)

    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_class = bundle.model(x).argmax(dim=1).item()
    print(f"  Predicted class index: {pred_class}")

    img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    methods = build_methods(bundle)
    results[model_name] = {}
    for method_name in method_names:
        method = methods[method_name]
        if method is None:
            print(f"  {method_name}: not applicable for this architecture (skipped)")
            results[model_name][method_name] = (img_display, None)
            continue
        print(f"  Computing {method_name}...")
        heatmap = method.explain(x, target=pred_class)
        results[model_name][method_name] = (img_display, heatmap)

# --- 4. Visualize as a methods x models grid ---
# Rows = methods, columns = models. Each cell shows the overlay.
n_rows = len(method_names)
n_cols = len(model_names)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

for r, method_name in enumerate(method_names):
    for c, model_name in enumerate(model_names):
        img_display, heatmap = results[model_name][method_name]
        ax = axes[r, c]

        if heatmap is not None:
            ax.imshow(img_display)
            ax.imshow(heatmap, cmap="hot", alpha=0.5)
        else:
            ax.set_facecolor("#2a2a2a")
            ax.text(
                0.5, 0.5, "n/a",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=20, color="#aaaaaa",
            )

        ax.set_xticks([])
        ax.set_yticks([])
        # Keep frame visible so empty cells look intentional, not broken.

        if r == 0:
            ax.set_title(model_display[model_name], fontsize=14)
        if c == 0:
            ax.text(
                -0.1, 0.5, method_name,
                transform=ax.transAxes,
                fontsize=14, rotation=90,
                va="center", ha="center",
            )

# --- 5. Save ---
plt.tight_layout()
out_path = FIGURES_DIR / "07_5_method_model_matrix_with_lrp_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()



