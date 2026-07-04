# -*- coding: utf-8 -*-
"""
Consolidation: run all available methods on all models, single image.
Produces a methods x models matrix of heatmaps.
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
model_names = ["resnet50", "vit_base", "mobilevit_s"]
method_names = ["IntegratedGradients", "GradCAM", "Random", "Chefer-LRP"]

# --- 2. Build methods per model ---
def build_methods(bundle):
    """
    Instantiate methods applicable to a given model bundle.

    Methods that do not apply to a given architecture family return None.
    The caller (and the visualization) must handle missing entries.
    """
    methods = {
        "IntegratedGradients": IntegratedGradientsMethod(bundle.model, n_steps=50),
        "GradCAM": GradCAMMethod(
            bundle.model,
            target_layer=bundle.target_layer,
            reshape_transform=bundle.reshape_transform,
        ),
        "Random": RandomBaselineMethod(bundle.model, seed=42),
        "Chefer-LRP": (
            CheferLRPMethod(bundle.model) if bundle.family == "vit" else None
        ),
    }
    return methods

# --- 3. Compute heatmaps for every (model, method) pair ---
# Nested dict: results[model_name][method_name] = (img_display, heatmap)
results = {}

for model_name in model_names:
    print(f"\n--- {model_name} ---")
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
            ax.set_title(model_name, fontsize=14)
        if c == 0:
            ax.text(
                -0.1, 0.5, method_name,
                transform=ax.transAxes,
                fontsize=14, rotation=90,
                va="center", ha="center",
            )

# --- 5. Save ---
plt.tight_layout()
out_path = FIGURES_DIR / "06_method_model_matrix_dog.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

plt.show()

