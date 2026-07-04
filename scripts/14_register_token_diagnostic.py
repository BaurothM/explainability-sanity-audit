# -*- coding: utf-8 -*-
"""
Diagnostic comparison: which methods on ViT-B/16 highlight the register-token
positions, and which miss them?

Compares IG, GradCAM, Chefer-LRP heatmaps side by side with the raw patch-token
L2 norms (which serve as ground truth for register-token locations).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods import IntegratedGradientsMethod, GradCAMMethod
from src.methods.chefer_lrp import CheferLRPMethod

import torch
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")

# Preprocessing from a reference model
ref_bundle = load_model("vit_base", device)
data_config = timm.data.resolve_model_data_config(ref_bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred_class = ref_bundle.model(x).argmax(dim=1).item()
print(f"Predicted class: {pred_class}")

# --- 2. Compute heatmaps (fresh model per method to avoid cross-talk) ---
print("\nComputing IG...")
bundle_ig = load_model("vit_base", device)
ig = IntegratedGradientsMethod(bundle_ig.model, n_steps=50)
heatmap_ig = ig.explain(x, target=pred_class)

print("Computing GradCAM...")
bundle_gc = load_model("vit_base", device)
gc = GradCAMMethod(
    bundle_gc.model,
    target_layer=bundle_gc.target_layer,
    reshape_transform=bundle_gc.reshape_transform,
)
heatmap_gc = gc.explain(x, target=pred_class)

print("Computing Chefer-LRP...")
bundle_chefer = load_model("vit_base", device)
chefer = CheferLRPMethod(bundle_chefer.model)
heatmap_chefer = chefer.explain(x, target=pred_class)

# --- 3. Token L2 norms (register-token ground truth) ---
print("Computing token norms...")
bundle_norm = load_model("vit_base", device)
final_output = {}

def hook(module, inputs, output):
    final_output["x"] = output.detach()

handle = bundle_norm.model.blocks[-1].register_forward_hook(hook)
with torch.no_grad():
    bundle_norm.model(x)
handle.remove()

tokens = final_output["x"].squeeze(0)
patch_norms = tokens[1:].norm(dim=-1).cpu().numpy()
norms_grid = patch_norms.reshape(14, 14)

# Upsample to input resolution for fair side-by-side comparison
norms_up = F.interpolate(
    torch.tensor(norms_grid)[None, None, :, :].float(),
    size=x.shape[-2:],
    mode="bilinear",
    align_corners=False,
).squeeze().numpy()
norms_normalized = (norms_up - norms_up.min()) / (norms_up.max() - norms_up.min() + 1e-8)

# --- 4. Visualize ---
img_display = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

panels = {
    "IG": heatmap_ig,
    "GradCAM": heatmap_gc,
    "Chefer-LRP": heatmap_chefer,
    "Token L2 norms\n(register-token ground truth)": norms_normalized,
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, (name, hm) in enumerate(panels.items()):
    im = axes[0, i].imshow(hm, cmap="hot")
    axes[0, i].set_title(name, fontsize=11)
    axes[0, i].axis("off")

    axes[1, i].imshow(img_display)
    axes[1, i].imshow(hm, cmap="hot", alpha=0.5)
    axes[1, i].set_title("overlay", fontsize=11)
    axes[1, i].axis("off")

plt.tight_layout()
out_path = FIGURES_DIR / "14_register_token_diagnostic.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")
plt.show()

