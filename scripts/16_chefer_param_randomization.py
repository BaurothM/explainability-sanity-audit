# -*- coding: utf-8 -*-
"""
Sanity check: does Chefer-LRP depend on the model's learned parameters?

Compares the heatmap on the trained ViT to heatmaps on the same model with
parameters replaced by random values, across multiple random seeds, because
a single seed is one realization of an intrinsically random output and its
spatial pattern is not reliably informative.

The diagnostic value lies in the distribution of Spearman correlations
across seeds (which should stay well below 1), not in any single heatmap.

This is the all-at-once prototype of the Cascading Parameter Randomization
sanity check (Adebayo et al., 2018).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR, FIGURES_DIR
from src.models import load_model
from src.methods.chefer_lrp import CheferLRPMethod

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
N_SEEDS = 30
EXAMPLE_SEEDS_TO_PLOT = 3  # how many random-heatmap examples to show alongside the trained one


def build_method(randomize_seed: int | None):
    """
    Load a fresh ViT bundle, optionally re-initialize all parameters
    (Adebayo et al.'s layer-wise re-init scheme), then attach Chefer-LRP.

    randomize_seed=None -> trained model untouched.
    randomize_seed=<int> -> every submodule's parameters reset to its own
    init distribution (PyTorch reset_parameters per submodule), BatchNorm
    running statistics reset to their pre-training defaults.
    """
    bundle = load_model("vit_base", device)
    if randomize_seed is not None:
        with torch.no_grad():
            torch.manual_seed(randomize_seed)
            for module in bundle.model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()

    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)
    method = CheferLRPMethod(bundle.model)
    return method, x


# --- 2. Trained baseline ---
print("\nComputing Chefer-LRP on the TRAINED model (baseline)...")
method_t, x_t = build_method(randomize_seed=None)
# Predict class on the trained model. We will use this as the target for ALL
# subsequent randomized-model heatmaps so that target stays constant.
pred_class = method_t.model(x_t).argmax(dim=1).item()
print(f"  Predicted class (target for all subsequent runs): {pred_class}")
heatmap_trained = method_t.explain(x_t, target=pred_class)

# --- 3. Multi-seed randomized runs ---
print(f"\nComputing Chefer-LRP on {N_SEEDS} randomly initialized models...")
correlations = []
example_heatmaps = []  # store a few for visualization

for i in range(N_SEEDS):
    seed = 42 + i
    method_r, x_r = build_method(randomize_seed=seed)
    heatmap_r = method_r.explain(x_r, target=pred_class)
    rho, _ = spearmanr(heatmap_trained.flatten(), heatmap_r.flatten())
    correlations.append(rho)
    if i < EXAMPLE_SEEDS_TO_PLOT:
        example_heatmaps.append((seed, heatmap_r))
    print(f"  Seed {seed}: Spearman ρ = {rho:+.4f}")

correlations = np.array(correlations)

# --- 4. Distribution summary ---
print("\n" + "=" * 70)
print("DISTRIBUTION OF SPEARMAN CORRELATIONS (trained vs. randomized)")
print("=" * 70)
print(f"  N seeds:  {N_SEEDS}")
print(f"  Min:      {correlations.min():+.4f}")
print(f"  Max:      {correlations.max():+.4f}")
print(f"  Mean:     {correlations.mean():+.4f}")
print(f"  Std:      {correlations.std():+.4f}")
print(f"  Median:   {np.median(correlations):+.4f}")

# Pass condition: all correlations must be substantially below 1.
# We require the maximum to be < 0.95.
passed = correlations.max() < 0.95
print()
if passed:
    print(f"PASS: All {N_SEEDS} correlations are below 0.95 (max = "
          f"{correlations.max():+.4f}). The method is sensitive to model "
          f"parameters.")
else:
    print(f"FAIL: at least one correlation reached >= 0.95.")

# --- 5. Visualize: trained heatmap + sample randomized heatmaps + histogram ---
img_display = x_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

n_cols = 1 + EXAMPLE_SEEDS_TO_PLOT  # trained + a few examples
fig = plt.figure(figsize=(5 * n_cols, 10))

# Top row: heatmap overlays
for col in range(n_cols):
    ax = fig.add_subplot(2, n_cols, col + 1)
    ax.imshow(img_display)
    if col == 0:
        ax.imshow(heatmap_trained, cmap="hot", alpha=0.5)
        ax.set_title("Trained model")
    else:
        seed, hm = example_heatmaps[col - 1]
        ax.imshow(hm, cmap="hot", alpha=0.5)
        ax.set_title(f"Random, seed={seed}\nρ = {correlations[col-1]:+.3f}")
    ax.axis("off")

# Bottom row, spanning all columns: histogram of correlations
ax_hist = fig.add_subplot(2, 1, 2)
ax_hist.hist(correlations, bins=12, range=(-1.0, 1.0),
             color="steelblue", edgecolor="black")
ax_hist.axvline(0.0, color="gray", linestyle="--", linewidth=1,
                label="ρ = 0 (no correlation)")
ax_hist.axvline(0.95, color="red", linestyle="--", linewidth=1,
                label="threshold (ρ = 0.95)")
ax_hist.set_xlim(-1.0, 1.0)
ax_hist.set_xlabel("Spearman ρ (trained vs. randomized heatmap)")
ax_hist.set_ylabel("Count")
ax_hist.set_title(
    f"Distribution of correlations across {N_SEEDS} random seeds\n"
    f"mean={correlations.mean():+.3f}, std={correlations.std():.3f}, "
    f"range=[{correlations.min():+.3f}, {correlations.max():+.3f}]"
)
ax_hist.legend(loc="upper left")

plt.tight_layout()
out_path = FIGURES_DIR / "16_chefer_param_randomization.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")
plt.show()