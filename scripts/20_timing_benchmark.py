# -*- coding: utf-8 -*-
"""
Timing benchmark: measure per-heatmap compute cost for each (method, model)
combination, then project total compute time for the planned dataset axis.

The point of this script is to make the compute budget for the dataset axis
predictable. The actual measurement is on a tiny sample (one image, repeated)
because timing variability between images is negligible, what dominates is
the fixed forward/backward cost per model.

Output: console table with per-call timings and projected totals for
500 / 1000 / 2000 images at one and two targets per image.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA_DIR
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
from PIL import Image

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

img = Image.open(DATA_DIR / "smoke_test_dog.jpg").convert("RGB")
model_names = ["resnet50", "vit_base", "mobilevit_s"]

# --- 2. Build per-model method dict ---
def build_methods(bundle):
    """
    Methods per architecture family, mirroring scripts/06's matrix.

    LRP (zennit) is appropriate for CNN only.
    Chefer-LRP is appropriate for ViT only.
    Hybrid (MobileViT) has no LRP variant in this audit.
    """
    return {
        "IntegratedGradients": IntegratedGradientsMethod(bundle.model, n_steps=50),
        "GradCAM": GradCAMMethod(
            bundle.model,
            target_layer=bundle.target_layer,
            reshape_transform=bundle.reshape_transform,
        ),
        "Random": RandomBaselineMethod(bundle.model, seed=42),
        "LRP": LRPMethod(bundle.model) if bundle.family == "cnn" else None,
        "Chefer-LRP": CheferLRPMethod(bundle.model) if bundle.family == "vit" else None,
    }

# --- 3. Time each (model, method) cell ---
N_REPEATS = 2  # heatmaps per (method, model) cell
N_WARMUP = 1   # warm-up runs before timing starts

timings = {}

for model_name in model_names:
    print(f"\n--- {model_name} ---")
    bundle = load_model(model_name, device)

    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_class = bundle.model(x).argmax(dim=1).item()

    methods = build_methods(bundle)
    timings[model_name] = {}

    for method_name, method in methods.items():
        if method is None:
            timings[model_name][method_name] = None
            continue

        print(f"  {method_name:25s} warm-up...", end="", flush=True)
        t_warm = time.perf_counter()
        _ = method.explain(x, target=pred_class)
        if device.type == "cuda":
            torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - t_warm
        print(f" done in {warm_seconds*1000:7.1f} ms", flush=True)

        # If warm-up was very slow, give the user a heads-up that the
        # full repeat loop will take roughly N_REPEATS * warm_seconds.
        if warm_seconds > 5.0:
            print(f"  {'':25s} (slow method, next {N_REPEATS} repeats "
                  f"will take ~{N_REPEATS * warm_seconds:.1f}s)",
                  flush=True)

        # Timed loop
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N_REPEATS):
            _ = method.explain(x, target=pred_class)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"  {'':25s} repeat {i+1}/{N_REPEATS}: "
                  f"{elapsed:.1f}s elapsed", flush=True)
        t1 = time.perf_counter()

        avg_seconds = (t1 - t0) / N_REPEATS
        timings[model_name][method_name] = avg_seconds
        print(f"  {'':25s} AVG: {avg_seconds*1000:7.1f} ms/call", flush=True)

    # After all methods on this model are timed, free the GPU memory
    # so the next model has a clean slate. Without this, model #3
    # may have to share VRAM with the leftovers of models #1 and #2.
    del bundle, methods, x
    if device.type == "cuda":
        torch.cuda.empty_cache()

# --- 4. Projection table ---
print("\n" + "=" * 70)
print("PROJECTED COMPUTE TIME FOR DATASET AXIS")
print("=" * 70)

# Total time per image, summed across all (model, method) cells
total_per_image = sum(
    t for model in timings.values() for t in model.values() if t is not None
)

print(f"\nTotal heatmap generation per image (all models, all methods): "
      f"{total_per_image:.2f} s")

print("\nProjected wall-clock time for full dataset run:")
print(f"  {'N images':>10}  {'1 target':>12}  {'2 targets':>12}")
for n_images in [500, 1000, 2000]:
    t1 = total_per_image * n_images
    t2 = total_per_image * n_images * 2
    print(f"  {n_images:>10}  {t1/3600:>9.2f} h  {t2/3600:>9.2f} h")

print("\nNote: this is heatmap generation only.")
print("Faithfulness experiments (pixel flipping etc.) add a multiplier")
print("of ~50-100x on top of these numbers.")

