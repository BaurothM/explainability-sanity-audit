# -*- coding: utf-8 -*-
"""
Activation saturation and numerical sanity test under layer-wise
re-initialization, across all three architectures.

Cascading parameter randomization is only meaningful if upstream
re-initialization actually changes the downstream activations. This
script verifies that condition by probing a downstream activation
layer at every cascading stage and checking that consecutive stages
produce measurably different outputs.

Three checks per model:

(1) Do consecutive stages produce measurably different activations
    at the probe layer? If two consecutive stages yield near-identical
    outputs, the cascading signal is lost in saturation downstream.

(2) Are the activations numerically well-behaved (no NaN, no Inf,
    no values exploding into 1e4+ territory)?

(3) Does layer-wise re-init introduce architecture-specific
    pathologies on any of the three models?

No heatmaps are generated here, the script only inspects forward-pass
activations at a chosen probe layer per model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, DATA_DIR
from src.models import load_model
from src.randomization_schedule import (
    SCHEDULES, iter_stages_cumulative, get_module_by_path,
)

import csv
import torch
import timm
import numpy as np
from PIL import Image


# --- 1. Configuration ------------------------------------------------------
SAMPLE_ID = 0
SEED = 42

# Per model: which downstream layer's output we probe to measure
# whether upstream cascading actually has an effect at that depth.
# We use the GradCAM target layer (from src/models.py) as the probe,
# because that's the activation GradCAM operates on.
PROBE_LAYER_PATH = {
    "resnet50":    "layer4",          # last bottleneck stack
    "vit_base":    "blocks.11",       # last transformer block
    "mobilevit_s": "final_conv",      # probe layer for downstream saturation effects
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sample image
with open(DATA_DIR / "manifests" / "subset_500x2_seed42.csv", "r",
          encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
target_row = next(r for r in rows if int(r["sample_id"]) == SAMPLE_ID)
img = Image.open(PROJECT_ROOT / target_row["image_path"]).convert("RGB")


# --- 2. Re-init helper -----------------------------------------------------
def reinit_paths(model, paths, seed):
    """
    Layer-wise re-initialization: each layer is reset to its own init
    distribution (PyTorch's reset_parameters), and BatchNorm running
    statistics are reset to their pre-training values.
    """
    with torch.no_grad():
        torch.manual_seed(seed)
        for path in paths:
            submodule = get_module_by_path(model, path)
            for module in submodule.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()


# --- 3. Forward-pass capture helper ----------------------------------------
def capture_probe_output(model_name, cumulative_paths, probe_path):
    """
    Load model fresh, re-init the cumulative paths, forward-pass the sample
    image, and return the captured probe-layer output as a numpy array.
    """
    bundle = load_model(model_name, device)
    reinit_paths(bundle.model, cumulative_paths, seed=SEED)

    data_config = timm.data.resolve_model_data_config(bundle.model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    x = transform(img).unsqueeze(0).to(device)

    captured = {}
    def hook(_m, _i, out):
        captured["out"] = out.detach().cpu().clone().numpy()
    probe_module = get_module_by_path(bundle.model, probe_path)
    h = probe_module.register_forward_hook(hook)
    with torch.no_grad():
        _ = bundle.model(x)
    h.remove()

    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return captured["out"]


# --- 4. Per-model test loop ------------------------------------------------
def summarize(arr, name):
    """Return a one-line summary of an activation tensor."""
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    return (
        f"{name}: shape={arr.shape}, "
        f"mean={arr.mean():+.4f}, std={arr.std():.4f}, "
        f"min={arr.min():+.4f}, max={arr.max():+.4f}, "
        f"nan={has_nan}, inf={has_inf}"
    )


def relative_diff(a, b):
    max_abs = np.abs(a - b).max()
    denom = max(np.abs(a).max(), np.abs(b).max(), 1e-9)
    return max_abs, max_abs / denom


print(f"\n{'#'*70}")
print("# Saturation test under layer-wise re-initialization")
print(f"# Sample {SAMPLE_ID}, seed {SEED}")
print(f"{'#'*70}")

results = {}
for model_name in ["resnet50", "vit_base", "mobilevit_s"]:
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Probe layer: {PROBE_LAYER_PATH[model_name]}")
    print(f"{'='*70}")

    schedule = SCHEDULES[model_name]
    probe_path = PROBE_LAYER_PATH[model_name]

    stage_outputs = {}
    for stage_idx, stage_name, cum_paths in iter_stages_cumulative(schedule):
        out = capture_probe_output(model_name, cum_paths, probe_path)
        stage_outputs[stage_idx] = (stage_name, out)
        print(f"  stage {stage_idx} ({stage_name}): "
              f"{summarize(out, 'probe')}")

    # Pairwise comparisons. Consecutive-stage comparisons are most informative 
    # (saturation check), with stage 0 vs stage 4 giving the full-range bound.
    print("\n  Pairwise comparisons (probe-layer output):")
    print(f"  {'pair':<14s} {'max abs diff':>14s} {'relative':>12s}  verdict")
    for i in range(5):
        for j in range(i + 1, 5):
            _, a = stage_outputs[i]
            _, b = stage_outputs[j]
            max_diff, rel_diff = relative_diff(a, b)
            # Verdict: "saturated" if relative diff < 1e-5, "ok" otherwise
            if rel_diff < 1e-5:
                verdict = "SATURATED"
            elif rel_diff < 1e-3:
                verdict = "near-equal"
            else:
                verdict = "changed"
            print(f"  st{i} vs st{j:>2d}  {max_diff:>14.4e}  "
                  f"{rel_diff:>12.4e}  {verdict}")

    results[model_name] = stage_outputs


# --- 5. Final verdict ------------------------------------------------------
print(f"\n{'#'*70}")
print("# Final verdict")
print(f"{'#'*70}")

all_clean = True

for model_name in results:
    print(f"\n{model_name}:")

    # Check 1: do consecutive stages differ?
    consecutive_saturated = []
    for i in range(4):
        _, a = results[model_name][i]
        _, b = results[model_name][i + 1]
        _, rel = relative_diff(a, b)
        if rel < 1e-5:
            consecutive_saturated.append((i, i + 1))

    if consecutive_saturated:
        print("  CONCERN: consecutive stages with relative diff < 1e-5:")
        for i, j in consecutive_saturated:
            print(f"    st{i} -> st{j}")
        all_clean = False
    else:
        print("  OK: all consecutive stage pairs differ measurably.")

    # Check 2: numerical sanity
    pathologies = []
    for stage_idx, (stage_name, arr) in results[model_name].items():
        if np.isnan(arr).any():
            pathologies.append(f"stage {stage_idx}: NaN present")
        if np.isinf(arr).any():
            pathologies.append(f"stage {stage_idx}: Inf present")
        if np.abs(arr).max() > 1e4:
            pathologies.append(
                f"stage {stage_idx}: max abs value "
                f"{np.abs(arr).max():.2e} (numerically unsafe)"
            )

    if pathologies:
        print("  CONCERN: numerical pathologies:")
        for p in pathologies:
            print(f"    {p}")
        all_clean = False
    else:
        print("  OK: no NaN/Inf, max abs activation in safe range.")

print(f"\n{'='*70}")
if all_clean:
    print("All three models pass the saturation + sanity test under re-init.")
    print("All three models pass the saturation and numerical sanity checks.")
else:
    print("At least one model shows a concern. Re-evaluate before proceeding.")
print(f"{'='*70}")