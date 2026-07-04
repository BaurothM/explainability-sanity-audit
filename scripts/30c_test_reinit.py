# -*- coding: utf-8 -*-
"""
Compare two randomization strategies for the cascading sanity-check
experiment: naive Gaussian randomization (std=0.02) and layer-wise
re-initialization (PyTorch's reset_parameters per submodule).

For each model and one cascading-stage path set, the script reports
per-layer parameter standard deviations for:
  - The trained model (as loaded from timm).
  - Gaussian randomization (randn * 0.02).
  - Layer-wise re-initialization.

The layer-wise re-init std should match each layer's own initialization
distribution, which is layer-specific and typically much larger than
0.02 for convolutional stages. The comparison motivates the choice
of layer-wise re-init in the main cascading script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model
from src.randomization_schedule import (
    SCHEDULES, iter_stages_cumulative, get_module_by_path,
)

import torch
import numpy as np


def reinit_paths(model, paths, seed):
    """Layer-wise re-initialization via reset_parameters."""
    with torch.no_grad():
        torch.manual_seed(seed)
        for path in paths:
            submodule = get_module_by_path(model, path)
            for module in submodule.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()


def randomize_paths_gaussian(model, paths, seed, std=0.02):
    """Naive Gaussian randomization, for comparison."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for path in paths:
            submodule = get_module_by_path(model, path)
            for p in submodule.parameters():
                p.copy_(torch.randn(p.shape, generator=gen, device=device) * std)


def param_std_per_layer(model, paths):
    """Return list of (layer_name, std) for parameters in the path set."""
    stats = []
    for path in paths:
        submodule = get_module_by_path(model, path)
        for name, p in submodule.named_parameters():
            full_name = f"{path}.{name}"
            stats.append((full_name, float(p.std())))
    return stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name in ["resnet50", "vit_base", "mobilevit_s"]:
    print(f"{'='*70}\nModel: {model_name}\n{'='*70}")

    # Take cascading stage 2 (a representative middle stage)
    schedule = SCHEDULES[model_name]
    stages_resolved = list(iter_stages_cumulative(schedule))
    _, stage_name, cum_paths = stages_resolved[2]
    print(f"Stage 2 ({stage_name}), {len(cum_paths)} paths\n")

    # Trained model
    bundle = load_model(model_name, device)
    stats_trained = param_std_per_layer(bundle.model, cum_paths)

    # Gaussian: randn * 0.02
    bundle = load_model(model_name, device)
    randomize_paths_gaussian(bundle.model, cum_paths, seed=42, std=0.02)
    stats_old = param_std_per_layer(bundle.model, cum_paths)

    # Layer-wise re-init
    bundle = load_model(model_name, device)
    reinit_paths(bundle.model, cum_paths, seed=42)
    stats_new = param_std_per_layer(bundle.model, cum_paths)

    # Print first 8 parameter tensors with all three stds side by side
    print(f"{'layer':<55s} {'trained':>10s} {'gauss(0.02)':>12s} {'reinit':>10s}")
    for (n, s_t), (_, s_o), (_, s_n) in list(
        zip(stats_trained, stats_old, stats_new)
    )[:8]:
        print(f"{n[:55]:<55s} {s_t:>10.4f} {s_o:>12.4f} {s_n:>10.4f}")

    # Aggregate across all parameters in the stage
    all_t = [s for _, s in stats_trained]
    all_o = [s for _, s in stats_old]
    all_n = [s for _, s in stats_new]
    print(f"\n  mean param std:  trained={np.mean(all_t):.4f}  "
          f"gauss(0.02)={np.mean(all_o):.4f}  reinit={np.mean(all_n):.4f}")

    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print()