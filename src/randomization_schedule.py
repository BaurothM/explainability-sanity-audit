# -*- coding: utf-8 -*-
"""
Cascading parameter randomization schedules per architecture.

Stage k of cascading randomization replaces the parameters of 
stages[0..k] (inclusive) with random values, leaving deeper layers 
(closer to the input) trained. Stage 0 means "only the head-most 
group randomized". The final stage means "almost the entire model 
randomized".

The schedules are not directly comparable across architectures in
absolute terms (different param counts, different structural roles),
but the SHAPE of the resulting Spearman-vs-stage curve is the
comparable quantity.

Layer names ('paths') refer to dotted attribute paths from the model
root: e.g. "blocks.9" means model.blocks[9]. Validation against an
actual model is provided by validate_schedule().
"""

from typing import List, Dict
import torch.nn as nn


# --- Schedule definitions ---------------------------------------------------
#
# Each schedule entry has:
#   name:   short human-readable label (for plotting + logging)
#   paths:  list of dotted attribute paths to randomize at this stage,
#           ON TOP OF the paths from all previous stages

SCHEDULES: Dict[str, List[Dict]] = {
    "resnet50": [
        {"name": "fc",           "paths": ["fc"]},
        {"name": "+layer4",      "paths": ["layer4"]},
        {"name": "+layer3",      "paths": ["layer3"]},
        {"name": "+layer2",      "paths": ["layer2"]},
        {"name": "+layer1+stem", "paths": ["layer1", "conv1", "bn1"]},
    ],
    "vit_base": [
        {"name": "head",          "paths": ["head"]},
        {"name": "+blocks9-11",   "paths": ["norm", "blocks.11", "blocks.10", "blocks.9"]},
        {"name": "+blocks6-8",    "paths": ["blocks.8", "blocks.7", "blocks.6"]},
        {"name": "+blocks3-5",    "paths": ["blocks.5", "blocks.4", "blocks.3"]},
        {"name": "+blocks0-2+pe", "paths": ["blocks.2", "blocks.1", "blocks.0", "patch_embed"]},
    ],
    "mobilevit_s": [
        {"name": "head",            "paths": ["head"]},
        {"name": "+final+stages4", "paths": ["final_conv", "stages.4"]},
        {"name": "+stages3",       "paths": ["stages.3"]},
        {"name": "+stages2",       "paths": ["stages.2"]},
        {"name": "+stages0-1+stem", "paths": ["stages.1", "stages.0", "stem"]},
    ],
}


# --- Helpers ---------------------------------------------------------------

def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """
    Resolve a dotted attribute path like 'blocks.9' or 'layer4' into
    the corresponding submodule of `model`.

    Raises AttributeError if any component is missing.
    """
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def validate_schedule(model: nn.Module, schedule: List[Dict]) -> None:
    """
    Verify that every path in the schedule resolves to an existing
    submodule of `model`. Raises AttributeError with a clear message
    if not.

    Should be called once before starting a cascading run, so typos
    are caught at the top of the script and not deep inside a loop.
    """
    for stage_idx, stage in enumerate(schedule):
        for path in stage["paths"]:
            try:
                get_module_by_path(model, path)
            except (AttributeError, IndexError, TypeError) as e:
                raise AttributeError(
                    f"Schedule stage {stage_idx} ('{stage['name']}'): "
                    f"path '{path}' does not resolve in model. "
                    f"Original error: {e}"
                )


def iter_stages_cumulative(schedule: List[Dict]):
    """
    Yield (stage_idx, stage_name, cumulative_paths) for each stage of
    the cascading schedule.

    cumulative_paths includes paths from this stage AND all previous ones,
    which is what 'cascading' randomization needs to replace at this step.
    """
    cumulative: List[str] = []
    for stage_idx, stage in enumerate(schedule):
        cumulative = cumulative + stage["paths"]
        yield stage_idx, stage["name"], list(cumulative)


# --- Smoke test -------------------------------------------------------------
# Quick check that every defined schedule validates against its model.
# Runs when this file is executed directly (not on import).

if __name__ == "__main__":
    import torch
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, schedule in SCHEDULES.items():
        print(f"Validating schedule for {model_name}...")
        bundle = load_model(model_name, device)
        validate_schedule(bundle.model, schedule)

        # Also print param counts per cumulative stage for inspection
        for stage_idx, stage_name, paths in iter_stages_cumulative(schedule):
            n_params = sum(
                p.numel()
                for path in paths
                for p in get_module_by_path(bundle.model, path).parameters()
            )
            print(f"  stage {stage_idx} '{stage_name}': "
                  f"{len(paths)} paths, {n_params:,} cumulative params")
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    print("All schedules valid.")