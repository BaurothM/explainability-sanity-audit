# -*- coding: utf-8 -*-
"""
Cascading parameter randomization (Adebayo et al., 2018).

For each (model, stage, seed), reload the model fresh, randomize all
parameters in the cumulative path set of this stage (top-down: head-most
group at stage 0, almost-entire-model at the last stage), then compute
heatmaps for every applicable method and sample in the manifest.

Output layout (per manifest):
    results/heatmaps_cascading/<manifest_stem>/
        sample_XXXXX_<modelname>_stage<S>_seed<R>.npz

Each .npz contains, for every applicable method on this model:
    {method_name} : (H, W) float32 heatmap for ground-truth class
plus 0-dim metadata arrays (class_idx_gt, stage_idx, stage_name,
seed_value, model_name).

Cross-target axis is NOT included: we generate heatmaps for the GT
class only. The cascading question is about model parameters, not
class sensitivity.

Resume: on each (sample, model, stage, seed), if the target .npz
already exists, the iteration is skipped. Files are written atomically.

RUN_MODE controls the sample budget:
    "smoke"    -> first 3 samples           (~7 min)
    "estimate" -> first 10 samples          (~22 min)
    "full"     -> all samples in manifest   (extrapolated from estimate)
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
from src.models import load_model
from src.methods import (
    IntegratedGradientsMethod,
    GradCAMMethod,
    RandomBaselineMethod,
    LRPMethod,
    CheferLRPMethod,
)
from src.randomization_schedule import (
    SCHEDULES,
    validate_schedule,
    iter_stages_cumulative,
    get_module_by_path,
)

import csv
import gc
import hashlib

import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm


# --- 1. Configuration -------------------------------------------------------
RUN_MODE = "full"          # one of "smoke", "estimate", "full"

MANIFEST_NAME = "subset_500x2_seed42.csv"
N_SEEDS_PER_STAGE = 1
MODEL_NAMES = ["resnet50", "vit_base", "mobilevit_s"]
SEEDS = [42, 43, 44][:N_SEEDS_PER_STAGE]

SAMPLE_LIMITS = {
    "smoke":    3,
    "estimate": 10,
    "full":     None,        # None means "use the whole manifest"
}

MANIFEST_PATH = DATA_DIR / "manifests" / MANIFEST_NAME
HEATMAPS_DIR = RESULTS_DIR / "heatmaps_cascading" / MANIFEST_PATH.stem
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:  {device}")
print(f"Manifest:      {MANIFEST_PATH}")
print(f"Output dir:    {HEATMAPS_DIR}")
print(f"Run mode:      {RUN_MODE}")


# --- 2. Read manifest -------------------------------------------------------
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    manifest_rows = list(reader)

n_total_in_manifest = len(manifest_rows)
sample_limit = SAMPLE_LIMITS[RUN_MODE]
if sample_limit is not None:
    manifest_rows = manifest_rows[:sample_limit]
n_samples = len(manifest_rows)
print(f"Samples in this run: {n_samples} / {n_total_in_manifest} in manifest")


# --- 3. Total work estimate -------------------------------------------------
# So we can compute a meaningful global ETA, and report scope up front.
n_stages_per_model = {m: len(SCHEDULES[m]) for m in MODEL_NAMES}
n_iterations_total = sum(
    n_stages_per_model[m] * N_SEEDS_PER_STAGE for m in MODEL_NAMES
)
n_heatmap_files_total = n_iterations_total * n_samples
print(
    f"Scope: {n_iterations_total} (model, stage, seed) iterations × "
    f"{n_samples} samples = {n_heatmap_files_total} .npz files"
)


# --- 4. Re-initialization helper -------------------------------------------
def reinit_paths(
    model: torch.nn.Module,
    paths: list,
    seed: int,
) -> None:
    """
    Layer-wise re-initialization (Adebayo et al., 2018): for every
    submodule in the given path set, reset its parameters to fresh
    values drawn from the layer's own initialization distribution
    (PyTorch's reset_parameters), and reset BatchNorm running statistics
    to their pre-training defaults (running_mean=0, running_var=1).

    This is the canonical sanity-check randomization scheme, layer-aware
    by construction.

    The seed seeds the default torch RNG, because reset_parameters()
    does not accept an explicit generator. Other RNG-driven code paths
    in this script (the Random baseline) use their own generators and
    are unaffected.
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


# --- 5. Method factory -----------------------------------------------------
def build_methods(bundle):
    """Instantiate methods applicable to a given model bundle (no Random)."""
    return {
        "IntegratedGradients": IntegratedGradientsMethod(bundle.model, n_steps=50),
        "GradCAM": GradCAMMethod(
            bundle.model,
            target_layer=bundle.target_layer,
            reshape_transform=bundle.reshape_transform,
        ),
        "LRP": LRPMethod(bundle.model) if bundle.family == "cnn" else None,
        "Chefer-LRP": CheferLRPMethod(bundle.model) if bundle.family == "vit" else None,
    }


# --- 6. Atomic save helper -------------------------------------------------
def atomic_savez(path: Path, **arrays):
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


# --- 7. Schedule pre-validation --------------------------------------------
print("\nValidating schedules...")
for model_name in MODEL_NAMES:
    bundle = load_model(model_name, device)
    validate_schedule(bundle.model, SCHEDULES[model_name])
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
print("All schedules valid.\n")


# --- 8. Main loop ----------------------------------------------------------
overall_start = time.time()
n_computed_total = 0
n_skipped_total = 0
n_iterations_done = 0
interrupted = False

try:
    for model_name in MODEL_NAMES:
        schedule = SCHEDULES[model_name]
        n_stages = len(schedule)

        print(f"{'='*70}\nModel: {model_name}  ({n_stages} stages, "
              f"{N_SEEDS_PER_STAGE} seeds each)\n{'='*70}")

        for stage_idx, stage_name, cumulative_paths in iter_stages_cumulative(schedule):
            for seed in SEEDS:
                # --- 8a. Fresh model + re-initialization ---
                bundle = load_model(model_name, device)
                reinit_paths(
                    bundle.model, cumulative_paths, seed=seed
                )

                data_config = timm.data.resolve_model_data_config(bundle.model)
                transform = timm.data.create_transform(
                    **data_config, is_training=False
                )

                methods = build_methods(bundle)
                active_methods = {k: v for k, v in methods.items() if v is not None}

                # --- 8b. Inner loop over samples ---
                desc = f"{model_name} st{stage_idx} ({stage_name}) sd{seed}"
                stage_seed_start = time.time()
                n_computed_local = 0
                n_skipped_local = 0

                pbar = tqdm(manifest_rows, desc=desc, unit="img", leave=False)
                for row in pbar:
                    sample_id = int(row["sample_id"])
                    class_idx_gt = int(row["class_idx"])
                    img_rel_path = row["image_path"]

                    out_path = (
                        HEATMAPS_DIR
                        / f"sample_{sample_id:05d}_{model_name}"
                          f"_stage{stage_idx}_seed{seed}.npz"
                    )

                    if out_path.exists():
                        n_skipped_local += 1
                        continue

                    img = Image.open(PROJECT_ROOT / img_rel_path).convert("RGB")
                    x = transform(img).unsqueeze(0).to(device)

                    arrays = {}

                    for method_name, method in active_methods.items():
                        hm = method.explain(x, target=class_idx_gt).astype(np.float32)
                        arrays[method_name] = hm

                    # Random: re-seeded per (sample, model, stage, seed_value)
                    seed_str = (
                        f"{sample_id}|{model_name}|stage{stage_idx}|seed{seed}|gt"
                    )
                    random_seed = int(
                        hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16
                    )
                    random_method = RandomBaselineMethod(bundle.model, seed=random_seed)
                    arrays["Random"] = random_method.explain(
                        x, target=class_idx_gt
                    ).astype(np.float32)

                    arrays["class_idx_gt"] = np.array(class_idx_gt)
                    arrays["stage_idx"] = np.array(stage_idx)
                    arrays["stage_name"] = np.array(stage_name)
                    arrays["seed_value"] = np.array(seed)
                    arrays["model_name"] = np.array(model_name)

                    atomic_savez(out_path, **arrays)
                    n_computed_local += 1

                # --- 8c. Per-(stage, seed) summary with global ETA ---
                stage_seed_elapsed = time.time() - stage_seed_start
                n_iterations_done += 1
                overall_elapsed = time.time() - overall_start
                avg_per_iter = overall_elapsed / n_iterations_done
                eta_s = avg_per_iter * (n_iterations_total - n_iterations_done)
                eta_min = eta_s / 60

                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"  [{ts}] {desc}: "
                    f"{n_computed_local} computed, {n_skipped_local} skipped, "
                    f"{stage_seed_elapsed:.1f}s "
                    f"| progress {n_iterations_done}/{n_iterations_total} "
                    f"| global ETA {eta_min:.1f} min"
                )

                n_computed_total += n_computed_local
                n_skipped_total += n_skipped_local

                # --- 8d. Tear down before next (stage, seed) ---
                del bundle, methods, active_methods
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

except KeyboardInterrupt:
    interrupted = True
    print("\n\nInterrupted by user. Progress is preserved. re-run to resume.")


# --- 9. Final summary ------------------------------------------------------
overall_elapsed = time.time() - overall_start
print(f"\n{'='*70}\nRun summary\n{'='*70}")
print(f"Mode:           {RUN_MODE}")
print(f"Samples:        {n_samples}")
print(f"Iterations:     {n_iterations_done} / {n_iterations_total}")
print(f"Total computed: {n_computed_total}")
print(f"Total skipped:  {n_skipped_total}")
print(f"Wall-clock:     {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
if n_computed_total > 0:
    print(f"Per-heatmap:    {overall_elapsed / n_computed_total * 1000:.0f} ms "
          f"(averaged across all methods)")
if interrupted:
    print("Status: INTERRUPTED")
else:
    print("Status: COMPLETE")
print(f"Output dir:     {HEATMAPS_DIR}")

# --- 10. Extrapolation to full manifest-------------------------------------
# Auto-extrapolate to "full" if we are NOT already in full mode and we
# actually computed something. This is a guideline, not a guarantee:
# the smoke and estimate runs use the FIRST n samples, which may differ
# from the average sample (some images may be slower to load, etc.).
if RUN_MODE != "full" and n_computed_total > 0 and not interrupted:
    full_n = n_total_in_manifest
    naive_scale = full_n / n_samples
    extrapolated_naive_s = overall_elapsed * naive_scale

    print()
    print(f"{'='*70}")
    print(f"Extrapolation to full manifest ({full_n} samples)")
    print(f"{'='*70}")
    print(f"Naive scaling (linear with samples): "
          f"{extrapolated_naive_s/60:.1f} min "
          f"= {extrapolated_naive_s/3600:.2f} h")
    print("Note: this includes both per-sample work and per-iteration overhead.")
    print("Real full-run time will be slightly LESS because the overhead")
    print("(model load, method construction) is amortized over more samples.")