# -*- coding: utf-8 -*-
"""
Batch-runner: generate heatmaps for every (sample, model, method, target)
combination in a manifest and save them as compressed .npz per (sample, model).

Loop order is models-outer, samples-inner: each model is loaded once, all
samples are processed, then the model is unloaded and VRAM is freed before
the next model. This avoids both per-sample model reloads (slow) and
cross-model VRAM accumulation (caused massive slowdowns in earlier tests).

Output layout (per manifest):
    results/heatmaps/<manifest_stem>/sample_XXXXX_<modelname>.npz

Each .npz contains, for every applicable method on this model:
    {method_name}_gt   : (H, W) float32 heatmap for ground-truth class
    {method_name}_pred : (H, W) float32 heatmap for top-1 prediction
plus 0-dim metadata arrays (class_idx_gt, class_idx_pred, model_name).

Resume: on each (sample_id, model_name) pair, if the target .npz already
exists, the pair is skipped. Files are written atomically (.tmp then rename)
so partial writes never look complete.

Determinism note: PyTorch / CUDA are not strictly deterministic by default.
Resumed runs may differ from single-shot runs at the ~1e-6 level, far below
heatmap value scales (~0 to 1). torch.use_deterministic_algorithms(True) is
NOT set, because the slowdown is not worth the precision gain for this audit.
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

import csv
import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm
import gc
import hashlib

# --- 1. Configuration ---
MANIFEST_NAME = "subset_500x2_seed42.csv"
LOG_EVERY_N = 10

MANIFEST_PATH = DATA_DIR / "manifests" / MANIFEST_NAME
HEATMAPS_DIR = RESULTS_DIR / "heatmaps" / MANIFEST_PATH.stem
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["resnet50", "vit_base", "mobilevit_s"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Manifest:     {MANIFEST_PATH}")
print(f"Output dir:   {HEATMAPS_DIR}")

# --- 2. Read manifest ---
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    manifest_rows = list(reader)
n_samples = len(manifest_rows)
print(f"Samples in manifest: {n_samples}")

# --- 3. Method factory ---
def build_methods(bundle):
    """
    Instantiate methods applicable to a given model bundle.

    Mirror of scripts/06's logic. n/a methods return None and are simply
    not written to the output .npz (the absence is the audit's "n/a" signal).
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

# --- 4. Atomic save helper ---
def atomic_savez(path: Path, **arrays):
    """
    Write a compressed .npz atomically: write to a temp file with .npz
    extension first (because np.savez_compressed enforces that extension),
    then rename to the final path. Prevents partial-write files from being
    mistaken for complete output on the next resume.
    """
    # numpy.savez_compressed appends .npz if not present. Use a tmp filename
    # that already ends in .npz so the actual written file matches what
    # we then rename.
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)  # atomic on POSIX and modern Windows

# --- 5. Main loop: models outer, samples inner ---
overall_start = time.time()
n_computed_total = 0
n_skipped_total = 0
interrupted = False

try:
    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}\nModel: {model_name}\n{'='*70}")

        bundle = load_model(model_name, device)
        data_config = timm.data.resolve_model_data_config(bundle.model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        methods = build_methods(bundle)
        # Filter out n/a methods up front to keep the inner loop clean
        active_methods = {k: v for k, v in methods.items() if v is not None}
        print(f"Active methods: {list(active_methods.keys())}")

        model_start = time.time()
        n_computed_model = 0
        n_skipped_model = 0

        pbar = tqdm(manifest_rows, desc=model_name, unit="img")
        for row in pbar:
            sample_id = int(row["sample_id"])
            class_idx_gt = int(row["class_idx"])
            img_rel_path = row["image_path"]

            out_path = HEATMAPS_DIR / f"sample_{sample_id:05d}_{model_name}.npz"

            # Resume: skip if this (sample, model) is already done
            if out_path.exists():
                n_skipped_model += 1
                continue

            # Load and preprocess image
            img = Image.open(PROJECT_ROOT / img_rel_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            # Determine top-1 prediction (for the "pred" target)
            with torch.no_grad():
                class_idx_pred = bundle.model(x).argmax(dim=1).item()

            # Compute heatmaps for both targets, all active methods
            arrays = {}
            for method_name, method in active_methods.items():
                # Random is a special case: per design (see
                # src/methods/random_baseline.py "Scope note"), the
                # constructor-time seed of RandomBaselineMethod is for
                # single-image use. In this batch context, Random must
                # be independently seeded per (sample, model, target)
                # so it serves as a valid noise floor on ALL comparison
                # axes (cross-method, cross-model, cross-target).
                # We therefore re-construct Random per call with a
                # deterministic hash-based seed, and we do NOT copy
                # gt->pred for Random even when class_idx_pred == class_idx_gt.
                if method_name == "Random":
                    for target_label, target_class in [
                        ("gt", class_idx_gt),
                        ("pred", class_idx_pred),
                    ]:
                        seed_str = f"{sample_id}|{model_name}|{target_label}"
                        seed = int(
                            hashlib.sha256(seed_str.encode()).hexdigest()[:8],
                            16,
                        )
                        random_method = RandomBaselineMethod(
                            bundle.model, seed=seed
                        )
                        hm = random_method.explain(
                            x, target=target_class
                        ).astype(np.float32)
                        arrays[f"{method_name}_{target_label}"] = hm
                    continue

                # All other methods: deterministic given (model, input,
                # target), so we can safely copy gt->pred when classes match.
                hm_gt = method.explain(x, target=class_idx_gt).astype(np.float32)
                arrays[f"{method_name}_gt"] = hm_gt
                if class_idx_pred == class_idx_gt:
                    arrays[f"{method_name}_pred"] = hm_gt
                else:
                    hm_pred = method.explain(
                        x, target=class_idx_pred
                    ).astype(np.float32)
                    arrays[f"{method_name}_pred"] = hm_pred

            # Metadata (stored as 0-dim arrays so np.load returns them cleanly)
            arrays["class_idx_gt"] = np.array(class_idx_gt)
            arrays["class_idx_pred"] = np.array(class_idx_pred)
            arrays["model_name"] = np.array(model_name)

            atomic_savez(out_path, **arrays)
            n_computed_model += 1

            # Periodic persistent log line
            if (n_computed_model + n_skipped_model) % LOG_EVERY_N == 0:
                elapsed = time.time() - model_start
                done = n_computed_model + n_skipped_model
                avg = elapsed / max(n_computed_model, 1)
                eta = avg * (n_samples - done)
                ts = datetime.now().strftime("%H:%M:%S")
                tqdm.write(
                    f"  [{ts}] {model_name}: {done}/{n_samples} done "
                    f"(computed {n_computed_model}, skipped {n_skipped_model}), "
                    f"avg {avg:.2f}s/img, ETA {eta:.0f}s"
                )
            
            # Periodic cleanup: torch holds intermediate buffers and captum
            # may accumulate small state across calls. Flushing every N
            # samples keeps the inner loop time stable (without this,
            # MobileViT-IG drifts from ~0.55s/img to ~0.92s/img over 50
            # samples. see the diagnostic scripts 23/24).
            if (n_computed_model + n_skipped_model) % LOG_EVERY_N == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # End of this model: per-model summary
        model_elapsed = time.time() - model_start
        print(f"\n{model_name} done: {n_computed_model} computed, "
              f"{n_skipped_model} skipped, {model_elapsed:.1f}s total")

        n_computed_total += n_computed_model
        n_skipped_total += n_skipped_model

        # Free VRAM before next model is loaded (this is the cleanup
        # that gave the massive speedup in script 20's clean run).
        del bundle, methods, active_methods
        if device.type == "cuda":
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    interrupted = True
    print("\n\nInterrupted by user. Progress is preserved. re-run to resume.")

# --- 6. Final summary ---
overall_elapsed = time.time() - overall_start
print(f"\n{'='*70}")
print("Run summary")
print(f"{'='*70}")
print(f"Total computed: {n_computed_total}")
print(f"Total skipped:  {n_skipped_total}")
print(f"Wall-clock:     {overall_elapsed:.1f}s")
if interrupted:
    print("Status: INTERRUPTED")
else:
    print("Status: COMPLETE")
print(f"Output dir:     {HEATMAPS_DIR}")