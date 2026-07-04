# -*- coding: utf-8 -*-
"""
Build a stratified subset of ImageNet-1k validation set and write a manifest.

The manifest is a CSV that drives all subsequent dataset-axis scripts. It
contains, for each selected sample: a sample_id, the relative image path,
the ImageNet class index, the synset ID, and a within-class sample counter.

Configuration is at the top of the script. To switch between stress-test
and full run, change N_CLASSES and SAMPLES_PER_CLASS only.

The script verifies after generation that class_idx values are consistent
with what timm-pretrained ResNet50 actually predicts (Top-5 check). This
catches sorting bugs and off-by-one errors that would otherwise corrupt
all downstream comparisons.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, IMAGENET_VAL_DIR, DATA_DIR
from src.models import load_model

import csv
import torch
import timm
from PIL import Image

# --- 1. Configuration ---
N_CLASSES = 500            # stress test: 50; full run: 500
SAMPLES_PER_CLASS = 2     # stress test: 1; full run: 2
SEED = 42

MANIFEST_DIR = DATA_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
manifest_filename = f"subset_{N_CLASSES}x{SAMPLES_PER_CLASS}_seed{SEED}.csv"
manifest_path = MANIFEST_DIR / manifest_filename

print(f"Building subset: {N_CLASSES} classes x {SAMPLES_PER_CLASS} samples")
print(f"Manifest target: {manifest_path}")

# --- 2. List all class folders, sorted alphabetically ---
# Alphabetical sort of synset IDs is the canonical ImageNet ordering,
# and (we will verify) matches the class index that timm models output.
all_class_dirs = sorted(
    d for d in IMAGENET_VAL_DIR.iterdir()
    if d.is_dir() and d.name.startswith("n")  # synset IDs start with 'n'
)
print(f"Found {len(all_class_dirs)} class folders.")
assert len(all_class_dirs) == 1000, \
    f"Expected 1000 class folders, got {len(all_class_dirs)}"

# --- 3. Select class subset (deterministic via seed) ---
rng = random.Random(SEED)
if N_CLASSES < 1000:
    selected_class_indices = sorted(rng.sample(range(1000), N_CLASSES))
else:
    selected_class_indices = list(range(1000))
selected_class_dirs = [all_class_dirs[i] for i in selected_class_indices]
print(f"Selected {len(selected_class_dirs)} classes.")

# --- 4. Pick samples per class ---
rows = []
sample_id = 0
for class_idx, class_dir in zip(selected_class_indices, selected_class_dirs):
    synset_id = class_dir.name
    # All JPEGs in this class folder, sorted for determinism
    all_imgs = sorted(class_dir.glob("*.JPEG"))
    if len(all_imgs) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Class {synset_id} has only {len(all_imgs)} images, "
            f"need {SAMPLES_PER_CLASS}"
        )
    # Deterministic sampling: seed once per class to avoid order dependence
    class_rng = random.Random(SEED + class_idx)
    chosen = class_rng.sample(all_imgs, SAMPLES_PER_CLASS)
    for sample_in_class, img_path in enumerate(chosen):
        rows.append({
            "sample_id": sample_id,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "class_idx": class_idx,
            "synset_id": synset_id,
            "sample_in_class": sample_in_class,
        })
        sample_id += 1

print(f"Total samples in manifest: {len(rows)}")

# --- 5. Write manifest CSV ---
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["sample_id", "image_path", "class_idx",
                       "synset_id", "sample_in_class"]
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote manifest to {manifest_path}")

# Show first 3 rows for visual sanity check
print("\nFirst 3 rows:")
for row in rows[:3]:
    print(f"  {row}")

# --- 6. Verify class_idx consistency with timm ResNet50 ---
# For each sample, predict with ResNet50 and check whether the manifest's
# class_idx is in the Top-5 predictions. Mismatches outside Top-5 likely
# indicate manifest bugs. Mismatches that are still in Top-5 are model
# misclassifications, not bugs.
print("\n--- Verification: class_idx vs. ResNet50 Top-5 predictions ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model("resnet50", device)
data_config = timm.data.resolve_model_data_config(bundle.model)
transform = timm.data.create_transform(**data_config, is_training=False)

top1_correct = 0
top5_correct = 0
suspect_rows = []  # rows where class_idx is NOT in top-5

for row in rows:
    img_path = PROJECT_ROOT / row["image_path"]
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bundle.model(x)
        top5 = logits.topk(5, dim=1).indices[0].cpu().tolist()
    expected = row["class_idx"]
    if expected == top5[0]:
        top1_correct += 1
    if expected in top5:
        top5_correct += 1
    else:
        suspect_rows.append({
            "sample_id": row["sample_id"],
            "synset_id": row["synset_id"],
            "expected_class_idx": expected,
            "model_top3": top5[:3],
        })

n = len(rows)
print(f"Top-1 accuracy: {top1_correct}/{n} = {top1_correct/n*100:.1f}%")
print(f"Top-5 accuracy: {top5_correct}/{n} = {top5_correct/n*100:.1f}%")
print(f"Suspect rows (class_idx NOT in Top-5): {len(suspect_rows)}")

# Expected ballpark for ResNet50 on ImageNet-Val: ~76% Top-1, ~93% Top-5.
# If Top-5 is far below 93% (e.g., 50%), or if suspect rows show a
# systematic pattern (e.g., all class indices off by exactly N), this is
# a manifest bug, not model noise.
if suspect_rows:
    print("\nSuspect rows (first 5):")
    for s in suspect_rows[:5]:
        print(f"  {s}")

if top5_correct / n < 0.85:
    print("\nWARNING: Top-5 accuracy unusually low. Investigate manifest "
          "for sorting/indexing bugs before proceeding.")
else:
    print("\nVerification PASSED: class_idx values appear consistent with timm.")