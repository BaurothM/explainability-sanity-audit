# -*- coding: utf-8 -*-
"""
Diagnostic: print the top-level named children for each model, in order,
plus one level deeper for the containers we plan to split (resnet50.layer*,
vit_base.blocks, mobilevit_s.stages).

Used to anchor the randomization schedule (src/randomization_schedule.py)
to the actual layer names timm exposes. Console output only.

Run to verify the printed names match what is used in the schedule
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_children(module, prefix="  "):
    """Print named_children with their type and parameter count."""
    for name, child in module.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"{prefix}{name:30s}  ({type(child).__name__}, {n_params:,} params)")


# --- 1. Top-level for all three models ---
for model_name in ["resnet50", "vit_base", "mobilevit_s"]:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  -  top-level")
    print(f"{'='*60}")
    bundle = load_model(model_name, device)
    print_children(bundle.model)
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --- 2. ResNet50: one level deeper inside each layer* ---
print(f"\n{'='*60}")
print("resnet50 - inside each layer*")
print(f"{'='*60}")
bundle = load_model("resnet50", device)
for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
    print(f"\n{layer_name} children:")
    print_children(getattr(bundle.model, layer_name), prefix="    ")
del bundle
if device.type == "cuda":
    torch.cuda.empty_cache()


# --- 3. ViT-B/16: blocks named children ---
print(f"\n{'='*60}")
print("vit_base - blocks children")
print(f"{'='*60}")
bundle = load_model("vit_base", device)
print_children(bundle.model.blocks, prefix="    ")
del bundle
if device.type == "cuda":
    torch.cuda.empty_cache()


# --- 4. MobileViT-S: stages named children ---
print(f"\n{'='*60}")
print("mobilevit_s - stages children")
print(f"{'='*60}")
bundle = load_model("mobilevit_s", device)
print_children(bundle.model.stages, prefix="    ")
del bundle
if device.type == "cuda":
    torch.cuda.empty_cache()

print("\nDone.")