# -*- coding: utf-8 -*-
"""
Inspect the architecture of timm's MobileViT-S, in preparation for extending
Chefer-style LRP to the hybrid case.

Three questions:
1. Top-level structure: how are conv stages and transformer stages arranged?
2. Inside a transformer stage: are the transformer blocks structurally
   compatible with our AttentionWithCapture wrapper from chefer_lrp.py?
3. At the boundary: how does the feature map (B, C, H, W) get folded into a
   token sequence (B, N, D) and back?

Produces no figure, all output is printed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model

import torch

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model("mobilevit_s", device)
model = bundle.model

# --- 2. Top-level structure ---
print("=" * 70)
print("TOP-LEVEL STRUCTURE")
print("=" * 70)
print(f"Model class: {type(model).__name__}")
print(f"Top-level children:")
for name, child in model.named_children():
    print(f"  {name:20s} -> {type(child).__name__}")

# --- 3. Stages: identify which are conv and which contain transformers ---
print()
print("=" * 70)
print("STAGES")
print("=" * 70)
# In timm's MobileViT, the main computation lives under `stages` (a Sequential).
# Each stage may be a pure conv stage or a "MobileViT block" containing both
# conv layers and a transformer.
if hasattr(model, "stages"):
    stages_container = model.stages
    print(f"Number of stages: {len(stages_container)}")
    for i, stage in enumerate(stages_container):
        # Look at the top-level children of this stage to get a quick sense
        # of what's inside.
        child_names = [type(c).__name__ for c in stage.children()]
        # Detect whether the stage holds a transformer somewhere inside.
        has_transformer = any(
            "Transformer" in type(m).__name__ or "Attention" in type(m).__name__
            for m in stage.modules()
        )
        marker = "  [contains transformer]" if has_transformer else ""
        print(f"  stage[{i}]: {type(stage).__name__:30s}{marker}")
        for j, c in enumerate(stage.children()):
            print(f"             child[{j}]: {type(c).__name__}")
else:
    print("Model has no .stages attribute - need to inspect differently.")

# --- 4. Find one transformer-bearing stage and inspect it ---
print()
print("=" * 70)
print("ONE TRANSFORMER-BEARING STAGE IN DETAIL")
print("=" * 70)
target_stage_idx = None
target_stage = None
if hasattr(model, "stages"):
    for i, stage in enumerate(model.stages):
        if any(
            "Transformer" in type(m).__name__ or "Attention" in type(m).__name__
            for m in stage.modules()
        ):
            target_stage_idx = i
            target_stage = stage
            break

if target_stage is not None:
    print(f"Stage index: {target_stage_idx}")
    print(f"Stage class: {type(target_stage).__name__}")
    print(f"Submodules (depth=2):")
    for name, module in target_stage.named_modules():
        # Limit depth to avoid output explosion.
        depth = name.count(".")
        if depth <= 1 and name != "":
            indent = "  " * (depth + 1)
            print(f"{indent}{name:30s} -> {type(module).__name__}")
else:
    print("No transformer-bearing stage found - unexpected for mobilevit_s.")

# --- 5. Find one Attention submodule and inspect it ---
print()
print("=" * 70)
print("AN ATTENTION SUBMODULE (the first one we find)")
print("=" * 70)
first_attn = None
first_attn_path = None
for name, module in model.named_modules():
    if "Attention" in type(module).__name__:
        first_attn = module
        first_attn_path = name
        break

if first_attn is not None:
    print(f"Found at: model.{first_attn_path}")
    print(f"Class: {type(first_attn).__name__}")
    print(f"Submodules:")
    for name, child in first_attn.named_children():
        print(f"  {name:20s} -> {type(child).__name__}")
    for attr in ["num_heads", "head_dim", "scale"]:
        if hasattr(first_attn, attr):
            print(f"  attr {attr}: {getattr(first_attn, attr)}")
else:
    print("No Attention module found.")

# --- 6. Forward-pass shape trace ---
print()
print("=" * 70)
print("FORWARD-PASS SHAPE TRACE (selected key submodules)")
print("=" * 70)
# We register hooks on the model's main building blocks and on whatever is
# inside the first transformer-bearing stage. The goal is to see where shapes
# change between (B, C, H, W) and (B, N, D).

shape_trace = []

def make_hook(name):
    def hook(module, inputs, outputs):
        in_shape = (
            tuple(inputs[0].shape)
            if isinstance(inputs, tuple) and len(inputs) > 0 and torch.is_tensor(inputs[0])
            else None
        )
        out_shape = (
            tuple(outputs.shape) if torch.is_tensor(outputs) else type(outputs).__name__
        )
        shape_trace.append((name, in_shape, out_shape))
    return hook

# Register hooks on:
# - each top-level child of model
# - each child of the first transformer-bearing stage
handles = []
for name, child in model.named_children():
    handles.append(child.register_forward_hook(make_hook(f"model.{name}")))
if target_stage is not None:
    for sub_name, sub_module in target_stage.named_modules():
        if sub_name == "":
            continue
        depth = sub_name.count(".")
        # Keep depth <= 2 to avoid drowning in details.
        if depth <= 2:
            handles.append(sub_module.register_forward_hook(
                make_hook(f"stages[{target_stage_idx}].{sub_name}")
            ))

dummy = torch.randn(1, 3, 256, 256, device=device)
# mobilevit_s typically takes 256x256. We pass that to avoid resize warnings.
with torch.no_grad():
    model(dummy)

for h in handles:
    h.remove()

for name, in_s, out_s in shape_trace:
    print(f"  {name:55s}  in={str(in_s):28s} out={out_s}")

