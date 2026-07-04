# -*- coding: utf-8 -*-
"""
Inspect the architecture of timm's ViT-B/16, focusing on the structure of
a single transformer block. This is Preparation for implementing 
Chefer-style LRP. Identifies which submodules exist, their order, and where 
residual connections sit.

Produces no figure, all output is printed to the console.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model

import torch

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model("vit_base", device)
model = bundle.model

# --- 2. Top-level structure ---
print("=" * 70)
print("TOP-LEVEL STRUCTURE")
print("=" * 70)
print(f"Model class: {type(model).__name__}")
print(f"Number of transformer blocks: {len(model.blocks)}")
print(f"Top-level attributes (children):")
for name, child in model.named_children():
    print(f"  {name:20s} -> {type(child).__name__}")

# --- 3. One transformer block, in detail ---
print()
print("=" * 70)
print("ONE TRANSFORMER BLOCK (model.blocks[0])")
print("=" * 70)
block = model.blocks[0]
print(f"Block class: {type(block).__name__}")
print(f"Submodules of the block:")
for name, child in block.named_children():
    print(f"  {name:20s} -> {type(child).__name__}")

# --- 4. Inside the attention submodule ---
print()
print("=" * 70)
print("INSIDE THE ATTENTION SUBMODULE (model.blocks[0].attn)")
print("=" * 70)
attn = block.attn
print(f"Attention class: {type(attn).__name__}")
for name, child in attn.named_children():
    print(f"  {name:20s} -> {type(child).__name__}")
# Useful scalar attributes if they exist
for attr in ["num_heads", "head_dim", "scale", "qkv_bias"]:
    if hasattr(attn, attr):
        print(f"  attr {attr}: {getattr(attn, attr)}")

# --- 5. Inside the MLP submodule ---
print()
print("=" * 70)
print("INSIDE THE MLP SUBMODULE (model.blocks[0].mlp)")
print("=" * 70)
mlp = block.mlp
print(f"MLP class: {type(mlp).__name__}")
for name, child in mlp.named_children():
    print(f"  {name:20s} -> {type(child).__name__}")

# --- 6. Forward-pass shape trace via hooks ---
print()
print("=" * 70)
print("FORWARD-PASS SHAPE TRACE (first block only)")
print("=" * 70)
# We register forward hooks on every submodule of block 0 and record
# the output shape. This shows the actual data flow at inference time.

shape_trace = []

def make_hook(name):
    def hook(module, inputs, outputs):
        in_shape = tuple(inputs[0].shape) if isinstance(inputs, tuple) and len(inputs) > 0 else None
        out_shape = tuple(outputs.shape) if torch.is_tensor(outputs) else type(outputs).__name__
        shape_trace.append((name, in_shape, out_shape))
    return hook

# Register hooks recursively on all submodules of block 0
handles = []
for name, module in block.named_modules():
    if name == "":  # skip the block itself
        continue
    handles.append(module.register_forward_hook(make_hook(name)))

# Run a dummy input through the whole model so block 0 sees a realistic tensor
dummy = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    model(dummy)

# Clean up
for h in handles:
    h.remove()

# Print the trace
for name, in_shape, out_shape in shape_trace:
    print(f"  {name:30s}  in={str(in_shape):25s}  out={out_shape}")

