# -*- coding: utf-8 -*-
"""
Sanity check: load all three models and verify target layers exist.
Verifies that all model bundles load correctly and have valid target layers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

for name in ["resnet50", "vit_base", "mobilevit_s"]:
    print(f"--- Loading {name} ---")
    try:
        bundle = load_model(name, device)
        n_params = sum(p.numel() for p in bundle.model.parameters())
        print(f"  OK: family={bundle.family}, params={n_params:,}")
        print(f"  Target layer type: {type(bundle.target_layer).__name__}")
        print(f"  Reshape transform: {'yes' if bundle.reshape_transform else 'none'}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

