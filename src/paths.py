# -*- coding: utf-8 -*-
"""
Project path utilities.
Resolves the project root regardless of where a script is executed from.
"""

from pathlib import Path

# This file lives at <PROJECT_ROOT>/src/paths.py
# So the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
# ImageNet-1k validation set (Kaggle: titericz/imagenet1k-val), 
# extracted as 1000 synset-ID folders containing JPEGs.
# URL "https://www.kaggle.com/datasets/titericz/imagenet1k-val/data?select=imagenet-val"
# Not tracked in git. User must download separately.
IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Ensure they exist when imported
for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)