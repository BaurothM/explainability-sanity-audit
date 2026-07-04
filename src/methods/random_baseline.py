# -*- coding: utf-8 -*-
"""
Random baseline explanation method.

Returns a random heatmap, independent of the model and (optionally) the input.
This is NOT a real explanation - it is a control / lower bound:

- In sanity checks (parameter randomization): a real method's explanation should
  change when the model is destroyed. Random doesn't know about the model, so it
  defines the "knows nothing" reference point.
- In faithfulness tests (pixel flipping): removing pixels in random order should
  degrade accuracy slowly. A real method must beat this to be worth anything.

Random calibrates the scale against which all other methods are judged.

Scope note: this class is designed for single-image use (smoke tests,
demos, side-by-side method comparisons on one input). For multi-image
runs across multiple models or targets, such as the dataset-axis batch
runner in scripts/22, Random must be re-seeded per (sample, model,
target) combination to remain a valid baseline on every comparison axis.
The batch runner handles this re-seeding in-line; do not rely on this
class's default constructor seed in batch contexts.
"""

import numpy as np
import torch

from .base import ExplanationMethod


class RandomBaselineMethod(ExplanationMethod):
    """
    Random attribution baseline.

    Produces uniform-random noise at the input's spatial resolution.
    A fixed seed can be set for reproducibility.
    """

    def __init__(self, model: torch.nn.Module, seed: int | None = None):
        super().__init__(model=model, name="Random")
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:
        # We ignore model and target entirely - that is the whole point.
        # Spatial size taken from the input tensor (1, C, H, W).
        height, width = x.shape[-2:]
        heatmap = self._rng.random((height, width))
        return self._normalize(heatmap)

