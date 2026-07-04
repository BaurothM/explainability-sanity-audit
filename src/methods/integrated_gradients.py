# -*- coding: utf-8 -*-
"""
Integrated Gradients explanation method.
Wrapper around captum's IntegratedGradients.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients as CaptumIG

from .base import ExplanationMethod


class IntegratedGradientsMethod(ExplanationMethod):
    """
    Integrated Gradients (Sundararajan, Taly, Yan, 2017).

    Integrates gradients along a straight-line path from a baseline to the input.
    Default baseline is a zero (black) image.
    """

    def __init__(self, model: torch.nn.Module, n_steps: int = 50):
        super().__init__(model=model, name="IntegratedGradients")
        self.n_steps = n_steps
        self._captum = CaptumIG(model)

    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:
        baseline = torch.zeros_like(x)
        attributions = self._captum.attribute(
            inputs=x,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
        )
        # (1, C, H, W) -> (H, W): absolute sum across channels
        attr = attributions.squeeze(0).detach().cpu().numpy()
        heatmap = np.abs(attr).sum(axis=0)
        return self._normalize(heatmap)