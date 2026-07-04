# -*- coding: utf-8 -*-
"""
Layer-wise Relevance Propagation (LRP) explanation method.

This version supports CNN-style models (ResNet; exploratorly for MobileViT) 
using zennit's EpsilonGammaBox composite - the established "best-practice" 
combination of LRP rules for image classification CNNs.

ViT support is NOT included here. ViTs need transformer-specific propagation
rules (Chefer et al., 2021). Trying to apply CNN-style LRP to a ViT will 
be flagged.
"""

import numpy as np
import torch
from zennit.composites import EpsilonGammaBox

from .base import ExplanationMethod


class LRPMethod(ExplanationMethod):
    """
    LRP for CNN-style architectures, via zennit's EpsilonGammaBox composite.

    Parameters
    ----------
    model : torch.nn.Module
        A CNN or CNN-ending hybrid (no pure transformer).
    low, high : float
        The min/max pixel values of the *normalized* input space. The Box rule
        uses these to bound the first layer's relevance attribution. For
        ImageNet normalization with the standard mean/std, the per-channel
        bounds vary, but using global (-3, 3) is a safe conservative choice.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        low: float = -3.0,
        high: float = 3.0,
    ):
        super().__init__(model=model, name="LRP")
        self.composite = EpsilonGammaBox(low=low, high=high)

    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:
        # zennit's Gradient attributor applies the composite during backward
        # and returns the relevance map at the input.
        x = x.clone().requires_grad_(True)

        # The target signal: a one-hot vector at the target class.
        # Shape must match the model's output.
        with self.composite.context(self.model) as modified_model:
            output = modified_model(x)
            target_onehot = torch.zeros_like(output)
            target_onehot[0, target] = 1.0
            relevance = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=target_onehot,
            )[0]

        # relevance shape: (1, C, H, W) -> collapse channels to (H, W)
        attr = relevance.squeeze(0).detach().cpu().numpy()
        heatmap = attr.sum(axis=0)  # signed sum across channels
        # Take absolute value: we want "how relevant", not "positive vs negative"
        heatmap = np.abs(heatmap)
        return self._normalize(heatmap)

