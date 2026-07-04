# -*- coding: utf-8 -*-
"""
GradCAM explanation method.
Uses the `grad-cam` library (pytorch_grad_cam), which supports both CNN-style
feature maps and transformer token sequences via a reshape_transform.
"""

from typing import Callable, Optional

import numpy as np
import torch
from pytorch_grad_cam import GradCAM

from .base import ExplanationMethod
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMMethod(ExplanationMethod):
    """
    GradCAM (Selvaraju et al., 2017).

    Computes attribution at a target layer's feature maps, weighted by
    class-conditional gradients, then upsamples to input size.

    For transformer models, a reshape_transform folds the token sequence
    into a spatial grid so GradCAM can treat it like a CNN feature map.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        reshape_transform: Optional[Callable] = None,
    ):
        super().__init__(model=model, name="GradCAM")
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self._cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:
        # grad-cam returns a (B, H, W) numpy array already upsampled to input size
        grayscale_cam = self._cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(target)],
        )
        # Take the first (and only) image in the batch
        heatmap = grayscale_cam[0]  # (H, W)
        return self._normalize(heatmap)

