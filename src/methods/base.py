# -*- coding: utf-8 -*-
"""
Abstract base class for all explanation methods.
Every method must produce a 2D heatmap of the same spatial size as the input image.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class ExplanationMethod(ABC):
    """
    Base class for post-hoc explanation methods.

    Subclasses configure themselves at construction time (e.g., target layers,
    number of integration steps) and expose a uniform `explain(x, target)` API.
    """

    def __init__(self, model: torch.nn.Module, name: str):
        self.model = model
        self.name = name

    @abstractmethod
    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:
        """
        Compute a 2D attribution heatmap for the given input and target class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (1, C, H, W).
        target : int
            Index of the class to explain.

        Returns
        -------
        np.ndarray
            2D heatmap of shape (H, W), normalized to [0, 1].
        """
        ...

    @staticmethod
    def _normalize(heatmap: np.ndarray) -> np.ndarray:
        """Normalize a heatmap to [0, 1]."""
        h_min = heatmap.min()
        h_max = heatmap.max()
        if h_max - h_min < 1e-8:
            return np.zeros_like(heatmap)
        return (heatmap - h_min) / (h_max - h_min)

