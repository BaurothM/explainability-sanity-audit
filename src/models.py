# -*- coding: utf-8 -*-
"""
Model loading and architecture-specific metadata.

Each model is wrapped together with the information needed to run GradCAM:
- the target layer (the layer whose activations GradCAM weights)
- a reshape_transform (None for CNNs; for transformers, folds the token
  sequence back into a 2D spatial grid)

This is where architecture differences are made explicit. A CNN's feature maps
are already spatial (B, C, H, W). A ViT's tokens are a sequence (B, N, D) and
must be reshaped to (B, D, H, W) before GradCAM can treat them spatially.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import timm
import torch


@dataclass
class ModelBundle:
    """A model plus the metadata needed to explain it."""
    name: str
    model: torch.nn.Module
    target_layer: torch.nn.Module
    reshape_transform: Optional[Callable]
    family: str  # "cnn", "vit", or "hybrid"


def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """
    Fold a ViT token sequence back into a spatial grid.

    Input:  (B, N, D)  where N = 1 (CLS) + H*W patch tokens
    Output: (B, D, H, W)

    We drop the first token (CLS) and reshape the remaining patch tokens.
    For ViT-B/16 at 224x224: 196 patch tokens -> 14x14 grid.
    """
    # Drop CLS token (index 0), keep patch tokens
    result = tensor[:, 1:, :]
    # (B, H*W, D) -> (B, H, W, D)
    result = result.reshape(result.size(0), height, width, result.size(2))
    # (B, H, W, D) -> (B, D, H, W) so it looks like a CNN feature map
    result = result.permute(0, 3, 1, 2)
    return result


def load_model(name: str, device: torch.device) -> ModelBundle:
    """
    Load a pretrained model by short name and attach GradCAM metadata.

    Supported names: "resnet50", "vit_base", "mobilevit_s".
    """
    if name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True)
        model.eval().to(device)
        return ModelBundle(
            name="resnet50",
            model=model,
            target_layer=model.layer4[-1],  # last bottleneck block
            reshape_transform=None,
            family="cnn",
        )

    elif name == "vit_base":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.eval().to(device)
        # For timm ViT, the norm before the final block of the last
        # transformer block is a common GradCAM target.
        return ModelBundle(
            name="vit_base",
            model=model,
            target_layer=model.blocks[-1].norm1,
            reshape_transform=_vit_reshape_transform,
            family="vit",
        )

    elif name == "mobilevit_s":
        model = timm.create_model("mobilevit_s", pretrained=True)
        model.eval().to(device)
        # MobileViT ends in conv-like stages. The final conv is spatial,
        # so no reshape_transform is needed here. We target the last
        # normalization/conv stage.
        return ModelBundle(
            name="mobilevit_s",
            model=model,
            target_layer=model.final_conv,
            reshape_transform=None,
            family="hybrid",
        )

    else:
        raise ValueError(f"Unknown model name: {name}")

