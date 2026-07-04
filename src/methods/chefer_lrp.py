# -*- coding: utf-8 -*-
"""
Manual Chefer-style LRP for Vision Transformers.

Provides two attribution methods:
    CheferLRPMethod          - for standard ViT models (timm's vit_base_patch16_224).
    CheferLRPMobileViTMethod - for hybrid MobileViT models, applying
                               Chefer-style relevance propagation to the
                               transformer stages only.

Both classes install AttentionWithCapture wrappers in place of the original
attention modules, so post-softmax attention matrices and their gradients
can be captured during forward and backward passes for relevance computation.

Important: heatmap computation requires gradient tracking. The internal
forward pass during explain() runs WITHOUT torch.no_grad(), as it must.
Running the wrapped model under torch.no_grad() externally is safe (no
crash), but the attention matrices captured in that mode will not have
usable gradients, so explain() must not be invoked from such a context.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import ExplanationMethod


class AttentionWithCapture(nn.Module):
    """
    Drop-in replacement for timm's vit Attention, exposing the attention
    matrix after softmax for downstream relevance propagation.

    Computes exactly the same forward as the original, but captures A 
    in self.attn_matrix.
    """

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        # Reuse the original parameters - no need to re-init weights.
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.attn_drop = original_attn.attn_drop
        self.proj_drop = original_attn.proj_drop
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = original_attn.scale

       # Sanity check: this implementation assumes q_norm and k_norm are Identity
       # (as they are in timm's standard ViT). Raise if a non-trivial norm is
       # present, since it would require additional handling in the forward pass.
        for norm_name in ("q_norm", "k_norm"):
            if hasattr(original_attn, norm_name):
                norm_mod = getattr(original_attn, norm_name)
                if not isinstance(norm_mod, nn.Identity):
                    raise NotImplementedError(
                        f"AttentionWithCapture assumes {norm_name} is Identity; "
                        f"got {type(norm_mod).__name__}."
                    )

        # Buffer for the captured attention matrix (set during forward).
        # Shape after forward: (B, num_heads, N, N)
        self.attn_matrix: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:

        if attn_mask is not None:
            raise NotImplementedError(
                "AttentionWithCapture does not support attn_mask; "
                "Chefer-style LRP assumes unmasked self-attention."
            )
        if is_causal:
            raise NotImplementedError(
                "AttentionWithCapture does not support causal attention."
            )

        B, N, D = x.shape

        # Compute Q, K, V in one linear pass, then split.
        qkv = self.qkv(x)                                     # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # each (B, H, N, head_dim)

        # Attention scores and softmax.
        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, N, N)
        attn = attn.softmax(dim=-1)

        # CAPTURE: store a reference to the post-softmax matrix.
        # We need it both for value and for gradient computation later, so we
        # keep it in the computation graph (no detach) and request that its
        # gradient be retained after backward (because it is a non-leaf tensor).

        # Only retain gradient when we are actually in a grad-tracking context.
        # In torch.no_grad() the matrix has requires_grad=False and retain_grad
        # would raise. We still want to capture the matrix itself, just without
        # gradient retention.
        if attn.requires_grad:
            attn.retain_grad()
        self.attn_matrix = attn

        attn = self.attn_drop(attn)

        # Weighted values and projection.
        out = (attn @ v)                                      # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)            # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def install_attention_capture(model: nn.Module) -> List[AttentionWithCapture]:
    """
    Walk the model, replace every timm vit Attention with an AttentionWithCapture
    that wraps it. Returns the list of wrapper instances in order so the caller
    can read out their .attn_matrix buffers after a forward pass.
    """
    wrappers: List[AttentionWithCapture] = []

    for block in model.blocks:
        wrapper = AttentionWithCapture(block.attn)
        block.attn = wrapper
        wrappers.append(wrapper)

    return wrappers


def _compute_chefer_relevance(wrappers: List[AttentionWithCapture]) -> torch.Tensor:
    """
    Combine captured attention matrices and gradients into a token-token
    relevance matrix, following Chefer et al. (2021).

    Per block:
        A_eff = mean_over_heads( ReLU(grad_A ⊙ A) )
        A_eff = A_eff + I
        A_eff = row-normalize(A_eff)

    Then accumulate across blocks via matrix multiplication.

    Returns
    -------
    R : torch.Tensor of shape (N, N)
        Token-to-token relevance matrix.
    """
    device = wrappers[0].attn_matrix.device
    N = wrappers[0].attn_matrix.shape[-1]  # 197 for ViT-B/16

    # Start with the identity: each token is fully relevant to itself.
    R = torch.eye(N, device=device)

    for w in wrappers:
        A = w.attn_matrix         # (1, H, N, N)
        gA = A.grad               # (1, H, N, N)

        # Gradient-weighted attention, clipped at zero (Chefer's ReLU).
        # Squeeze the batch dim (we work with batch size 1).
        gradcam_like = (gA * A).clamp(min=0)            # (1, H, N, N)
        gradcam_like = gradcam_like.squeeze(0)          # (H, N, N)

        # Average across heads.
        A_eff = gradcam_like.mean(dim=0)                # (N, N)

        # Add identity (Chefer's modeling of the skip connection).
        A_eff = A_eff + torch.eye(N, device=device)     # (N, N)

        # Row-normalize so each row sums to 1 (conservation per block).
        A_eff = A_eff / A_eff.sum(dim=-1, keepdim=True)

        # Accumulate.
        R = A_eff @ R

    return R


class CheferLRPMethod(ExplanationMethod):
    """
    Manual Chefer-style LRP for Vision Transformers.

    Wraps the model's attention modules in-place so attention matrices and
    their gradients can be captured during forward and backward passes.
    Per block: A_eff = mean_over_heads(ReLU(grad ⊙ A)) + I, row-normalized.
    Per model: R = prod_b A_eff_b, heatmap = reshape(R[CLS, patch_tokens], 14, 14).

    Parameters
    ----------
    model : torch.nn.Module
        A timm ViT model. The constructor wraps its attention modules in-place.
        After construction, the model is numerically identical but captures the
        data we need.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model=model, name="Chefer-LRP")
        self.wrappers = install_attention_capture(model)

    def explain(self, x: torch.Tensor, target: int) -> np.ndarray:

        # Forward + backward to populate matrices and gradients.
        self.model.zero_grad()
        logits = self.model(x)
        target_signal = torch.zeros_like(logits)
        target_signal[0, target] = 1.0
        logits.backward(gradient=target_signal)

        # Combine into a token-token relevance matrix.
        R = _compute_chefer_relevance(self.wrappers)    # (N, N)

        # Read the CLS row, drop the CLS-to-CLS entry.
        cls_relevance = R[0, 1:]                        # (N-1,) = (196,)

        # Reshape patch tokens to 14x14.
        side = int(cls_relevance.numel() ** 0.5)        # 14
        heatmap_low = cls_relevance.reshape(side, side) # (14, 14)

        # Upsample to input resolution (224x224 typically) with bilinear interp.
        H, W = x.shape[-2:]
        heatmap = torch.nn.functional.interpolate(
            heatmap_low[None, None, :, :],              # (1, 1, 14, 14)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1].
        return self._normalize(heatmap)


def install_attention_capture_mobilevit(
    model: nn.Module,
) -> List[List[AttentionWithCapture]]:
    """
    Walk a timm MobileViT model and replace the Attention modules inside
    every MobileVitBlock's transformer with AttentionWithCapture wrappers.

    Returns a list-of-lists: the outer list corresponds to MobileVitBlocks
    (in spatial order), the inner lists contain the wrappers of the
    transformer blocks within each MobileVitBlock.

    For mobilevit_s with 3 MobileVitBlocks each containing 2 transformer
    blocks, the result is [[w00, w01], [w10, w11], [w20, w21]].
    """
    wrappers_per_block: List[List[AttentionWithCapture]] = []

    for stage in model.stages:
        for sub in stage.children():
            if type(sub).__name__ != "MobileVitBlock":
                continue
            block_wrappers: List[AttentionWithCapture] = []
            for transformer_block in sub.transformer:
                wrapper = AttentionWithCapture(transformer_block.attn)
                transformer_block.attn = wrapper
                block_wrappers.append(wrapper)
            wrappers_per_block.append(block_wrappers)

    return wrappers_per_block


def _compute_per_slot_token_importance(
    wrappers: List[AttentionWithCapture],
) -> torch.Tensor:
    """
    For one MobileVitBlock, accumulate Chefer-style relevance through its
    inner transformer blocks - separately for each unfold slot.

    Inputs:
        wrappers: list of length T (number of transformer blocks in this
                  MobileVitBlock; typically 2).
                  Each attn_matrix has shape (S, H, N, N) where:
                    S = number of unfold slots (typically 4)
                    H = number of attention heads
                    N = number of patch positions within a slot

    Returns:
        Tensor of shape (S, N) - per slot, the row-mean of the accumulated
        relevance matrix.
    """
    device = wrappers[0].attn_matrix.device
    S, _, N, _ = wrappers[0].attn_matrix.shape

    # Identity-of-slots: (S, N, N), one identity matrix per slot.
    eye = torch.eye(N, device=device).unsqueeze(0).expand(S, N, N).contiguous()
    R = eye.clone()

    for w in wrappers:
        A = w.attn_matrix              # (S, H, N, N)
        gA = A.grad                    # (S, H, N, N)

        # Per-slot grad-weighted attention, ReLU-clipped, head-averaged.
        combined = (gA * A).clamp(min=0)              # (S, H, N, N)
        A_eff = combined.mean(dim=1)                  # (S, N, N)

        # Add identity (skip connection model), per slot.
        A_eff = A_eff + eye                           # (S, N, N)
        # Row-normalize, per slot.
        A_eff = A_eff / A_eff.sum(dim=-1, keepdim=True)

        # Accumulate per slot: torch's bmm respects the batch dimension.
        R = torch.bmm(A_eff, R)                       # (S, N, N)

    # Per-slot row-mean of R: how much relevance each token receives on
    # average from others within its slot.
    return R.mean(dim=-1)                             # (S, N)


def _slots_to_spatial(
    per_slot_importance: torch.Tensor,
    patch_h: int = 2,
    patch_w: int = 2,
) -> torch.Tensor:
    """
    Reinterleave per-slot per-token importance back into a single 2D map.

    Inputs:
        per_slot_importance: (S, N) where S = patch_h * patch_w and
                              N is a perfect square (side*side).
        patch_h, patch_w: MobileViT's sub-patch dimensions (2x2 default).

    Returns:
        2D tensor of shape (side*patch_h, side*patch_w). Each value comes
        from exactly one (slot, token) pair, placed at the position the
        unfold operation originally took it from.
    """
    S, N = per_slot_importance.shape
    if S != patch_h * patch_w:
        raise ValueError(
            f"Expected {patch_h*patch_w} slots, got {S}."
        )
    side = int(round(N ** 0.5))
    if side * side != N:
        raise ValueError(
            f"Token count {N} per slot is not a perfect square."
        )

    # Reshape per-slot vectors to (S, side, side) - each slot is a small grid
    # of token-importance values.
    per_slot_grid = per_slot_importance.reshape(S, side, side)

    # Allocate the full spatial map and fill it by interleaving slots.
    # Slot s corresponds to position (s // patch_w, s % patch_w) within each
    # patch_h x patch_w sub-patch.
    out_h = side * patch_h
    out_w = side * patch_w
    out = torch.zeros(out_h, out_w, device=per_slot_importance.device)

    for s in range(S):
        r_off = s // patch_w
        c_off = s % patch_w
        # Place per_slot_grid[s] at every position (r_off, c_off) of every
        # patch_h x patch_w sub-patch.
        out[r_off::patch_h, c_off::patch_w] = per_slot_grid[s]

    return out


class CheferLRPMobileViTMethod(ExplanationMethod):
    """
    Chefer-style LRP for the transformer stages of timm's MobileViT-S.

    Parameters
    ----------
    model : torch.nn.Module
        A timm MobileViT model. Constructor wraps its attention modules
        in-place. The model remains numerically identical (forward
        equivalence verified for the ViT case in scripts/10).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model=model, name="Chefer-LRP (hybrid)")
        self.wrappers_per_block = install_attention_capture_mobilevit(model)
        if not self.wrappers_per_block:
            raise ValueError(
                "No MobileVitBlock found in the model. "
                "CheferLRPMobileViTMethod expects a timm MobileViT-style model."
            )

    def explain(
        self,
        x: torch.Tensor,
        target: int,
        return_per_block_maps: bool = False,
    ):
        """
        Compute the hybrid heatmap.

        If return_per_block_maps is True, return a tuple:
            (final_heatmap, [per_block_heatmap_1, per_block_heatmap_2, ...])
        where each per_block_heatmap is already upsampled to input size and
        independently normalized to [0, 1]. The list is in the order of the
        MobileVitBlocks (early -> late).

        Default behaviour (return_per_block_maps=False) is unchanged: returns
        a single 2D numpy array of the final aggregated heatmap.
        """
        # Forward + backward to populate matrices and gradients.
        self.model.zero_grad()
        logits = self.model(x)
        target_signal = torch.zeros_like(logits)
        target_signal[0, target] = 1.0
        logits.backward(gradient=target_signal)

        # Per-MobileVitBlock: 2D spatial relevance map.
        per_block_maps: List[torch.Tensor] = []
        for block_wrappers in self.wrappers_per_block:
            per_slot = _compute_per_slot_token_importance(block_wrappers)
            spatial = _slots_to_spatial(per_slot, patch_h=2, patch_w=2)
            per_block_maps.append(spatial)

        # Upsample each per-block map to the input resolution for both the
        # final aggregation and (optionally) for per-block inspection.
        H, W = x.shape[-2:]
        upsampled: List[torch.Tensor] = []
        for grid in per_block_maps:
            up = F.interpolate(
                grid[None, None, :, :].float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            upsampled.append(up)

        # Final heatmap: mean of upsampled per-block maps.
        combined = torch.stack(upsampled, dim=0).mean(dim=0)
        final = self._normalize(combined.detach().cpu().numpy())

        if not return_per_block_maps:
            return final

        # Per-block: independently normalize each upsampled map for fair
        # visual comparison (each block has its own scale).
        per_block_normed = [
            self._normalize(u.detach().cpu().numpy()) for u in upsampled
        ]
        return final, per_block_normed


