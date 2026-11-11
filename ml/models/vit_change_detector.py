"""Siamese Vision Transformer (ViT) change detector with multi-depth spatial token decoding."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.blocks import (
    BaseChangeDetector,
    ConvBNAct,
    DecoderUpBlock,
    DoubleConv,
    TemporalFusionBlock,
    create_timm_backbone,
)


class ViTChangeDetector(BaseChangeDetector):
    """Siamese Vision Transformer change detector built on ``timm`` ViT backbones.

    Extracts spatial patch-token grids from multiple Transformer encoder depths,
    reshapes patch tokens into 2D spatial feature maps, fuses temporal differences
    at each depth, and progressively decodes them to full-resolution change logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        backbone_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        allow_random_fallback: bool = True,
        decoder_channels: Sequence[int] = (192, 128, 96, 64),
    ) -> None:
        backbone, weight_src = create_timm_backbone(
            backbone_name,
            pretrained=pretrained,
            allow_random_fallback=allow_random_fallback,
            dynamic_img_size=True,
            in_chans=in_channels,
            num_classes=0,
        )

        super().__init__(
            model_name="vit",
            architecture_family="Vision Transformer (Siamese ViT)",
            backbone_name=backbone_name,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            backbone_weights_source=weight_src,
        )

        self.backbone = backbone
        self.embed_dim = int(getattr(backbone, "embed_dim", 192))
        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
        patch_size_attr = getattr(backbone.patch_embed, "patch_size", (16, 16))
        self.patch_size = (
            int(patch_size_attr[0])
            if isinstance(patch_size_attr, (tuple, list))
            else int(patch_size_attr)
        )

        num_blocks = len(self.backbone.blocks)
        # Sample 4 evenly distributed encoder blocks (early, early-mid, late-mid, final)
        self.feature_indices = sorted(
            {
                max(0, num_blocks // 4 - 1),
                max(0, num_blocks // 2 - 1),
                max(0, (3 * num_blocks) // 4 - 1),
                num_blocks - 1,
            }
        )
        # Ensure exactly 4 indices even for shallow test backbones
        while len(self.feature_indices) < 4:
            self.feature_indices.insert(0, 0)

        d_ch = list(decoder_channels)
        if len(d_ch) != 4:
            raise ValueError("ViTChangeDetector decoder_channels must contain 4 channel widths.")
        self.decoder_channels = d_ch  # [H/16, H/8, H/4, H/2]

        # Fuse temporal features at 4 transformer depths
        self.fusion_blocks = nn.ModuleList(
            [TemporalFusionBlock(self.embed_dim, d_ch[3 - idx]) for idx in range(4)]
        )
        # Reassemble multi-depth spatial grids to pyramid scales (H/16, H/8, H/4, H/2)
        self.reassemble_2x = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvBNAct(d_ch[1], d_ch[1], kernel_size=3, padding=1),
        )
        self.reassemble_4x = nn.Sequential(
            nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False),
            ConvBNAct(d_ch[2], d_ch[2], kernel_size=3, padding=1),
        )
        self.reassemble_8x = nn.Sequential(
            nn.Upsample(scale_factor=8.0, mode="bilinear", align_corners=False),
            ConvBNAct(d_ch[3], d_ch[3], kernel_size=3, padding=1),
        )

        self.up_h8 = DecoderUpBlock(d_ch[0], d_ch[1], d_ch[1])
        self.up_h4 = DecoderUpBlock(d_ch[1], d_ch[2], d_ch[2])
        self.up_h2 = DecoderUpBlock(d_ch[2], d_ch[3], d_ch[3])
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            DoubleConv(d_ch[3], d_ch[3]),
            nn.Conv2d(d_ch[3], num_classes, kernel_size=1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[2:]
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if (h > 1 and w > 1) else "constant")
        return padded, (h, w)

    def _extract_spatial_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run ViT backbone and obtain 4 spatial feature maps ``[B, D, H_p, W_p]``."""
        feats = self.backbone.forward_intermediates(
            x, indices=self.feature_indices, intermediates_only=True
        )
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 intermediate spatial feature maps from ViT, got {len(feats)}."
            )
        return list(feats)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(
                f"Temporal input shapes must match, got {tuple(before.shape)} vs {tuple(after.shape)}."
            )

        b_pad, orig_hw = self._pad_to_patch_multiple(before)
        a_pad, _ = self._pad_to_patch_multiple(after)

        feats_b = self._extract_spatial_features(b_pad)
        feats_a = self._extract_spatial_features(a_pad)

        # index 0 = earliest block -> H/2; index 3 = deepest block -> H/16
        f_early = self.reassemble_8x(self.fusion_blocks[0](feats_b[0], feats_a[0]))
        f_mid1 = self.reassemble_4x(self.fusion_blocks[1](feats_b[1], feats_a[1]))
        f_mid2 = self.reassemble_2x(self.fusion_blocks[2](feats_b[2], feats_a[2]))
        f_deep = self.fusion_blocks[3](feats_b[3], feats_a[3])

        x = self.up_h8(f_deep, f_mid2)
        x = self.up_h4(x, f_mid1)
        x = self.up_h2(x, f_early)
        logits = self.final_up(x)

        if logits.shape[2:] != b_pad.shape[2:]:
            logits = F.interpolate(logits, size=b_pad.shape[2:], mode="bilinear", align_corners=False)

        return logits[:, :, : orig_hw[0], : orig_hw[1]]

    @contextmanager
    def _capture_block_attentions(self) -> Iterator[list[torch.Tensor]]:
        """Temporarily disable fused attention and capture post-softmax attention matrices."""
        captured: list[torch.Tensor] = []
        handles = []
        original_flags: list[tuple[nn.Module, bool]] = []

        def _hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured.append(inputs[0].detach().cpu())

        for block in self.backbone.blocks:
            attn_mod = getattr(block, "attn", None)
            if attn_mod is None or not hasattr(attn_mod, "attn_drop"):
                continue
            if hasattr(attn_mod, "fused_attn"):
                original_flags.append((attn_mod, bool(attn_mod.fused_attn)))
                attn_mod.fused_attn = False
            handles.append(attn_mod.attn_drop.register_forward_hook(_hook))

        try:
            yield captured
        finally:
            for h in handles:
                h.remove()
            for mod, prev in original_flags:
                mod.fused_attn = prev

    def forward_with_attention(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run inference while capturing genuine self-attention weights from all ViT blocks."""
        if before.shape != after.shape:
            raise ValueError(
                f"Temporal input shapes must match, got {tuple(before.shape)} vs {tuple(after.shape)}."
            )
        b_pad, orig_hw = self._pad_to_patch_multiple(before)
        a_pad, _ = self._pad_to_patch_multiple(after)
        grid_hw = (b_pad.shape[2] // self.patch_size, b_pad.shape[3] // self.patch_size)

        with self._capture_block_attentions() as cap_b:
            feats_b = self._extract_spatial_features(b_pad)
        with self._capture_block_attentions() as cap_a:
            feats_a = self._extract_spatial_features(a_pad)

        f_early = self.reassemble_8x(self.fusion_blocks[0](feats_b[0], feats_a[0]))
        f_mid1 = self.reassemble_4x(self.fusion_blocks[1](feats_b[1], feats_a[1]))
        f_mid2 = self.reassemble_2x(self.fusion_blocks[2](feats_b[2], feats_a[2]))
        f_deep = self.fusion_blocks[3](feats_b[3], feats_a[3])

        x = self.up_h8(f_deep, f_mid2)
        x = self.up_h4(x, f_mid1)
        x = self.up_h2(x, f_early)
        logits = self.final_up(x)
        if logits.shape[2:] != b_pad.shape[2:]:
            logits = F.interpolate(logits, size=b_pad.shape[2:], mode="bilinear", align_corners=False)
        logits = logits[:, :, : orig_hw[0], : orig_hw[1]]

        attn_payload = {
            "supported": len(cap_b) > 0 and len(cap_a) > 0,
            "before_attns": cap_b,
            "after_attns": cap_a,
            "num_prefix_tokens": self.num_prefix_tokens,
            "patch_grid": grid_hw,
            "patch_size": self.patch_size,
        }
        return logits, attn_payload

