"""Siamese U-Net CNN baseline for multi-temporal change detection."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.blocks import (
    BaseChangeDetector,
    DecoderUpBlock,
    DoubleConv,
    TemporalFusionBlock,
    create_timm_backbone,
)


class _LightweightCNNEncoder(nn.Module):
    """Self-contained 4-stage CNN encoder fallback requiring no external downloads."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stage1 = DoubleConv(in_channels, base_channels)  # H, W
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))  # H/2
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4)
        )  # H/4
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8)
        )  # H/8
        self.feature_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class SiameseUNet(BaseChangeDetector):
    """Siamese U-Net architecture with shared CNN encoder and multi-scale temporal fusion.

    Supports both pretrained ``timm`` CNN backbones (e.g. ``resnet18``, ``resnet34``)
    and a lightweight ``custom_cnn`` encoder for resource-constrained environments.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        allow_random_fallback: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ) -> None:
        if encoder_name == "custom_cnn":
            encoder: nn.Module = _LightweightCNNEncoder(in_channels=in_channels, base_channels=32)
            enc_channels: list[int] = encoder.feature_channels
            weight_src = "random_init"
        else:
            encoder, weight_src = create_timm_backbone(
                encoder_name,
                pretrained=pretrained,
                allow_random_fallback=allow_random_fallback,
                features_only=True,
                in_chans=in_channels,
            )
            enc_channels = list(encoder.feature_info.channels())

        super().__init__(
            model_name="siamese_unet",
            architecture_family="CNN (Siamese U-Net)",
            backbone_name=encoder_name,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            backbone_weights_source=weight_src,
        )

        self.encoder = encoder
        self.encoder_channels = enc_channels

        # Align decoder width dynamically to the number of encoder feature stages
        num_stages = len(enc_channels)
        dec_list = list(decoder_channels)
        if len(dec_list) < num_stages:
            dec_list = dec_list + [dec_list[-1]] * (num_stages - len(dec_list))
        elif len(dec_list) > num_stages:
            dec_list = dec_list[-num_stages:]
        self.decoder_channels = dec_list  # coarse-to-fine order matching reversed stages

        # Build temporal fusion at each encoder scale (fine -> coarse)
        # Map encoder stage i -> fusion output matching reversed decoder stage channels
        rev_dec = list(reversed(self.decoder_channels))
        self.fusion_blocks = nn.ModuleList(
            [TemporalFusionBlock(enc_ch, rev_dec[i]) for i, enc_ch in enumerate(enc_channels)]
        )

        self.up_blocks = nn.ModuleList()
        in_ch = self.decoder_channels[0]
        for idx in range(1, num_stages):
            skip_ch = rev_dec[num_stages - 1 - idx]
            out_ch = self.decoder_channels[idx]
            self.up_blocks.append(DecoderUpBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.refine = DoubleConv(in_ch, in_ch)
        self.seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the final spatial convolutional block suitable for Grad-CAM."""
        return self.refine

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(
                f"Temporal input shapes must match, got {tuple(before.shape)} vs {tuple(after.shape)}."
            )
        target_hw = before.shape[2:]

        feats_b = self.encoder(before)
        feats_a = self.encoder(after)

        fused_stages = [
            fusion(fb, fa) for fusion, fb, fa in zip(self.fusion_blocks, feats_b, feats_a)
        ]

        x = fused_stages[-1]
        for idx, up in enumerate(self.up_blocks):
            skip = fused_stages[-2 - idx]
            x = up(x, skip)

        if x.shape[2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        x = self.refine(x)
        return self.seg_head(x)

