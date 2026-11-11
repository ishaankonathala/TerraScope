"""Siamese Swin Transformer hierarchical change detector."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

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


class SwinChangeDetector(BaseChangeDetector):
    """Siamese Swin Transformer change detector with multi-scale hierarchical fusion.

    Uses a shared ``timm`` Swin Transformer encoder producing 4 hierarchical feature
    stages (at H/4, H/8, H/16, and H/32 resolutions), fuses bi-temporal differences
    at every stage, and reconstructs full-resolution binary/multi-class logits via a
    U-Net/FPN decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        allow_random_fallback: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ) -> None:
        backbone, weight_src = create_timm_backbone(
            backbone_name,
            pretrained=pretrained,
            allow_random_fallback=allow_random_fallback,
            img_size=img_size,
            dynamic_img_size=True,
            in_chans=in_channels,
            num_classes=0,
        )

        super().__init__(
            model_name="swin",
            architecture_family="Swin Transformer (Hierarchical Siamese)",
            backbone_name=backbone_name,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            backbone_weights_source=weight_src,
        )

        self.backbone = backbone
        feat_info = backbone.feature_info
        if hasattr(feat_info, "channels"):
            stage_channels = [int(c) for c in feat_info.channels()]
        elif isinstance(feat_info, (list, tuple)):
            stage_channels = [int(entry["num_chs"]) for entry in feat_info]
        else:
            raise TypeError(f"Unsupported Swin feature_info type: {type(feat_info)}")
        if len(stage_channels) != 4:
            raise ValueError(
                f"Expected 4 hierarchical feature stages from Swin backbone, got {len(stage_channels)}."
            )
        self.stage_channels = stage_channels  # e.g. [96, 192, 384, 768] for swin_tiny

        dec_ch = list(decoder_channels)  # coarse-to-fine: [256, 128, 64, 32]
        if len(dec_ch) != 4:
            raise ValueError("SwinChangeDetector decoder_channels must contain 4 channel widths.")
        self.decoder_channels = dec_ch

        # Map each hierarchical stage [H/4, H/8, H/16, H/32] -> [32, 64, 128, 256]
        rev_dec = list(reversed(dec_ch))
        self.fusion_blocks = nn.ModuleList(
            [TemporalFusionBlock(sc, rev_dec[i]) for i, sc in enumerate(stage_channels)]
        )

        self.up_h16 = DecoderUpBlock(dec_ch[0], dec_ch[1], dec_ch[1])
        self.up_h8 = DecoderUpBlock(dec_ch[1], dec_ch[2], dec_ch[2])
        self.up_h4 = DecoderUpBlock(dec_ch[2], dec_ch[3], dec_ch[3])

        # Stage 0 is at H/4; upsample cleanly H/4 -> H/2 -> H
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            DoubleConv(dec_ch[3], dec_ch[3]),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            DoubleConv(dec_ch[3], dec_ch[3]),
            nn.Conv2d(dec_ch[3], num_classes, kernel_size=1),
        )

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[2:]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if (h > 1 and w > 1) else "constant")
        return padded, (h, w)

    def _extract_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = self.backbone.forward_intermediates(x, intermediates_only=True)
        out: list[torch.Tensor] = []
        for idx, feat in enumerate(stages):
            expected_c = self.stage_channels[idx]
            # Guard in case a timm variant outputs NHWC instead of NCHW
            if feat.ndim == 4 and feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out

    def _decode_stages(
        self, feats_b: list[torch.Tensor], feats_a: list[torch.Tensor], target_hw: tuple[int, int]
    ) -> torch.Tensor:
        fused = [fb_fn(fb, fa) for fb_fn, fb, fa in zip(self.fusion_blocks, feats_b, feats_a)]
        x = self.up_h16(fused[3], fused[2])
        x = self.up_h8(x, fused[1])
        x = self.up_h4(x, fused[0])
        logits = self.final_head(x)
        if logits.shape[2:] != target_hw:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return logits

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(
                f"Temporal input shapes must match, got {tuple(before.shape)} vs {tuple(after.shape)}."
            )
        b_pad, orig_hw = self._pad_to_multiple(before, multiple=32)
        a_pad, _ = self._pad_to_multiple(after, multiple=32)

        feats_b = self._extract_stages(b_pad)
        feats_a = self._extract_stages(a_pad)
        logits = self._decode_stages(feats_b, feats_a, b_pad.shape[2:])
        return logits[:, :, : orig_hw[0], : orig_hw[1]]

    @contextmanager
    def _capture_window_attentions(self) -> Iterator[list[dict[str, Any]]]:
        """Temporarily disable fused attention and record WindowAttention softmax weights."""
        captured: list[dict[str, Any]] = []
        handles = []
        original_flags: list[tuple[nn.Module, bool]] = []

        for layer_idx, layer in enumerate(getattr(self.backbone, "layers", [])):
            for block_idx, block in enumerate(getattr(layer, "blocks", [])):
                attn_mod = getattr(block, "attn", None)
                if attn_mod is None or not hasattr(attn_mod, "attn_drop"):
                    continue
                if hasattr(attn_mod, "fused_attn"):
                    original_flags.append((attn_mod, bool(attn_mod.fused_attn)))
                    attn_mod.fused_attn = False

                win_size = getattr(attn_mod, "window_size", (7, 7))
                input_res = getattr(block, "input_resolution", None)

                def _make_hook(l_i: int, b_i: int, ws: Any, res: Any):
                    def _hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            captured.append(
                                {
                                    "layer": l_i,
                                    "block": b_i,
                                    "window_size": tuple(ws) if isinstance(ws, (tuple, list)) else (ws, ws),
                                    "input_resolution": tuple(res) if res is not None else None,
                                    "attn": inputs[0].detach().cpu(),
                                }
                            )

                    return _hook

                handles.append(
                    attn_mod.attn_drop.register_forward_hook(
                        _make_hook(layer_idx, block_idx, win_size, input_res)
                    )
                )

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
        """Run inference while capturing genuine Swin WindowAttention maps per stage."""
        if before.shape != after.shape:
            raise ValueError(
                f"Temporal input shapes must match, got {tuple(before.shape)} vs {tuple(after.shape)}."
            )
        b_pad, orig_hw = self._pad_to_multiple(before, multiple=32)
        a_pad, _ = self._pad_to_multiple(after, multiple=32)

        with self._capture_window_attentions() as cap_b:
            feats_b = self._extract_stages(b_pad)
        with self._capture_window_attentions() as cap_a:
            feats_a = self._extract_stages(a_pad)

        logits = self._decode_stages(feats_b, feats_a, b_pad.shape[2:])
        logits = logits[:, :, : orig_hw[0], : orig_hw[1]]

        attn_payload = {
            "supported": len(cap_b) > 0 and len(cap_a) > 0,
            "before_window_attns": cap_b,
            "after_window_attns": cap_a,
            "stage_shapes": [tuple(f.shape[2:]) for f in feats_b],
            "padded_hw": tuple(b_pad.shape[2:]),
        }
        return logits, attn_payload
