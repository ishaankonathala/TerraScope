"""Phase 3-5 tests for Siamese U-Net, ViT, Swin Transformer, and Model Factory."""

from __future__ import annotations

import pytest
import torch

from ml.models import (
    SiameseUNet,
    SwinChangeDetector,
    ViTChangeDetector,
    get_model,
    list_available_models,
)


def test_model_registry_listing() -> None:
    catalog = list_available_models()
    ids = {entry["model_id"] for entry in catalog}
    assert ids == {"siamese_unet", "vit", "swin"}


@pytest.mark.parametrize("model_name", ["siamese_unet", "vit", "swin"])
@pytest.mark.parametrize("hw", [(128, 128), (224, 224), (256, 256), (150, 170)])
def test_all_models_output_shapes_and_probabilities(
    model_name: str, hw: tuple[int, int]
) -> None:
    h, w = hw
    model = get_model(model_name, in_channels=3, num_classes=1, img_size=256, pretrained=False)
    model.eval()

    before = torch.randn(1, 3, h, w)
    after = torch.randn(1, 3, h, w)
    with torch.no_grad():
        logits = model(before, after)
        probs = model.predict_proba(before, after)

    assert logits.shape == (1, 1, h, w)
    assert probs.shape == (1, 1, h, w)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0
    assert model.parameter_count() > 0


def test_multispectral_and_multiclass_interface() -> None:
    for name in ("siamese_unet", "vit", "swin"):
        model = get_model(name, in_channels=4, num_classes=5, img_size=128, pretrained=False)
        model.eval()
        b = torch.randn(2, 4, 128, 128)
        a = torch.randn(2, 4, 128, 128)
        with torch.no_grad():
            logits = model(b, a)
            probs = model.predict_proba(b, a)
        assert logits.shape == (2, 5, 128, 128)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, 128, 128), atol=1e-5)


def test_siamese_unet_custom_cnn_and_gradcam_layer() -> None:
    model = SiameseUNet(in_channels=3, num_classes=1, encoder_name="custom_cnn", pretrained=False)
    assert model.get_gradcam_target_layer() is not None
    x = torch.randn(1, 3, 64, 64)
    out = model(x, x)
    assert out.shape == (1, 1, 64, 64)


def test_vit_attention_capture_and_restoration() -> None:
    model = ViTChangeDetector(in_channels=3, num_classes=1, img_size=256, pretrained=False)
    model.eval()
    orig_flags = [getattr(b.attn, "fused_attn", None) for b in model.backbone.blocks]

    b = torch.randn(1, 3, 256, 256)
    a = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, payload = model.forward_with_attention(b, a)

    assert logits.shape == (1, 1, 256, 256)
    assert payload["supported"] is True
    assert len(payload["before_attns"]) == len(model.backbone.blocks)
    assert len(payload["after_attns"]) == len(model.backbone.blocks)
    assert payload["patch_grid"] == (16, 16)
    # Verify fused_attn flags restored
    restored_flags = [getattr(blk.attn, "fused_attn", None) for blk in model.backbone.blocks]
    assert restored_flags == orig_flags


def test_swin_window_attention_capture() -> None:
    model = SwinChangeDetector(in_channels=3, num_classes=1, img_size=256, pretrained=False)
    model.eval()
    b = torch.randn(1, 3, 256, 256)
    a = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, payload = model.forward_with_attention(b, a)

    assert logits.shape == (1, 1, 256, 256)
    assert payload["supported"] is True
    assert len(payload["before_window_attns"]) > 0
    assert len(payload["after_window_attns"]) > 0
    assert len(payload["stage_shapes"]) == 4


def test_mismatched_temporal_input_raises() -> None:
    model = get_model("siamese_unet", pretrained=False)
    with pytest.raises(ValueError, match="must match"):
        model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 128, 128))

