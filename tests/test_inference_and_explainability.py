"""Phase 8-9 tests for tiled inference, postprocessing, and genuine XAI heatmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ml.explainability import INTERPRETABILITY_DISCLAIMER, explain_prediction
from ml.inference import ChangePredictor, PostprocessingConfig
from ml.models import get_model


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_tiled_inference_and_postprocessing(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    before = rng.integers(40, 120, size=(320, 320, 3), dtype=np.uint8)
    after = before.copy()
    after[80:140, 100:180] = 230

    b_path = tmp_path / "before.png"
    a_path = tmp_path / "after.png"
    _save_png(b_path, before)
    _save_png(a_path, after)

    model = get_model(
        "siamese_unet", backbone_name="custom_cnn", pretrained=False, img_size=128
    )
    predictor = ChangePredictor(model=model, device="cpu")
    res = predictor.predict_files(
        b_path,
        a_path,
        threshold=0.5,
        tile_size=128,
        stride=96,
        postprocess_config=PostprocessingConfig(
            enabled=True, morph_operation="open", morph_kernel_size=3, min_region_pixels=10
        ),
    )

    assert res.probability_map.shape == (320, 320)
    assert res.binary_mask.shape == (320, 320)
    assert res.overlay_rgb.shape == (320, 320, 3)
    assert res.total_valid_pixels == 320 * 320
    assert res.masked_invalid_pixels == 0
    assert 0.0 <= res.change_percentage <= 100.0
    assert res.metadata["postprocessing"]["enabled"] is True


def test_inference_excludes_cloud_masked_pixels(tmp_path: Path) -> None:
    arr = np.full((4, 64, 64), 1500.0, dtype=np.float32)
    scl = np.full((64, 64), 4.0, dtype=np.float32)
    scl[:16, :16] = 9.0  # Cloud high probability (256 pixels)
    arr[3] = scl

    b_tif = tmp_path / "b.tif"
    a_tif = tmp_path / "a.tif"
    tf = from_origin(500000.0, 2000000.0, 10.0, 10.0)
    for p in (b_tif, a_tif):
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=4,
            dtype="float32",
            crs=CRS.from_user_input("EPSG:32643"),
            transform=tf,
        ) as dst:
            dst.write(arr)
            for idx, name in enumerate(["B04", "B03", "B02", "SCL"], start=1):
                dst.set_band_description(idx, name)

    model = get_model("siamese_unet", backbone_name="custom_cnn", pretrained=False, img_size=64)
    predictor = ChangePredictor(model=model, device="cpu")
    res = predictor.predict_files(b_tif, a_tif, tile_size=64, stride=64, normalize_mode="sentinel2_sr")

    assert res.masked_invalid_pixels == 256
    assert res.total_valid_pixels == (64 * 64) - 256
    assert not np.any(res.binary_mask[:16, :16])
    assert res.metadata["cloud_mask_before"]["cloud_masking_available"] is True


def test_explainability_all_three_architectures(tmp_path: Path) -> None:
    b_t = torch.randn(1, 3, 128, 128)
    a_t = torch.randn(1, 3, 128, 128)
    b_rgb = np.full((128, 128, 3), 90, dtype=np.uint8)
    a_rgb = np.full((128, 128, 3), 130, dtype=np.uint8)

    configs = [
        ("siamese_unet", "custom_cnn", "gradcam.png"),
        ("vit", None, "vit_rollout.png"),
        ("swin", None, "swin_window.png"),
    ]
    for model_name, bb, fname in configs:
        model = get_model(
            model_name, backbone_name=bb, pretrained=False, img_size=128
        )
        out = explain_prediction(
            model=model,
            before_tensor=b_t,
            after_tensor=a_t,
            before_rgb=b_rgb,
            after_rgb=a_rgb,
            output_path=tmp_path / fname,
        )
        assert out["supported"] is True
        assert out["heatmap"].shape == (128, 128)
        assert float(out["heatmap"].min()) >= 0.0
        assert float(out["heatmap"].max()) <= 1.0
        assert out["disclaimer"] == INTERPRETABILITY_DISCLAIMER
        assert (tmp_path / fname).is_file()

