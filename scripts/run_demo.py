"""End-to-end local verification demo script (does not pollute real benchmark logs)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability import explain_prediction
from ml.inference import ChangePredictor
from ml.models import get_model
from ml.preprocessing import load_image, normalize_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end satellite change-detection demo.")
    parser.add_argument("--before", default=None, help="Path to before image (optional).")
    parser.add_argument("--after", default=None, help="Path to after image (optional).")
    parser.add_argument("--model", default="siamese_unet", choices=["siamese_unet", "vit", "swin"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.before and args.after:
            b_path = Path(args.before)
            a_path = Path(args.after)
        else:
            print("No external images provided; running isolated synthetic pipeline smoke demo in /tmp.")
            rng = np.random.default_rng(42)
            b_arr = rng.integers(50, 120, size=(256, 256, 3), dtype=np.uint8)
            a_arr = b_arr.copy()
            a_arr[70:150, 80:170] = 220
            b_path = Path(tmp) / "demo_before.png"
            a_path = Path(tmp) / "demo_after.png"
            Image.fromarray(b_arr).save(b_path)
            Image.fromarray(a_arr).save(a_path)

        model = get_model(args.model, pretrained=False, img_size=256)
        predictor = ChangePredictor(
            model=model, checkpoint_path=args.checkpoint, device=args.device
        )
        res = predictor.predict_files(b_path, a_path, tile_size=256, stride=192)
        print(
            f"Model={args.model} | Status={model.training_status} | "
            f"Change%={res.change_percentage:.2f}% | Time={res.inference_time_sec}s"
        )

        # Quick XAI pass
        b_img = load_image(b_path, expected_channels=3)
        a_img = load_image(a_path, expected_channels=3)
        b_n, a_n = normalize_pair(b_img.data[:, :256, :256], a_img.data[:, :256, :256])
        bt = torch.from_numpy(b_n).unsqueeze(0).to(predictor.device)
        at = torch.from_numpy(a_n).unsqueeze(0).to(predictor.device)
        xai = explain_prediction(
            model,
            bt,
            at,
            before_rgb=res.before_rgb[:256, :256],
            after_rgb=res.after_rgb[:256, :256],
        )
        print(f"Explainability supported={xai.get('supported')} | Method={xai.get('method')}")


if __name__ == "__main__":
    main()

