"""Fair 3-model evaluation and benchmarking script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets import CustomPairDataset, LEVIRCDDataset, OSCDDataset
from ml.evaluation import benchmark_models, plot_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Siamese U-Net, ViT, and Swin on identical test splits.")
    parser.add_argument("--dataset", default="LEVIR-CD", choices=["LEVIR-CD", "OSCD", "Custom"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "checkpoints"))
    parser.add_argument("--experiments-dir", default=str(PROJECT_ROOT / "experiments"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"])
    args = parser.parse_args()

    ds_key = args.dataset.lower()
    if ds_key == "levir-cd":
        ds = LEVIRCDDataset(args.data_dir, split=args.split, patch_size=args.patch_size, stride=args.patch_size)
        in_ch = 3
    elif ds_key == "oscd":
        ds = OSCDDataset(
            args.data_dir,
            split=args.split,
            bands=tuple(args.bands),
            patch_size=args.patch_size,
            stride=args.patch_size,
        )
        in_ch = len(args.bands)
    else:
        ds = CustomPairDataset(root_dir=args.data_dir, image_size=(args.patch_size, args.patch_size))
        in_ch = 3

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ckpt_dir = Path(args.checkpoint_dir)
    safe_ds = args.dataset.lower().replace(" ", "_")

    model_configs = []
    for m_id in ("siamese_unet", "vit", "swin"):
        best_ckpt = ckpt_dir / f"{m_id}_{safe_ds}_best.pt"
        model_configs.append(
            {
                "model_name": m_id,
                "in_channels": in_ch,
                "img_size": args.patch_size,
                "pretrained": not args.no_pretrained,
                "checkpoint_path": str(best_ckpt) if best_ckpt.is_file() else None,
            }
        )

    payload = benchmark_models(
        model_configs=model_configs,
        data_loader=loader,
        dataset_name=args.dataset,
        split=args.split,
        device=args.device,
        threshold=args.threshold,
        experiments_dir=args.experiments_dir,
        is_synthetic=False,
    )
    chart_path = (
        PROJECT_ROOT / "outputs" / "visualizations" / f"{payload['benchmark_id']}_comparison.png"
    )
    plot_model_comparison(payload, chart_path)
    print(f"Benchmark saved: {payload['benchmark_id']} | Chart: {chart_path}")


if __name__ == "__main__":
    main()

