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
from ml.evaluation import (
    benchmark_models,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_samples,
)
from ml.models import get_model
from ml.training.callbacks import load_checkpoint
from ml.utils.device import detect_device


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
    parser.add_argument("--save-visuals", action="store_true", help="Save confusion matrices and sample predictions.")
    parser.add_argument("--visual-samples", type=int, default=4, help="Number of qualitative prediction tiles to export.")
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
    viz_dir = PROJECT_ROOT / "outputs" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    chart_path = viz_dir / f"{payload['benchmark_id']}_comparison.png"
    plot_model_comparison(payload, chart_path)
    print(f"Benchmark saved: {payload['benchmark_id']} | Chart: {chart_path}")

    if args.save_visuals:
        # 1. Save confusion matrices for each evaluated model
        for rec in payload.get("models", []):
            cm = rec.get("confusion_matrix")
            if cm:
                cm_path = viz_dir / f"{payload['benchmark_id']}_{rec['model_name']}_cm.png"
                plot_confusion_matrix(cm, cm_path, title=f"Confusion Matrix — {rec['display_name']}")
                print(f"  Confusion matrix saved: {cm_path.name}")

        # 2. Export sample qualitative prediction masks
        dev = detect_device(args.device)
        try:
            first_batch = next(iter(loader))
            b_t1 = first_batch["before"].to(dev)
            b_t2 = first_batch["after"].to(dev)
            b_gt = first_batch["mask"].cpu()
            n_samples = min(int(b_t1.shape[0]), max(int(args.visual_samples), 1))

            for cfg in model_configs:
                m_name = cfg["model_name"]
                model = get_model(
                    m_name,
                    in_channels=in_ch,
                    img_size=args.patch_size,
                    pretrained=cfg["pretrained"],
                    allow_random_fallback=True,
                ).to(dev)
                if cfg.get("checkpoint_path"):
                    load_checkpoint(cfg["checkpoint_path"], model, map_location=dev)
                model.eval()
                import torch
                with torch.no_grad():
                    logits = model(b_t1, b_t2)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                pred_dir = viz_dir / "predictions" / f"{payload['benchmark_id']}_{m_name}"
                for s_idx in range(n_samples):
                    pred_path = pred_dir / f"sample_{s_idx:02d}.png"
                    plot_prediction_samples(
                        before=b_t1[s_idx],
                        after=b_t2[s_idx],
                        ground_truth=b_gt[s_idx],
                        prediction=probs[s_idx] > args.threshold,
                        output_path=pred_path,
                        title=f"{rec['display_name']} Prediction Sample #{s_idx + 1}",
                    )
                print(f"  Exported {n_samples} visual prediction tiles to: {pred_dir}")
        except Exception as exc:
            print(f"Warning: could not export visual prediction samples: {exc}")



if __name__ == "__main__":
    main()

