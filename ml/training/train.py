"""High-level training runner and CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ml.datasets import CustomPairDataset, LEVIRCDDataset, OSCDDataset
from ml.training.trainer import ChangeDetectionTrainer, TrainingConfig


def build_dataloaders(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int = 4,
    patch_size: int = 256,
    num_workers: int = 0,
    oscd_bands: tuple[str, ...] = ("B04", "B03", "B02"),
) -> tuple[DataLoader, DataLoader]:
    """Construct paired train and validation DataLoaders for supported datasets."""
    ds_key = dataset_name.strip().lower()
    if ds_key in {"levir", "levir-cd", "levir_cd"}:
        train_ds = LEVIRCDDataset(
            root_dir=data_dir, split="train", patch_size=patch_size, stride=patch_size
        )
        val_ds = LEVIRCDDataset(
            root_dir=data_dir, split="val", patch_size=patch_size, stride=patch_size
        )
    elif ds_key == "oscd":
        train_ds = OSCDDataset(
            root_dir=data_dir, split="train", bands=oscd_bands, patch_size=patch_size, stride=patch_size
        )
        val_ds = OSCDDataset(
            root_dir=data_dir, split="val", bands=oscd_bands, patch_size=patch_size, stride=patch_size
        )
    elif ds_key == "custom":
        base = Path(data_dir)
        train_dir = base / "train" if (base / "train").is_dir() else base
        val_dir = base / "val" if (base / "val").is_dir() else base
        train_ds = CustomPairDataset(
            root_dir=train_dir, image_size=(patch_size, patch_size), is_train=True
        )
        val_ds = CustomPairDataset(
            root_dir=val_dir, image_size=(patch_size, patch_size), is_train=False
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose LEVIR-CD, OSCD, or Custom.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader


def run_training(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, Any]:
    """Instantiate ChangeDetectionTrainer and run training end-to-end."""
    trainer = ChangeDetectionTrainer(config)
    return trainer.fit(train_loader=train_loader, val_loader=val_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a satellite change-detection model.")
    parser.add_argument("--model", default="siamese_unet", choices=["siamese_unet", "vit", "swin"])
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--dataset", default="LEVIR-CD", choices=["LEVIR-CD", "OSCD", "Custom"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice"])
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "step", "none"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"])
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_channels = len(args.bands) if args.dataset.lower() == "oscd" else 3
    train_loader, val_loader = build_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        oscd_bands=tuple(args.bands),
    )
    cfg = TrainingConfig(
        model_name=args.model,
        backbone_name=args.backbone,
        dataset_name=args.dataset,
        in_channels=in_channels,
        img_size=args.patch_size,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        experiments_dir=args.experiments_dir,
        backup_dir=args.backup_dir,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
    )
    summary = run_training(cfg, train_loader, val_loader)
    print(f"Training completed: {summary['experiment_id']} | Best Val F1: {summary['best_val_f1']}")


if __name__ == "__main__":
    main()

