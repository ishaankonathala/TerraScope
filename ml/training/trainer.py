"""PyTorch training engine for Siamese CNN, ViT, and Swin change detectors."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.evaluation.metrics import MetricAccumulator
from ml.models import get_model
from ml.training.callbacks import (
    EarlyStopping,
    ExperimentLogger,
    ModelCheckpointManager,
    load_checkpoint,
)
from ml.training.losses import get_loss_function
from ml.training.optimizer import create_optimizer, create_scheduler
from ml.utils.device import (
    detect_device,
    device_info_dict,
    get_peak_memory_mb,
    reset_peak_memory_stats,
    set_seed,
)


@dataclass
class TrainingConfig:
    """Configuration for reproducible training on Apple Silicon MPS, CUDA, or CPU."""

    model_name: str = "siamese_unet"
    backbone_name: str | None = None
    dataset_name: str = "LEVIR-CD"
    in_channels: int = 3
    num_classes: int = 1
    img_size: int = 256
    pretrained: bool = True
    allow_random_fallback: bool = True

    epochs: int = 20
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    loss: str = "bce_dice"
    pos_weight: float | None = None
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    threshold: float = 0.5
    grad_clip_norm: float | None = 1.0
    early_stopping_patience: int = 8

    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    resume_from: str | None = None
    checkpoint_dir: str = "models/checkpoints"
    experiments_dir: str = "experiments"
    experiment_id: str | None = None
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChangeDetectionTrainer:
    """End-to-end trainer with validation, checkpointing, early stopping, and resume."""

    def __init__(self, config: TrainingConfig, model: nn.Module | None = None) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = detect_device(config.device)

        if model is None:
            self.model = get_model(
                config.model_name,
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                img_size=config.img_size,
                pretrained=config.pretrained,
                allow_random_fallback=config.allow_random_fallback,
                backbone_name=config.backbone_name,
            )
        else:
            self.model = model
        self.model.to(self.device)

        self.criterion = get_loss_function(
            config.loss,
            pos_weight=config.pos_weight,
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            num_classes=config.num_classes,
        )
        self.optimizer = create_optimizer(
            self.model,
            name=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            name=config.scheduler,
            epochs=config.epochs,
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.experiment_id = (
            config.experiment_id
            or f"{config.model_name}_{config.dataset_name.lower()}_{stamp}"
        )
        self.ckpt_manager = ModelCheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
        )
        self.exp_logger = ExperimentLogger(
            experiments_dir=config.experiments_dir,
            experiment_id=self.experiment_id,
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")

        self.start_epoch = 1
        self.best_f1 = -1.0

        if config.resume_from:
            ckpt = load_checkpoint(
                config.resume_from,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            self.start_epoch = int(ckpt.get("epoch", 0)) + 1
            self.best_f1 = float(ckpt.get("best_metric", -1.0))
            self.early_stopping.best_score = self.best_f1 if self.best_f1 >= 0 else None

        self.exp_logger.save_config(
            {
                **config.to_dict(),
                "experiment_id": self.experiment_id,
                "device_info": device_info_dict(config.device),
                "model_info": self.model.model_info() if hasattr(self.model, "model_info") else {},
            }
        )

    def _run_epoch(
        self, loader: DataLoader, is_train: bool
    ) -> tuple[float, dict[str, Any]]:
        self.model.train(is_train)
        total_loss = 0.0
        batches = 0
        acc = MetricAccumulator(threshold=self.config.threshold)

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch in loader:
                before = batch["before"].to(self.device)
                after = batch["after"].to(self.device)
                mask = batch["mask"].to(self.device)
                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(before, after)
                loss = self.criterion(logits, mask, valid_mask)

                if is_train:
                    loss.backward()
                    if self.config.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    self.optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                batches += 1
                acc.update(
                    preds=logits.detach(),
                    targets=mask.detach(),
                    valid_mask=valid_mask.detach() if valid_mask is not None else None,
                    from_logits=True,
                )

        avg_loss = total_loss / max(batches, 1)
        metrics = acc.compute().to_dict()
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Run training loop with validation, checkpointing, and early stopping."""
        reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        training_status = (
            "fine_tuned"
            if getattr(self.model, "backbone_weights_source", "random_init") == "timm_pretrained"
            else "fully_trained"
        )
        if hasattr(self.model, "training_status"):
            self.model.training_status = training_status

        best_val_metrics: dict[str, Any] = {}
        epochs_completed = 0

        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                train_loss, train_metrics = self._run_epoch(train_loader, is_train=True)
                val_loss, val_metrics = self._run_epoch(val_loader, is_train=False)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["f1_score"])
                    else:
                        self.scheduler.step()

                current_lr = float(self.optimizer.param_groups[0]["lr"])
                is_best = val_metrics["f1_score"] > self.best_f1
                if is_best:
                    self.best_f1 = float(val_metrics["f1_score"])
                    best_val_metrics = val_metrics

                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=self.best_f1,
                    config=self.config.to_dict(),
                    is_best=is_best,
                    training_status=training_status,
                )

                self.exp_logger.log_epoch(
                    epoch,
                    {
                        "lr": current_lr,
                        "train_loss": round(train_loss, 6),
                        "train_f1": train_metrics["f1_score"],
                        "train_iou": train_metrics["iou"],
                        "val_loss": round(val_loss, 6),
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1_score"],
                        "val_iou": val_metrics["iou"],
                        "val_dice": val_metrics["dice"],
                    },
                )
                epochs_completed = epoch

                self.early_stopping.step(val_metrics["f1_score"])
                if self.early_stopping.should_stop:
                    break
        finally:
            self.exp_logger.close()

        duration_sec = round(time.perf_counter() - t_start, 3)
        best_ckpt = self.ckpt_manager.best_path
        ckpt_size_mb = (
            round(best_ckpt.stat().st_size / (1024.0 * 1024.0), 3)
            if best_ckpt.is_file()
            else None
        )

        summary = {
            "experiment_id": self.experiment_id,
            "model_name": self.config.model_name,
            "backbone_name": getattr(self.model, "backbone_name", self.config.backbone_name),
            "dataset_name": self.config.dataset_name,
            "is_synthetic": self.config.is_synthetic,
            "training_status": training_status,
            "device": str(self.device),
            "epochs_completed": epochs_completed,
            "training_duration_sec": duration_sec,
            "best_val_f1": round(self.best_f1, 6) if self.best_f1 >= 0 else None,
            "best_val_metrics": best_val_metrics,
            "total_parameters": (
                self.model.parameter_count() if hasattr(self.model, "parameter_count") else None
            ),
            "checkpoint_size_mb": ckpt_size_mb,
            "best_checkpoint_path": str(best_ckpt) if best_ckpt.is_file() else None,
            "latest_checkpoint_path": str(self.ckpt_manager.latest_path),
            "peak_memory_mb": get_peak_memory_mb(self.device),
            "history": self.exp_logger.history,
        }
        self.exp_logger.save_summary(summary)
        return summary

