"""Phase 6-7 tests for losses, training engine, resume checkpoints, metrics, and benchmarking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ml.evaluation import (
    benchmark_models,
    build_comparison_table,
    compute_binary_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_curves,
)
from ml.models import SiameseUNet
from ml.training import ChangeDetectionTrainer, TrainingConfig, get_loss_function


class _SyntheticPairTorchDataset(Dataset):
    """In-memory synthetic dataset strictly for unit testing."""

    def __init__(self, num_samples: int = 4, size: int = 64) -> None:
        self.samples = []
        rng = np.random.default_rng(7)
        for _ in range(num_samples):
            b = rng.normal(0.0, 0.2, size=(3, size, size)).astype(np.float32)
            a = b.copy()
            mask = np.zeros((size, size), dtype=np.float32)
            a[:, 16:40, 16:40] += 1.5
            mask[16:40, 16:40] = 1.0
            valid = np.ones((size, size), dtype=bool)
            self.samples.append(
                {
                    "before": torch.from_numpy(b),
                    "after": torch.from_numpy(a),
                    "mask": torch.from_numpy(mask),
                    "valid_mask": torch.from_numpy(valid),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def test_losses_and_valid_mask_exclusion() -> None:
    logits = torch.zeros(2, 1, 16, 16)
    targets = torch.zeros(2, 16, 16)
    targets[:, :8, :8] = 1.0
    valid = torch.ones(2, 16, 16, dtype=torch.bool)
    valid[:, :4, :4] = False

    for name in ("bce", "dice", "bce_dice"):
        fn = get_loss_function(name, pos_weight=2.0)
        loss = fn(logits, targets, valid)
        assert torch.isfinite(loss)
        assert float(loss) > 0.0

    mc_fn = get_loss_function("cross_entropy", num_classes=4)
    mc_logits = torch.randn(2, 4, 16, 16)
    mc_loss = mc_fn(mc_logits, targets.long(), valid)
    assert torch.isfinite(mc_loss)


def test_metrics_calculation_and_division_by_zero_safety() -> None:
    # Known 4x4 grid: 4 TP, 2 FP, 2 FN, 8 TN -> Acc=12/16=0.75, Prec=4/6, Rec=4/6, IoU=4/8=0.5
    pred = np.zeros((4, 4), dtype=np.float32)
    true = np.zeros((4, 4), dtype=np.float32)
    true[:2, :3] = 1.0  # 6 positives
    pred[:2, :2] = 0.9  # 4 TP
    pred[2, :2] = 0.9   # 2 FP -> total 6 pred positives, 2 FN

    m = compute_binary_metrics(pred, true, threshold=0.5)
    assert m.tp == 4 and m.fp == 2 and m.fn == 2 and m.tn == 8
    assert pytest.approx(m.accuracy, rel=1e-4) == 0.75
    assert pytest.approx(m.precision, rel=1e-4) == 4.0 / 6.0
    assert pytest.approx(m.recall, rel=1e-4) == 4.0 / 6.0
    assert pytest.approx(m.f1_score, rel=1e-4) == 4.0 / 6.0
    assert pytest.approx(m.iou, rel=1e-4) == 0.5

    # Valid mask exclusion: mask out the 2 FPs and verify precision jumps to 1.0
    vm = np.ones((4, 4), dtype=bool)
    vm[2, :2] = False
    m_masked = compute_binary_metrics(pred, true, valid_mask=vm, threshold=0.5)
    assert m_masked.fp == 0
    assert pytest.approx(m_masked.precision, rel=1e-4) == 1.0

    # All-zero division-by-zero edge case
    zero_m = compute_binary_metrics(np.zeros((4, 4)), np.zeros((4, 4)))
    assert zero_m.precision == 0.0
    assert zero_m.recall == 0.0
    assert zero_m.f1_score == 0.0
    assert zero_m.iou == 0.0
    assert zero_m.accuracy == 1.0


def test_training_resume_and_logging(tmp_path: Path) -> None:
    ds = _SyntheticPairTorchDataset(num_samples=4, size=64)
    train_loader = DataLoader(ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)

    ckpt_dir = tmp_path / "checkpoints"
    exp_dir = tmp_path / "experiments"

    cfg = TrainingConfig(
        model_name="siamese_unet",
        backbone_name="custom_cnn",
        dataset_name="synthetic_unit",
        img_size=64,
        pretrained=False,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        experiments_dir=str(exp_dir),
        experiment_id="unit_run_1",
        is_synthetic=True,
    )
    model = SiameseUNet(in_channels=3, encoder_name="custom_cnn", pretrained=False)
    trainer = ChangeDetectionTrainer(cfg, model=model)
    summary = trainer.fit(train_loader, val_loader)

    assert summary["epochs_completed"] == 2
    assert (ckpt_dir / "siamese_unet_synthetic_unit_best.pt").is_file()
    assert (ckpt_dir / "siamese_unet_synthetic_unit_latest.pt").is_file()
    assert (exp_dir / "results" / "unit_run_1_history.csv").is_file()
    assert (exp_dir / "results" / "unit_run_1_summary.json").is_file()

    # Resume training for 1 more epoch
    cfg_resume = TrainingConfig(
        model_name="siamese_unet",
        backbone_name="custom_cnn",
        dataset_name="synthetic_unit",
        img_size=64,
        pretrained=False,
        epochs=3,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        resume_from=str(ckpt_dir / "siamese_unet_synthetic_unit_latest.pt"),
        checkpoint_dir=str(ckpt_dir),
        experiments_dir=str(exp_dir),
        experiment_id="unit_run_resume",
        is_synthetic=True,
    )
    model2 = SiameseUNet(in_channels=3, encoder_name="custom_cnn", pretrained=False)
    trainer2 = ChangeDetectionTrainer(cfg_resume, model=model2)
    assert trainer2.start_epoch == 3
    summary2 = trainer2.fit(train_loader, val_loader)
    assert summary2["epochs_completed"] == 3

    curve_path = plot_training_curves(summary["history"], tmp_path / "curves.png")
    assert curve_path is not None and curve_path.is_file()


def test_benchmark_and_empty_comparison_table(tmp_path: Path) -> None:
    empty_table = build_comparison_table(experiments_dir=tmp_path / "no_exp")
    assert len(empty_table) == 3
    assert all(row["f1_score"] == "Not evaluated" for row in empty_table)

    ds = _SyntheticPairTorchDataset(num_samples=2, size=64)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    bench = benchmark_models(
        model_configs=[
            {
                "model_name": "siamese_unet",
                "backbone_name": "custom_cnn",
                "pretrained": False,
                "img_size": 64,
            },
            {"model_name": "vit", "pretrained": False, "img_size": 64},
            {"model_name": "swin", "pretrained": False, "img_size": 64},
        ],
        data_loader=loader,
        dataset_name="synthetic_unit",
        split="test",
        device="cpu",
        experiments_dir=tmp_path / "exp",
        is_synthetic=True,
    )
    assert len(bench["models"]) == 3
    assert all(m["training_status"] == "untrained" for m in bench["models"])

    cm_img = plot_confusion_matrix(bench["models"][0]["confusion_matrix"], tmp_path / "cm.png")
    cmp_img = plot_model_comparison(bench, tmp_path / "cmp.png")
    assert cm_img.is_file()
    assert cmp_img is not None and cmp_img.is_file()

