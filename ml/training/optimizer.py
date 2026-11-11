"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create Adam or AdamW optimizer over trainable model parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters.")

    key = (name or "adamw").strip().lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'. Choose 'adam' or 'adamw'.")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    epochs: int = 20,
    min_lr: float = 1e-6,
    patience: int = 3,
    step_size: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create learning-rate scheduler."""
    key = (name or "cosine").strip().lower()
    if key == "none":
        return None
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=min_lr
        )
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=patience, min_lr=min_lr
        )
    if key == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(step_size, 1), gamma=0.5)
    raise ValueError(f"Unsupported scheduler '{name}'. Choose cosine, plateau, step, or none.")

