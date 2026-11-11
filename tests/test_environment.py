"""Phase 1 smoke tests for runtime environment, device detection, and seeding."""

from __future__ import annotations

import numpy as np
import torch

from ml.utils.device import detect_device, device_info_dict, get_peak_memory_mb, set_seed


def test_device_detection_and_metadata() -> None:
    device = detect_device("auto")
    assert device.type in {"cpu", "mps", "cuda"}

    info = device_info_dict("auto")
    assert info["selected_device"] == str(device)
    assert info["python_version"].startswith("3.11")
    # Never claim CUDA on Apple Silicon
    if info["machine"] == "arm64" and info["os_name"] == "Darwin":
        assert info["cuda_available"] is False


def test_seed_reproducibility() -> None:
    set_seed(1234)
    a_np = np.random.rand(4, 4)
    a_t = torch.rand(4, 4)

    set_seed(1234)
    b_np = np.random.rand(4, 4)
    b_t = torch.rand(4, 4)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_t, b_t)


def test_memory_reporting_returns_positive_value() -> None:
    mem_mb = get_peak_memory_mb("cpu")
    assert mem_mb is not None
    assert mem_mb > 0.0

