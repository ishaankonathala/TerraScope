"""Hardware device detection, reproducibility seeding, and memory tracking utilities."""

from __future__ import annotations

import os
import platform
import random
import resource
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of the current compute runtime and available accelerators."""

    selected_device: str
    os_name: str
    os_release: str
    machine: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    mps_built: bool
    mps_available: bool


def detect_device(preferred: str = "auto") -> torch.device:
    """Detect and return the appropriate PyTorch compute device.

    Supports ``"auto"``, ``"cuda"``, ``"mps"``, and ``"cpu"``. If a specific
    accelerator is requested but unavailable, raises a clear ValueError rather
    than silently misreporting hardware support.
    """
    choice = (preferred or "auto").strip().lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested, but torch.cuda.is_available() is False on this system."
            )
        return torch.device("cuda")

    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "Apple MPS device requested, but torch.backends.mps.is_available() is False."
            )
        return torch.device("mps")

    if choice == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option '{preferred}'. Choose auto, cpu, mps, or cuda.")


def resolve_device_safe(preferred: str = "auto") -> torch.device:
    """Resolve requested device, falling back to auto detection if unavailable."""
    try:
        return detect_device(preferred)
    except ValueError:
        return detect_device("auto")


def get_device_info(preferred: str = "auto") -> DeviceInfo:
    """Collect factual runtime hardware and PyTorch device information."""
    device = resolve_device_safe(preferred)
    cuda_avail = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_avail else None
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return DeviceInfo(
        selected_device=str(device),
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        cuda_device_name=cuda_name,
        mps_built=mps_built,
        mps_available=mps_avail,
    )


def device_info_dict(preferred: str = "auto") -> dict[str, Any]:
    """Return device metadata as a JSON-serializable dictionary."""
    return asdict(get_device_info(preferred))


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reset_peak_memory_stats(device: torch.device | str | None = None) -> None:
    """Reset peak accelerator memory counters if supported by the active device."""
    dev = torch.device(device) if device is not None else detect_device("auto")
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


def get_peak_memory_mb(device: torch.device | str | None = None) -> float | None:
    """Return peak memory usage in MiB when measurable for the active device."""
    dev = torch.device(device) if device is not None else detect_device("auto")
    if dev.type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0), 3)
    if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        try:
            mem = torch.mps.driver_allocated_memory()
            return round(mem / (1024.0 * 1024.0), 3)
        except Exception:
            pass
    # Fallback to OS process max RSS (ru_maxrss is bytes on macOS, KiB on Linux)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024.0 * 1024.0), 3)
    return round(usage / 1024.0, 3)

