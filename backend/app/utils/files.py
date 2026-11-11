"""Safe file handling, ID validation, and upload size enforcement."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ml.preprocessing.image_loader import SUPPORTED_IMAGE_EXTENSIONS

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,80}$")


def validate_safe_id(value: str, label: str = "identifier") -> str:
    """Reject empty or path-traversal identifiers strictly."""
    if not _SAFE_ID_RE.match(value or ""):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{value}'.")
    return value


def safe_resolve_child(base_dir: Path, *parts: str) -> Path:
    """Resolve a path inside ``base_dir`` and block path traversal."""
    base_resolved = base_dir.resolve()
    target = base_resolved.joinpath(*parts).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise HTTPException(status_code=400, detail="Access outside managed storage is forbidden.")
    return target


async def save_upload_validated(
    upload: UploadFile,
    dest_dir: Path,
    prefix: str,
    max_size_mb: float,
) -> Path:
    """Validate upload extension & byte size while streaming safely to disk."""
    filename = Path(upload.filename or f"{prefix}.png").name
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}' for '{filename}'. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            ),
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = safe_resolve_child(dest_dir, f"{prefix}{ext}")
    max_bytes = int(max_size_mb * 1024 * 1024)
    total = 0

    with dest_path.open("wb") as out_fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                out_fh.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file '{filename}' exceeds {max_size_mb:.1f} MB limit.",
                )
            out_fh.write(chunk)

    if total == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")

    return dest_path

