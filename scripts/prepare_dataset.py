"""Prepare and tile custom or public datasets into uniform patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets import LEVIRCDDataset, OSCDDataset, prepare_custom_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare dataset patches.")
    parser.add_argument("--dataset", required=True, choices=["custom", "levir", "oscd"])
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"])
    args = parser.parse_args()

    if args.dataset == "custom":
        summary = prepare_custom_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "patches"}, indent=2))
    elif args.dataset == "levir":
        counts = {}
        for sp in ("train", "val", "test"):
            ds = LEVIRCDDataset(args.source_dir, split=sp, patch_size=args.patch_size, stride=args.stride)
            counts[sp] = len(ds)
        print("Verified LEVIR-CD patch counts:", counts)
    elif args.dataset == "oscd":
        counts = {}
        for sp in ("train", "val", "test"):
            try:
                ds = OSCDDataset(
                    args.source_dir,
                    split=sp,
                    bands=tuple(args.bands),
                    patch_size=args.patch_size,
                    stride=args.stride,
                )
                counts[sp] = len(ds)
            except Exception as exc:
                counts[sp] = f"Unavailable/Error: {exc}"
        print("Verified OSCD patch counts:", counts)


if __name__ == "__main__":
    main()

