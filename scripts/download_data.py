"""Dataset setup checker and official acquisition guide (does not fabricate data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets import LEVIR_CD_URL, OSCD_URL


def check_levir_cd(data_dir: Path) -> bool:
    levir = data_dir / "raw" / "LEVIR-CD"
    ok = all((levir / sp / sub).is_dir() for sp in ("train", "val", "test") for sub in ("A", "B", "label"))
    print(f"[LEVIR-CD] Path: {levir} -> {'FOUND' if ok else 'NOT FOUND'}")
    if not ok:
        print(f"  Official site: {LEVIR_CD_URL}")
        print("  Required layout: data/raw/LEVIR-CD/{train,val,test}/{A,B,label}/")
        print("  Note: Please review LEVIR-CD academic license terms before downloading.")
    return ok


def check_oscd(data_dir: Path) -> bool:
    oscd = data_dir / "raw" / "OSCD"
    img_dir = oscd / "images" if (oscd / "images").is_dir() else oscd
    cities = [p for p in img_dir.iterdir() if p.is_dir()] if img_dir.is_dir() else []
    ok = len(cities) > 0
    print(f"[OSCD] Path: {oscd} -> {'FOUND (' + str(len(cities)) + ' cities)' if ok else 'NOT FOUND'}")
    if not ok:
        print(f"  Official site: {OSCD_URL}")
        print("  Required layout: data/raw/OSCD/images/<city>/imgs_{1,2}_rect/ + labels/")
        print("  License: CC-BY-4.0 / academic terms documented by Onera.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local satellite datasets and display acquisition instructions.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--dataset", default="all", choices=["all", "levir", "oscd"])
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    for sub in ("raw", "processed", "samples"):
        (data_path / sub).mkdir(parents=True, exist_ok=True)

    if args.dataset in {"all", "levir"}:
        check_levir_cd(data_path)
    if args.dataset in {"all", "oscd"}:
        check_oscd(data_path)


if __name__ == "__main__":
    main()

