# Datasets Directory

This directory stores local satellite change-detection datasets without fabricating or silently replacing missing benchmarks.

## Directory Layout

- `data/raw/` — Unmodified dataset downloads (`LEVIR-CD/`, `OSCD/`).
- `data/processed/` — Tiled/prepared patches created by `scripts/prepare_dataset.py`.
- `data/samples/` — User-supplied custom image pairs (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`) or exported Google Earth Engine GeoTIFF pairs.

## Supported Public Datasets

### 1. LEVIR-CD (Building Change Detection)
- **Official Website:** https://justchenhao.github.io/LEVIR/
- **Expected Structure:**
  ```text
  data/raw/LEVIR-CD/
  ├── train/
  │   ├── A/      # T1 (before) RGB images
  │   ├── B/      # T2 (after) RGB images
  │   └── label/  # Binary change masks
  ├── val/
  │   ├── A/
  │   ├── B/
  │   └── label/
  └── test/
      ├── A/
      ├── B/
      └── label/
  ```
- **Licensing / Terms:** Please review the original LEVIR-CD academic usage terms on the official project page before downloading or redistributing.

### 2. OSCD (Onera Satellite Change Detection)
- **Official Website:** https://rcdaudt.github.io/oscd/
- **Expected Structure:**
  ```text
  data/raw/OSCD/
  ├── images/  # or root city directories (e.g., abudhabi/, beirut/, ...)
  │   └── <city>/
  │       ├── imgs_1_rect/ (or imgs_1/)  # B01..B12 Sentinel-2 GeoTIFF bands
  │       ├── imgs_2_rect/ (or imgs_2/)
  │       └── pair/ (optional RGB preview)
  └── labels/  # or train_labels/ / test_labels/ containing <city>/cm/cm.png (or <city>-cm.tif)
  ```
- **Licensing / Terms:** Released under CC-BY-4.0 / academic terms documented on the OSCD project website and IEEE DataPort.

Use `python scripts/download_data.py --help` to check dataset availability and directory setup instructions.

