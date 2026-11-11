# Project Status — Vision Transformers for Satellite Land-Use Monitoring

## Environment & Hardware Baseline

- **Operating System:** macOS 15.8.2 (`Darwin 24.6.0 arm64`)
- **CPU / SoC:** Apple M1 (`arm64`), 8 GiB unified memory
- **Python Version:** 3.11.5 (`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`)
- **Workspace Virtual Environment:** `/Users/ishaan/satellite/.venv` (`--system-site-packages`)
- **PyTorch Version:** 2.11.0 (`torchvision` 0.26.0)
- **Compute Devices:**
  - `CUDA`: Not available (Apple Silicon architecture)
  - `MPS`: Built in PyTorch (`mps_built=True`); runtime `mps_avail` depends on Metal access (false inside restricted sandbox, hardware-accelerated on host macOS)
  - `CPU`: Available (fallback)
- **Disk Capacity:** 228 GiB total, ~16 GiB free on `/System/Volumes/Data` (lightweight patch sizes and `--no-cache-dir` dependency management used intentionally)

---

## Implementation Checklist

- [x] **Phase 1: Repository inspection and project setup**
  - [x] Inspect OS, Python, PyTorch devices, disk space, and existing workspace files
  - [x] Create isolated `.venv` (`--system-site-packages`) and resolve `pandas`/`matplotlib`/NumPy 2.x ABI mismatch
  - [x] Verify installed ML/geospatial/GEE packages (`timm`, `rasterio`, `geopandas`, `pyproj`, `folium`, `earthengine-api`, `tensorboard`)
  - [x] Create project directory tree, `.gitignore`, `.env.example`, `pytest.ini`, `PROJECT_STATUS.md`
  - [x] Initialize `git` repository and `requirements.txt`
- [x] **Phase 2: Dataset loaders and preprocessing**
  - [x] Unified image loader (`RGB` + `GeoTIFF`/multispectral) & metadata preservation
  - [x] Alignment checks, reprojection/resampling to common grid, tiling & normalization
  - [x] Configurable Sentinel-2 cloud/shadow masking (`SCL` / `MSK_CLDPRB`) + RGB limitation warning
  - [x] Paired temporal geometric & radiometric augmentations
  - [x] LEVIR-CD, OSCD, and Custom Dataset loaders + validation utility
- [x] **Phase 3: CNN baseline implementation (`SiameseUNet`)**
- [x] **Phase 4: ViT change-detection implementation (`ViTChangeDetector`)**
- [x] **Phase 5: Swin Transformer implementation (`SwinChangeDetector`)**
- [x] **Phase 6: Training and checkpoint management**
- [x] **Phase 7: Evaluation and benchmarking**
- [x] **Phase 8: Inference and change-map generation**
- [x] **Phase 9: Explainable AI (ViT Attention Rollout, Swin Window Attention, CNN Grad-CAM)**
- [x] **Phase 10: Geospatial analysis (GeoTIFF preservation, geodesic/equal-area area calc, Folium maps)**
- [x] **Phase 11: Google Earth Engine integration & Satellite Data Explorer**
- [x] **Phase 12: FastAPI backend**
- [x] **Phase 13: Streamlit dashboard (8 pages + responsive dark/light theme)**
- [x] **Phase 14: Testing and bug fixing** (`pytest` full suite passing)
- [x] **Phase 15: Documentation and final integration**

---

## Dataset & Training Provenance Log

- **LEVIR-CD:** Pending manual download/placement under `data/raw/LEVIR-CD`
- **OSCD:** Pending manual download/placement under `data/raw/OSCD`
- **Google Earth Engine:** Optional external satellite acquisition (`COPERNICUS/S2_SR_HARMONIZED`), requires user GCP project & Earth Engine authentication
- **Benchmark Status:** Not evaluated on real datasets yet (synthetic test fixtures are strictly isolated to `tests/` and temporary directories and never recorded in real benchmark logs)

