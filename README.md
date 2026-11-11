# Vision Transformers for Satellite Land-Use Monitoring

An end-to-end geospatial deep-learning research platform for multi-temporal satellite image change detection, comparing **Siamese U-Net (CNN)**, **Vision Transformer (Siamese ViT)**, and **Swin Transformer (Hierarchical Siamese)** architectures under reproducible training, evaluation, explainability, and Google Earth Engine workflows.

---

## 1. Project Overview

This system detects pixel-level land-use changes across paired bi-temporal satellite or aerial observations (`T1` before, `T2` after). Rather than assuming Transformer models automatically outperform Convolutional Neural Networks, the codebase provides standardized preprocessing, identical train/validation/test splits, matched loss/threshold evaluation, explainable AI visualizations, real-world area estimation, and interactive monitoring dashboards to establish comparative performance experimentally.

## 2. Problem Statement

Manual interpretation of multi-temporal Earth observation imagery is slow, subjective, and non-scalable across urban expansion, deforestation fronts, flood footprints, and post-hazard monitoring. Automated pixel-level change detection faces severe challenges:

- Seasonal phenology and illumination shifts between acquisition dates
- Cloud and cloud-shadow contamination
- Georeferencing CRS and spatial resolution mismatches (e.g., 10 m vs. 20 m Sentinel-2 bands)
- Extreme class imbalance (changed pixels often comprise <5% of scenes)
- Lack of model transparency without spatial attribution

## 3. Objectives & Research Questions

- **RQ1:** How do Vision Transformer-based architectures (`ViT`, `Swin Transformer`) compare with traditional CNN baselines (`Siamese U-Net`) under identical dataset splits and preprocessing?
- **RQ2:** How effectively do global self-attention, hierarchical shifted-window attention, and convolutional skip connections capture spatial changes?
- **RQ3:** How do seasonal differences, cloud interference, and image misalignment impact detection reliability?
- **RQ4:** Can attention-based visualizations (`Attention Rollout`, `Swin Window Attention`) and `Grad-CAM` improve interpretability of predicted changes?

## 4. System Architecture

1. **Preprocessing & Geospatial Layer (`ml/preprocessing/`, `ml/geospatial/`):** Format/dimension/channel validation, raster CRS reprojection (`rasterio.warp.reproject`), Sentinel-2 SCL/cloud-probability masking, joint temporal normalization, overlapping Hann-blended tiling, geodesic/equal-area area calculation, and Folium mapping.
2. **Model Zoo (`ml/models/`):** Shared-weight Siamese encoders (`SiameseUNet`, `ViTChangeDetector`, `SwinChangeDetector`) unified via `get_model(...)` with `TemporalFusionBlock` (`[|f1 - f2|, f1, f2]`) and multi-scale decoders.
3. **Training & Benchmarking (`ml/training/`, `ml/evaluation/`):** BCE/Dice/Combined losses with valid-mask exclusion, Adam/AdamW + Cosine/Plateau schedulers, best/latest checkpointing + resume, TensorBoard + JSON/CSV experiment logging, and unified 3-model benchmarking.
4. **Explainable AI (`ml/explainability/`):** Genuine ViT Attention Rollout, Swin Window-Attention aggregation, and Siamese U-Net Grad-CAM.
5. **Google Earth Engine Integration (`ml/satellite/`):** Optional `COPERNICUS/S2_SR_HARMONIZED` search, SCL pixel masking, temporal median composites, NDVI comparison, and GeoTIFF export (local, Drive, GCS).
6. **Backend & Frontend (`backend/`, `frontend/`):** FastAPI REST service + 8-page Streamlit geospatial analytics dashboard (with Dark/Light theme toggle and fallback direct local runtime).

## 5. Technology Stack

- **AI / ML:** Python 3.11, PyTorch, Torchvision, `timm`, NumPy, Pandas, Scikit-learn, TensorBoard
- **Geospatial / Imaging:** Rasterio, GeoPandas, PyProj, Shapely, OpenCV, Pillow, Matplotlib
- **Satellite Acquisition (Optional):** `earthengine-api`, Google Auth
- **Backend / Frontend / Reports:** FastAPI, Uvicorn, Pydantic, Streamlit, Plotly, Folium, ReportLab, Pytest

## 6. Dataset Details

| Dataset | Purpose | Bands / Format | Official Source |
| :--- | :--- | :--- | :--- |
| **LEVIR-CD** | Building change detection | RGB (`A/`, `B/`, `label/`) | https://justchenhao.github.io/LEVIR/ |
| **OSCD** | Urban multispectral change detection | Sentinel-2 13 bands (`GeoTIFF` / `PNG`) | https://rcdaudt.github.io/oscd/ |
| **Custom / GEE** | User pairs & Sentinel-2 GeoTIFF exports | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | Local uploads or GEE `COPERNICUS/S2_SR_HARMONIZED` |

Public datasets are **not** fabricated or bundled; see `docs/dataset_documentation.md` for licenses and manual setup.

## 7. Model Descriptions

- **Siamese U-Net (`siamese_unet`):** Shared CNN encoder (`resnet18` or `custom_cnn`) + multi-scale temporal fusion + skip-connection decoder.
- **Vision Transformer (`vit`):** Shared `timm` ViT (`vit_tiny_patch16_224` with `dynamic_img_size=True`) reshaping multi-depth spatial patch tokens into 2D grids and decoding via progressive pyramid upsampling.
- **Swin Transformer (`swin`):** Shared hierarchical `timm` Swin (`swin_tiny_patch4_window7_224`) extracting 4 spatial stages (`H/4..H/32`) fused by a multi-scale FPN/U-Net decoder.

## 8. Installation Instructions

```bash
# 1. Create virtual environment (Python 3.11)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment configuration
cp .env.example .env
```

## 9. Dataset Preparation Instructions

```bash
# Verify dataset folder structure and acquisition instructions
python scripts/download_data.py

# Prepare / validate tiled patches for custom pairs
python scripts/prepare_dataset.py --dataset custom --source-dir data/samples --output-dir data/processed --patch-size 256
```

## 10. Training Commands

```bash
# Train Siamese U-Net baseline (Apple M1 / CPU / CUDA auto-detected)
python scripts/train_model.py --model siamese_unet --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20 --batch-size 4 --patch-size 256

# Train Vision Transformer (ViT)
python scripts/train_model.py --model vit --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20 --batch-size 4 --patch-size 256

# Train Swin Transformer
python scripts/train_model.py --model swin --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20 --batch-size 4 --patch-size 256
```

## 11. Evaluation Commands

```bash
# Benchmark all three architectures on identical test splits
python scripts/evaluate_model.py --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --split test --patch-size 256 --batch-size 4
```

## 12. Inference Instructions

```bash
# Run end-to-end verification demo (isolated /tmp synthetic pair or user files)
python scripts/run_demo.py --model siamese_unet
python scripts/run_demo.py --model vit --before path/to/t1.tif --after path/to/t2.tif
```

## 13. Backend Startup

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
# Interactive Swagger docs: http://127.0.0.1:8000/docs
```

## 14. Frontend Startup

```bash
streamlit run frontend/app.py
```

## 15. API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Runtime, compute device, and GEE availability status |
| `GET` | `/api/models` | Architecture registry and saved checkpoint list |
| `GET` | `/api/datasets` | Local LEVIR-CD, OSCD, and sample dataset inventory |
| `POST` | `/api/upload` | Validate & stage before/after image pairs |
| `POST` | `/api/predict` | Run change detection (+ optional XAI & geospatial stats) |
| `POST` | `/api/evaluate` | Evaluate a saved prediction against a ground-truth mask |
| `GET` | `/api/experiments` | Logged real training & benchmark metrics (`Not evaluated` if none) |
| `GET` | `/api/results` | List recent analyses |
| `GET` | `/api/results/{result_id}` | Fetch detailed prediction summary |
| `GET` | `/api/reports/{result_id}` | Export `json`, `csv`, `pdf`, `png`, `overlay`, `geotiff`, `geojson`, `xai_panel` |
| `GET` | `/api/satellite/status` | Google Earth Engine auth status & preset AOIs |
| `POST` | `/api/satellite/search` | Query `COPERNICUS/S2_SR_HARMONIZED` candidate scenes |
| `POST` | `/api/satellite/preview` | Preview SCL-masked median composites & NDVI |
| `POST` | `/api/satellite/export` | Export aligned GeoTIFF pair locally or to Drive/GCS |
| `GET` | `/api/satellite/tasks/{task_id}` | Poll GEE export task status |

## 16. Expected Folder Structure

See `docs/architecture.md` for the complete modular layout (`backend/`, `frontend/`, `ml/`, `data/`, `models/`, `experiments/`, `outputs/`, `scripts/`, `docs/`, `tests/`).

## 17. Limitations

- Binary change detection masks indicate spatial/radiometric differences only; identifying specific land-cover categories (e.g., deforestation vs. construction) requires multi-class labeled training data.
- Standard RGB image uploads (`.png`, `.jpg`) lack georeferencing metadata and spectral quality bands, so geographic area and Sentinel-2 cloud masking are unavailable for those inputs.
- Model performance metrics on real LEVIR-CD / OSCD benchmarks remain `Not evaluated` until checkpoints are trained on the respective datasets.

## 18. Future Enhancements

- Multi-class semantic land-use change segmentation (`forest loss`, `urban growth`, `flood`, `burn scar`) when compatible multi-class datasets (e.g., SECOND / DynamicEarthNet) are supplied.
- Bi-temporal cross-attention decoder heads and SAR/optical fusion (`Sentinel-1` + `Sentinel-2`).
- Automated scheduled monitoring alerts via GEE batch pipelines.

