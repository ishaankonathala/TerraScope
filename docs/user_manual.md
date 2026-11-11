# User Manual

Complete operational guide for running the FastAPI backend, 8-page Streamlit dashboard, dataset preparation utilities, training/benchmarking CLI scripts, and optional Google Earth Engine workflow.

---

## 1. Environment Setup

Run commands from the repository root (`/Users/ishaan/satellite`).

### Option A: Use the Configured Virtual Environment

```bash
source .venv/bin/activate
cp .env.example .env
```

### Option B: Fresh Python 3.11 Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Verify core libraries without external downloads:

```bash
.venv/bin/python -m pytest tests/test_environment.py
```

---

## 2. Launching the Application

The system runs cleanly as two local processes: the **FastAPI** inference/geospatial server and the **Streamlit** dashboard.

### Start the FastAPI Backend

```bash
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Start the Streamlit Frontend (in a second terminal)

```bash
.venv/bin/streamlit run frontend/app.py
```

If the backend listens on a custom host/port, set `API_BASE_URL` in `.env` before starting Streamlit.

---

## 3. Streamlit Dashboard Pages (`frontend/pages/`)

1. **[`frontend/app.py`](file:///Users/ishaan/satellite/frontend/app.py) & [`01_Dashboard.py`](file:///Users/ishaan/satellite/frontend/pages/01_Dashboard.py):** System health (`device`, memory, GEE status), training status badges (`untrained` vs `fine_tuned`), dataset availability, and latest logged benchmark summary.
2. **[`02_Change_Detection.py`](file:///Users/ishaan/satellite/frontend/pages/02_Change_Detection.py):** Upload before/after images (PNG/JPG/TIFF/GeoTIFF) or select GEE exports, pick an architecture (`Siamese U-Net`, `ViT`, `Swin`), tune threshold / tile overlap / ECC registration, and inspect probability maps, binary change masks, overlays, and alignment warnings.
3. **[`03_Model_Comparison.py`](file:///Users/ishaan/satellite/frontend/pages/03_Model_Comparison.py):** Side-by-side quantitative tables (`IoU`, `F1`, `Precision`, `Recall`, latency, memory) and charts from real benchmark JSON/CSV runs (`experiments/results/`). Distinguishes synthetic/smoke logs cleanly from real dataset runs.
4. **[`04_Explainable_AI.py`](file:///Users/ishaan/satellite/frontend/pages/04_Explainable_AI.py):** Inspects model-specific heatmaps — CNN Grad-CAM (`SiameseUNet`), ViT Attention Rollout (`ViTChangeDetector`), or Swin Window Attention (`SwinChangeDetector`) — alongside mandatory interpretability caveats.
5. **[`05_Geospatial_Analysis.py`](file:///Users/ishaan/satellite/frontend/pages/05_Geospatial_Analysis.py):** Displays preserved CRS, affine transform, WGS84 bounding boxes, polygon count, interactive Folium map overlays, and geodesic area in $\text{m}^2$ / hectares / $\text{km}^2$ (or image-relative statistics for unprojected PNG/JPG uploads).
6. **[`06_Dataset_Explorer.py`](file:///Users/ishaan/satellite/frontend/pages/06_Dataset_Explorer.py):** Checks local LEVIR-CD, OSCD, and custom directories, displays licensing notes, and lets you inspect valid patch configurations.
7. **[`07_Satellite_Data_Explorer.py`](file:///Users/ishaan/satellite/frontend/pages/07_Satellite_Data_Explorer.py):** Queries Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` via Google Earth Engine (AOI presets, bounding box, or GeoJSON), previews candidate cloud-filtered acquisitions & median composites, computes optional NDVI change indicators, and exports GeoTIFF pairs straight to change detection. Displays clear status instructions when `GEE_ENABLED=false`.
8. **[`08_Reports.py`](file:///Users/ishaan/satellite/frontend/pages/08_Reports.py):** Lists completed predictions and benchmarks, allowing download of structured JSON, CSV, or GeoJSON outputs.

---

## 4. FastAPI REST Endpoints

| Method & Route | Description |
| :--- | :--- |
| `GET /api/health` | Runtime device, Python/PyTorch status, and GEE availability |
| `GET /api/models` | Available architectures, backbones, checkpoints, and training status |
| `GET /api/datasets` | Availability check for LEVIR-CD, OSCD, and custom datasets |
| `POST /api/upload` | Validates and stores uploaded before/after image pairs |
| `POST /api/predict` | Runs tiled inference, postprocessing, geospatial stats, and optional XAI |
| `POST /api/evaluate` | Evaluates predictions against an uploaded ground-truth binary mask |
| `GET /api/experiments` | Lists logged training and benchmark runs from `experiments/` |
| `GET /api/results` | Lists saved inference result manifests |
| `GET /api/results/{result_id}` | Fetches metadata, URLs/paths, and metrics for a specific prediction |
| `GET /api/reports/{result_id}` | Exports structured report JSON / GeoJSON / summary artifacts |
| `GET /api/satellite/status` | Reports GEE initialization state without exposing credentials |
| `POST /api/satellite/search` | Searches Sentinel-2 scenes for AOI, dates, and max cloud threshold |
| `POST /api/satellite/preview` | Builds SCL-masked median composites and preview thumbnail URLs |
| `POST /api/satellite/export` | Downloads GeoTIFF pair locally or launches async Drive/GCS export |
| `GET /api/satellite/tasks/{task_id}` | Polls GEE export task status |

---

## 5. Command-Line Workflows (`scripts/`)

### 5.1 End-to-End Inference & XAI Smoke Check

```bash
.venv/bin/python scripts/run_demo.py --model vit
# Or pass real paired images + trained checkpoint:
.venv/bin/python scripts/run_demo.py \
  --before path/to/before.tif \
  --after path/to/after.tif \
  --model swin \
  --checkpoint models/checkpoints/swin_levir-cd_best.pt
```

### 5.2 Verify Local Datasets & Tile Custom Pairs

```bash
.venv/bin/python scripts/download_data.py --data-dir data --dataset all
.venv/bin/python scripts/prepare_dataset.py --dataset custom --source-dir path/to/custom --output-dir data/processed/custom
```

### 5.3 Train & Fairly Benchmark All 3 Architectures

```bash
.venv/bin/python scripts/train_model.py \
  --model swin \
  --dataset LEVIR-CD \
  --data-dir data/raw/LEVIR-CD \
  --epochs 20 \
  --batch-size 4 \
  --patch-size 256

.venv/bin/python scripts/evaluate_model.py \
  --dataset LEVIR-CD \
  --data-dir data/raw/LEVIR-CD \
  --split test
```

---

## 6. Running the Test Suite

```bash
.venv/bin/python -m pytest
```

Matplotlib cache automatically defaults to `/tmp/matplotlib_cache` so test execution works cleanly in read-only or sandboxed home environments.

