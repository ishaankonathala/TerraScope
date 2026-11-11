# Google Earth Engine (GEE) Integration Guide

Google Earth Engine serves as the **single primary external satellite imagery acquisition and cloud-based preprocessing platform** for this project ([Official Site](https://earthengine.google.com/) | [Python API Docs](https://developers.google.com/earth-engine/guides/python_install)).

GEE integration is **strictly optional**: when `GEE_ENABLED=false` or credentials are absent, all local GeoTIFF/PNG/JPG workflows, dataset loaders, training, benchmarking, and Streamlit pages remain fully operational.

---

## 1. Eligibility, Account Setup & Quotas

1. **Eligibility & Registration:** Register an Earth Engine account and link a Google Cloud Project (`GEE_PROJECT_ID`). Note that GEE non-commercial / academic tiers have specific licensing terms, concurrent export quotas, and interactive `getDownloadURL` / `getThumbURL` payload limits; commercial deployments require an Earth Engine commercial agreement.
2. **Environment Configuration (`.env`):** Copy `.env.example` to `.env` and populate **only** the variables you need (never commit credentials or keys to Git):

```env
GEE_ENABLED=true
GEE_PROJECT_ID=your-gcp-project-id
# Option A: Leave service account fields blank to use interactive credentials
# Option B: Service account authentication
GEE_SERVICE_ACCOUNT=
GEE_PRIVATE_KEY_PATH=
# Supported export modes: local_geotiff, google_drive, gcs
GEE_EXPORT_DESTINATION=local_geotiff
GEE_DRIVE_FOLDER=satellite_change_exports
GEE_GCS_BUCKET=
```

3. **Interactive Authentication (Option A — recommended for local development):**

```bash
.venv/bin/earthengine authenticate
```

Credentials are stored locally under `~/.config/earthengine/credentials` and detected automatically by [`initialize_gee`](file:///Users/ishaan/satellite/ml/satellite/gee_client.py#L66).

4. **Service Account Authentication (Option B — headless servers):** Provide `GEE_SERVICE_ACCOUNT`, `GEE_PRIVATE_KEY_PATH`, and `GEE_PROJECT_ID`. [`GEEStatus`](file:///Users/ishaan/satellite/ml/satellite/gee_client.py#L45) reports structured status codes (`ready`, `disabled`, `unauthenticated`, `missing_project`, `missing_library`, `error`) without exposing private keys.

---

## 2. Sentinel-2 Data Source (`COPERNICUS/S2_SR_HARMONIZED`)

Implemented in [`ml/satellite/sentinel2.py`](file:///Users/ishaan/satellite/ml/satellite/sentinel2.py):

| Band | Name | Native Scale | Central Wavelength | Typical Role |
| :--- | :--- | :---: | :---: | :--- |
| `B2` | Blue | 10 m | ~490 nm | True-color composite, atmospheric scattering |
| `B3` | Green | 10 m | ~560 nm | True-color composite, water/vegetation contrast |
| `B4` | Red | 10 m | ~665 nm | True-color composite, NDVI red band |
| `B8` | NIR | 10 m | ~842 nm | Vegetation biomass / NDVI NIR band |
| `B11` | SWIR 1 | 20 m | ~1610 nm | Moisture / burn scar / built-up separation |
| `B12` | SWIR 2 | 20 m | ~2190 nm | Soil / urban / fire severity separation |
| `SCL` | Scene Classification Layer | 20 m | — | Per-pixel cloud, cirrus, and shadow masking |

### Safeguards & Export Bounds

- **AOI Specification (`AOISpec`):** Supports `(west, south, east, north)` WGS84 bounding boxes, uploaded/drawn GeoJSON `Polygon` / `MultiPolygon` geometries, or 5 built-in monitoring presets (`PRESET_REGIONS`). Bounding boxes wider or taller than **2.0 degrees** are rejected to respect interactive export size limits.
- **Export Scale (`scale_m`):** Bounded between `10` m (native Sentinel-2 RGB/NIR GSD) and `250` m in [`validate_sentinel2_params`](file:///Users/ishaan/satellite/ml/satellite/sentinel2.py#L100).
- **Mixed-Resolution Co-Registration:** When combining `10 m` and `20 m` bands, [`build_temporal_composite`](file:///Users/ishaan/satellite/ml/satellite/composites.py#L22) explicitly applies `.reproject(crs=target_crs, scale=clean_scale)` so before and after composites share an exact target projection and pixel grid.

---

## 3. Cloud Filtering vs. Pixel-Level SCL Masking

1. **Scene-Level Filter:** `.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))` drops heavily clouded scenes first. If zero scenes match, the pipeline stops cleanly with `status_code="empty_collection"` and instructs the user to widen dates or raise `max_cloud_pct`.
2. **Pixel-Level SCL Masking ([`mask_s2_scl_ee`](file:///Users/ishaan/satellite/ml/satellite/sentinel2.py#L130)):** Excludes SCL classes:
   - `1`: Saturated or defective
   - `3`: Cloud shadow
   - `8`: Cloud medium probability
   - `9`: Cloud high probability
   - `10`: Thin cirrus
3. **Documented Limitations:**
   - `SCL` has `20 m` native resolution while `B2/B3/B4/B8` are `10 m`; thin cloud fringes, sub-pixel cloud shadows, and bright urban roofs or snow can still leak into composites.
   - Median compositing reduces transient cloud outliers only when multiple valid acquisitions exist across the date window.

---

## 4. Spectral Change Indicator (NDVI)

Implemented in [`ml/satellite/indices.py`](file:///Users/ishaan/satellite/ml/satellite/indices.py):

$$\text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$

- Guards against zero denominators (`|B8 + B4| <= 1e-6`), NaN/Inf values, and SCL-masked pixels before computing summary statistics (`mean`, `median`, `std`, `min`, `max`, `valid_coverage_pct`) and [`compare_ndvi`](file:///Users/ishaan/satellite/ml/satellite/indices.py#L51) delta indicators.
- **Scientific Caveat (`NDVI_SCIENTIFIC_NOTE`):** NDVI decreases or increases alone do not prove land-cover conversion (deforestation, construction, wildfire); seasonal phenology, rainfall, crop harvesting cycles, and sensor illumination geometry all shift NDVI independently.

---

## 5. Export Modes & Pipeline Integration

Implemented in [`ml/satellite/export.py`](file:///Users/ishaan/satellite/ml/satellite/export.py):

- **`local_geotiff`:** Downloads matched before and after GeoTIFFs via Earth Engine `getDownloadURL(..., format="GEO_TIFF")` directly into `outputs/gee_exports/`, immediately ready to feed into [`ChangePredictor`](file:///Users/ishaan/satellite/ml/inference/predictor.py) or the **Change Detection** page without manual re-upload.
- **`google_drive` / `gcs`:** Starts asynchronous Earth Engine batch tasks (`ee.batch.Export.image.toDrive` / `toCloudStorage`), tracks task IDs in [`GEETaskRegistry`](file:///Users/ishaan/satellite/ml/satellite/export.py#L25), and supports status polling via `GET /api/satellite/export/{task_id}`.

