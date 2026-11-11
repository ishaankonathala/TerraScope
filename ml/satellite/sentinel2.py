"""Sentinel-2 Surface Reflectance Harmonized query builder, AOI parsing, and SCL masking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Sequence

from ml.satellite.gee_client import GEEConfig, initialize_gee

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Documented native spatial resolutions (meters) and spectral roles
SENTINEL2_BANDS: dict[str, dict[str, Any]] = {
    "B2": {"name": "Blue", "native_scale_m": 10, "wavelength_nm": 490},
    "B3": {"name": "Green", "native_scale_m": 10, "wavelength_nm": 560},
    "B4": {"name": "Red", "native_scale_m": 10, "wavelength_nm": 665},
    "B8": {"name": "NIR", "native_scale_m": 10, "wavelength_nm": 842},
    "B11": {"name": "SWIR 1", "native_scale_m": 20, "wavelength_nm": 1610},
    "B12": {"name": "SWIR 2", "native_scale_m": 20, "wavelength_nm": 2190},
    "SCL": {"name": "Scene Classification Layer", "native_scale_m": 20, "wavelength_nm": None},
}

# Built-in reference monitoring regions for quick selection (lat, lon, half_width_deg)
PRESET_REGIONS: dict[str, dict[str, Any]] = {
    "Bengaluru Urban Expansion Corridor": {
        "bbox": [77.65, 12.82, 77.73, 12.90],
        "description": "Peri-urban growth corridor in southeastern Bengaluru, India.",
    },
    "Delhi NCR Peri-Urban Fringe": {
        "bbox": [77.02, 28.42, 77.10, 28.50],
        "description": "Rapid infrastructure corridor in Gurugram / Delhi NCR.",
    },
    "Dubai Waterfront & Desert Expansion": {
        "bbox": [55.12, 25.02, 55.20, 25.10],
        "description": "Coastal and urban development in Dubai, UAE.",
    },
    "Rondonia Forest Clearing Sample": {
        "bbox": [-63.12, -10.58, -63.04, -10.50],
        "description": "Mixed agricultural and tropical forest edge in Rondonia, Brazil.",
    },
    "Northern California Foothills": {
        "bbox": [-121.62, 39.72, -121.54, 39.80],
        "description": "Conifer/chaparral foothills terrain in Butte County, California.",
    },
}


@dataclass
class AOISpec:
    """Validated geographic Area of Interest specification."""

    bbox: tuple[float, float, float, float] | None = None  # (west, south, east, north)
    geojson_geometry: dict[str, Any] | None = None
    place_name: str | None = None

    def resolve_bounds(self) -> tuple[float, float, float, float]:
        if self.place_name and self.place_name in PRESET_REGIONS:
            w, s, e, n = PRESET_REGIONS[self.place_name]["bbox"]
            return (float(w), float(s), float(e), float(n))

        if self.bbox is not None:
            w, s, e, n = [float(v) for v in self.bbox]
            if not (-180.0 <= w < e <= 180.0 and -90.0 <= s < n <= 90.0):
                raise ValueError(
                    f"Invalid bbox {self.bbox}. Expected (west, south, east, north) in WGS84 degrees."
                )
            if (e - w) > 2.0 or (n - s) > 2.0:
                raise ValueError(
                    "AOI span exceeds 2.0 degrees. Select a smaller region to stay within GEE export limits."
                )
            return (w, s, e, n)

        if self.geojson_geometry is not None:
            geom_type = self.geojson_geometry.get("type")
            coords = self.geojson_geometry.get("coordinates")
            if geom_type not in {"Polygon", "MultiPolygon"} or not coords:
                raise ValueError("GeoJSON AOI geometry must be a valid Polygon or MultiPolygon.")
            ring = coords[0] if geom_type == "Polygon" else coords[0][0]
            lons = [float(pt[0]) for pt in ring]
            lats = [float(pt[1]) for pt in ring]
            return (min(lons), min(lats), max(lons), max(lats))

        raise ValueError("Provide bbox, geojson_geometry, or a supported place_name.")

    def to_ee_geometry(self) -> Any:
        if ee is None:
            raise RuntimeError("earthengine-api is not installed.")
        if self.geojson_geometry is not None:
            return ee.Geometry(self.geojson_geometry)
        w, s, e, n = self.resolve_bounds()
        return ee.Geometry.Rectangle([w, s, e, n])


def validate_sentinel2_params(
    bands: Sequence[str],
    start_date: str,
    end_date: str,
    max_cloud_pct: float,
    scale_m: int,
) -> tuple[list[str], int]:
    """Validate bands, ISO dates, cloud filter, and export scale safely."""
    clean_bands = [b.strip().upper() for b in bands if b.strip()]
    if not clean_bands:
        raise ValueError("At least one Sentinel-2 band must be selected.")
    invalid = [b for b in clean_bands if b not in SENTINEL2_BANDS]
    if invalid:
        raise ValueError(
            f"Unsupported Sentinel-2 band(s): {invalid}. Supported: {list(SENTINEL2_BANDS.keys())}"
        )

    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date)
    if d1 < d0:
        raise ValueError(f"end_date ({end_date}) must be on or after start_date ({start_date}).")

    if not (0.0 <= max_cloud_pct <= 100.0):
        raise ValueError("max_cloud_pct must be between 0 and 100.")
    if not (10 <= scale_m <= 250):
        raise ValueError("Export scale_m must be between 10m (native Sentinel-2 RGB/NIR) and 250m.")

    return clean_bands, scale_m


def mask_s2_scl_ee(image: Any, mask_classes: Sequence[int] = (1, 3, 8, 9, 10)) -> Any:
    """Apply pixel-level Sentinel-2 Scene Classification Layer (SCL) cloud/shadow mask in GEE.

    Note: SCL (20m native) detects medium/high cloud probability, thin cirrus, cloud
    shadows, and saturated pixels, but misses sub-pixel cloud edges and bright urban surfaces.
    """
    scl = image.select("SCL")
    keep_mask = scl.neq(int(mask_classes[0]))
    for cls_id in mask_classes[1:]:
        keep_mask = keep_mask.And(scl.neq(int(cls_id)))
    return image.updateMask(keep_mask)


def search_sentinel2_scenes(
    aoi: AOISpec,
    start_date: str,
    end_date: str,
    max_cloud_pct: float = 20.0,
    max_results: int = 20,
    config: GEEConfig | None = None,
) -> dict[str, Any]:
    """Query ``COPERNICUS/S2_SR_HARMONIZED`` metadata for candidate acquisition dates."""
    validate_sentinel2_params(["B4", "B3", "B2"], start_date, end_date, max_cloud_pct, 10)
    status = initialize_gee(config, force=True)
    if not status.initialized:
        return {
            "ok": False,
            "status_code": status.status_code,
            "error": status.message,
            "collection": SENTINEL2_COLLECTION,
            "scenes": [],
        }

    try:
        geom = aoi.to_ee_geometry()
        col = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_cloud_pct)))
            .sort("system:time_start")
            .limit(int(max_results))
        )
        info = col.getInfo()
        features = info.get("features", []) if isinstance(info, dict) else []

        scenes: list[dict[str, Any]] = []
        for feat in features:
            props = feat.get("properties", {})
            ts_ms = props.get("system:time_start")
            acq_date = (
                datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if ts_ms is not None
                else "Unknown"
            )
            scenes.append(
                {
                    "scene_id": feat.get("id") or props.get("system:index"),
                    "acquisition_date": acq_date,
                    "cloudy_pixel_percentage": round(
                        float(props.get("CLOUDY_PIXEL_PERCENTAGE", 0.0)), 2
                    ),
                    "mgrs_tile": props.get("MGRS_TILE"),
                    "spacecraft": props.get("SPACECRAFT_NAME"),
                }
            )

        if not scenes:
            return {
                "ok": True,
                "empty": True,
                "collection": SENTINEL2_COLLECTION,
                "scenes": [],
                "message": (
                    f"No Sentinel-2 scenes found between {start_date} and {end_date} with "
                    f"metadata cloud cover <= {max_cloud_pct}%. Try expanding the date window "
                    "or raising the cloud percentage threshold."
                ),
            }

        return {
            "ok": True,
            "empty": False,
            "collection": SENTINEL2_COLLECTION,
            "count": len(scenes),
            "scenes": scenes,
            "aoi_bounds": aoi.resolve_bounds(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": "gee_query_error",
            "error": f"Failed to query Earth Engine Sentinel-2 catalog: {exc}",
            "collection": SENTINEL2_COLLECTION,
            "scenes": [],
        }

