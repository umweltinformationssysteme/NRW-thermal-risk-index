"""
NRW Thermal Risk Index
======================
Daily perceived temperature forecast for all municipalities in
North Rhine-Westphalia (NRW), classified according to the DWD
thermal hazard index (Klima-Michel model / VDI 3787).

Data source : DWD OpenData – Health Forecasts (ICON-EU-Nest, GRIB2)
Classification: VDI 3787 Part 2 / DWD Klima-Michel model
"""

import json
import logging
import re
import sys
import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).parent
CSV_PATH      = REPO_ROOT / "data" / "municipality_nrw.csv"
OUTPUT_JSON   = REPO_ROOT / "output" / "thermal_index_nrw.json"
OUTPUT_GEOJSON = REPO_ROOT / "output" / "thermal_index_nrw.geojson"
GEOJSON_SRC   = REPO_ROOT / "municipality_nrw.geojson"   # polygon boundaries
README_FILE   = REPO_ROOT / "README.md"
TEMPLATE_FILE = REPO_ROOT / "README_template.md"

BASE_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/"
TOP_N    = 10

# ---------------------------------------------------------------------------
# Full 24-step colour scale from the official DWD legend CSV
# (Legende_Thermischer_Gefahrenindex.CSV)
# Each entry: (PT upper bound °C, sensation_en, risk_en, hex, text_hex)
# ---------------------------------------------------------------------------
COLOUR_SCALE = [
    ( -39, "Very cold",     "Very high", "#011AFD", "#ffffff"),
    ( -34, "Very cold",     "Very high", "#1967FD", "#ffffff"),
    ( -30, "Cold",          "High",      "#2F72FF", "#ffffff"),
    ( -26, "Cold",          "High",      "#3D81FE", "#ffffff"),
    ( -21, "Cool",          "Elevated",  "#5BA0FD", "#000000"),
    ( -17, "Cool",          "Elevated",  "#6AAFFD", "#000000"),
    ( -13, "Cool",          "Elevated",  "#81BEFD", "#000000"),
    (  -8, "Slightly cool", "Low",       "#A5DAFA", "#000000"),
    (  -4, "Slightly cool", "Low",       "#B4E6FF", "#000000"),
    (   0, "Slightly cool", "Low",       "#CAEFFF", "#000000"),
    (   4, "Comfortable",   "None",      "#8DFF8D", "#000000"),
    (   8, "Comfortable",   "None",      "#01FE03", "#000000"),
    (  12, "Comfortable",   "None",      "#00E700", "#000000"),
    (  16, "Comfortable",   "None",      "#88FF02", "#000000"),
    (  20, "Comfortable",   "None",      "#C8FF2F", "#000000"),
    (  23, "Slightly warm", "Low",       "#FFFF7D", "#000000"),
    (  26, "Slightly warm", "Low",       "#FEE362", "#000000"),
    (  29, "Warm",          "Elevated",  "#FFAF34", "#000000"),
    (  32, "Warm",          "Elevated",  "#FD7D1A", "#000000"),
    (  35, "Hot",           "High",      "#FF3001", "#ffffff"),
    (  38, "Hot",           "High",      "#E11902", "#ffffff"),
    (  41, "Very hot",      "Very high", "#AD1AE4", "#ffffff"),
    (  44, "Very hot",      "Very high", "#E04BFF", "#000000"),
    ( 999, "Very hot",      "Very high", "#FD7EFF", "#000000"),
]

# Compact 9-group legend for the README table (one row per sensation)
LEGEND_GROUPS = [
    # (sensation, risk, representative_hex, pt_range)
    ("Very cold",     "Very high", "#1967FD", "≤ −39 °C"),
    ("Cold",          "High",      "#3D81FE", "−39 to −26 °C"),
    ("Cool",          "Elevated",  "#81BEFD", "−26 to −13 °C"),
    ("Slightly cool", "Low",       "#B4E6FF", "−13 to 0 °C"),
    ("Comfortable",   "None",      "#00E700", "0 to +20 °C"),
    ("Slightly warm", "Low",       "#FEE362", "+20 to +26 °C"),
    ("Warm",          "Elevated",  "#FD7D1A", "+26 to +32 °C"),
    ("Hot",           "High",      "#E11902", "+32 to +38 °C"),
    ("Very hot",      "Very high", "#E04BFF", "≥ +38 °C"),
]

def classify(temp_c: float) -> dict:
    for upper, sensation, risk, bg, fg in COLOUR_SCALE:
        if temp_c <= upper:
            return {"sensation": sensation, "risk": risk,
                    "bg_color": bg, "fg_color": fg}
    return {"sensation": "Unknown", "risk": "–",
            "bg_color": "#cccccc", "fg_color": "#000000"}

def badge(hex_bg: str) -> str:
    """Colour square badge via placehold.co — same style as NRW Ozone project."""
    c = hex_bg.lstrip("#")
    return f"![](https://placehold.co/14x14/{c}/{c}.png)"

def to_celsius(v: float) -> float:
    return v - 273.15 if v > 100 else v

# ---------------------------------------------------------------------------
# DWD directory → latest GFT URL
# ---------------------------------------------------------------------------
def get_latest_gft_url() -> str:
    log.info("Scanning DWD directory: %s", BASE_URL)
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
    files = re.findall(r'href="([^"]*icreu_gft[^"]*\.(?:bin|grib2))"', r.text)
    if not files:
        raise RuntimeError("No GFT files found in DWD directory.")
    files.sort()
    url = BASE_URL + files[-1]
    log.info("Latest file: %s", files[-1])
    return url

# ---------------------------------------------------------------------------
# Parse model run and validity time from filename
# ---------------------------------------------------------------------------
def parse_filename(url: str):
    fname = url.split("/")[-1]
    run_dt = valid_dt = None
    m = re.search(r"EDZW_(\d{14})", fname)
    if m:
        run_dt = datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
    m = re.search(r"_(\d{10})_HPC", fname)
    if m:
        valid_dt = datetime.datetime.strptime("20" + m.group(1), "%Y%m%d%H%M").replace(
            tzinfo=datetime.timezone.utc)
    return run_dt, valid_dt

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(url: str, dest: Path) -> None:
    log.info("Downloading forecast file …")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        dest.write_bytes(b"".join(r.iter_content(65536)))
    log.info("  → %.1f MB saved", dest.stat().st_size / 1e6)

# ---------------------------------------------------------------------------
# Open GRIB2
# ---------------------------------------------------------------------------
def open_grib(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(str(path), engine="cfgrib",
                               backend_kwargs={"indexpath": ""})
    except Exception as e:
        log.warning("open_dataset failed (%s) – trying open_datasets", e)
    import cfgrib
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not datasets:
        raise RuntimeError("Could not read GRIB2 file.")
    log.info("open_datasets: %d message(s) found", len(datasets))
    return datasets[0]

def find_var(ds: xr.Dataset) -> str:
    for c in ("PT1M", "PT", "t2m", "2t", "pt", "perceived_temperature"):
        if c in ds.data_vars:
            return c
    first = list(ds.data_vars)[0]
    log.warning("No standard variable found – using '%s'. Available: %s",
                first, list(ds.data_vars))
    return first

# ---------------------------------------------------------------------------
# Vectorised processing via KD-Tree  (~100× faster than 396× xr.sel)
# ---------------------------------------------------------------------------
def process(ds: xr.Dataset, df: pd.DataFrame,
            dates: dict[str, datetime.date]) -> list[dict]:

    var = find_var(ds)
    log.info("Using variable '%s'", var)

    # --- KD-Tree: handle 1-D and 2-D coordinate arrays ---
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
        grid_lat = lat_2d.ravel()
        grid_lon = lon_2d.ravel()
    else:
        # 2-D curvilinear grid (e.g. DWD ICON-EU: 657 × 1377)
        grid_lat = lat_vals.ravel()
        grid_lon = lon_vals.ravel()

    tree = cKDTree(np.column_stack([grid_lat, grid_lon]))
    _, nn_idx = tree.query(df[["lat", "lon"]].values)
    log.info("KD-Tree: %d grid points, %d municipalities mapped",
             len(grid_lat), len(df))

    # --- Time axis ---
    tc = next((c for c in ("valid_time", "time") if c in ds.coords), None)
    if tc is None:
        log.error("No time coordinate found in dataset.")
        sys.exit(1)
    times = pd.to_datetime(ds[tc].values)
    is_scalar = times.ndim == 0

    # --- Flatten raw data to (time × grid) ---
    raw = ds[var].values
    if is_scalar:
        flat  = raw.ravel()[np.newaxis, :]
        times = np.array([times])
    elif raw.ndim == 3:
        flat = raw.reshape(raw.shape[0], -1)
    elif raw.ndim == 2:
        flat = raw if raw.shape[0] == len(times) else raw.T
    else:
        flat = raw.reshape(1, -1)
    log.info("Data shape after flattening: %s (timesteps × grid points)", flat.shape)

    # Pre-compute date masks
    date_masks = {k: np.array([t.date() == d for t in times])
                  for k, d in dates.items()}

    out = []
    for i, row in df.iterrows():
        gi = nn_idx[i]
        forecasts = {}
        for key, mask in date_masks.items():
            if not mask.any():
                forecasts[key] = None
                continue
            vals = flat[mask, gi]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                forecasts[key] = None
                continue
            t = to_celsius(float(np.max(vals)))
            forecasts[key] = {"perceived_temp_c": round(t, 1), **classify(t)}
        out.append({"name": row["name"], "lat": row["lat"],
                    "lon": row["lon"], "forecasts": forecasts})

    log.info("%d municipalities processed", len(out))
    return out

# ---------------------------------------------------------------------------
# README table  — same style as NRW Ozone project
# ---------------------------------------------------------------------------
def build_table(results: list[dict], dates: dict,
                run_dt: datetime.datetime) -> str:

    top = sorted(
        results,
        key=lambda r: -(r["forecasts"].get("today") or {}).get("perceived_temp_c", -999)
    )[:TOP_N]

    def fmt_temp(d):
        if d is None:
            return "–"
        return f"{d['perceived_temp_c']:.1f} °C"

    run_str  = run_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = dates["today"].strftime("%Y-%m-%d")
    d_today  = dates["today"].strftime("%Y-%m-%d")
    d_tom    = dates["tomorrow"].strftime("%Y-%m-%d")
    d_dat    = dates["day_after_tomorrow"].strftime("%Y-%m-%d")

    lines = [
        "<!-- THERMAL_TABLE_START -->",
        "",
        f"## Top 10 — Highest Perceived Temperatures Today ({date_str})",
        "",
        f"*Forecast base: {run_dt.strftime('%Y-%m-%d %H:%M')} UTC · Generated: {run_str}*",
        "",
        f"|   | Municipality | Today ({d_today}) | Tomorrow ({d_tom}) | Day after ({d_dat}) | Health risk |",
        "|:---:|:---|:---|:---|:---|:---|",
    ]

    for i, r in enumerate(top, 1):
        fh = r["forecasts"].get("today")
        fm = r["forecasts"].get("tomorrow")
        fu = r["forecasts"].get("day_after_tomorrow")

        def cell(d):
            if d is None:
                return "–"
            b = badge(d["bg_color"])
            return f"{b} **{d['perceived_temp_c']:.1f} °C** · {d['sensation']}"

        risk = fh["risk"] if fh else "–"
        b_today = badge(fh["bg_color"]) if fh else ""
        lines.append(
            f"| {b_today} | **{r['name']}** | {cell(fh)} | {cell(fm)} | {cell(fu)} | {risk} |"
        )

    lines += [
        "",
        "### Colour scale",
        "",
        "| Colour | Perceived temperature | Thermal sensation | Health risk |",
        "|:------:|----------------------|-------------------|-------------|",
    ]
    for sensation, risk, hex_c, pt_range in LEGEND_GROUPS:
        lines.append(f"| {badge(hex_c)} | {pt_range} | {sensation} | {risk} |")
    lines += ["", "<!-- THERMAL_TABLE_END -->"]
    return "\n".join(lines)

def update_readme(table: str) -> None:
    START = "<!-- THERMAL_TABLE_START -->"
    END   = "<!-- THERMAL_TABLE_END -->"
    src   = TEMPLATE_FILE if TEMPLATE_FILE.exists() else README_FILE
    base  = src.read_text("utf-8") if src.exists() else "# NRW Thermal Risk Index\n\n"
    if START in base and END in base:
        content = base[:base.index(START)] + table + base[base.index(END) + len(END):]
    else:
        content = base.rstrip("\n") + "\n\n" + table + "\n"
    README_FILE.write_text(content, "utf-8")
    log.info("README.md updated")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def export_geojson(results: list[dict], dates: dict) -> None:
    """
    Merge forecast data into municipality polygon GeoJSON via spatial join.

    Uses a spatial join (centroid-in-polygon) instead of name matching to avoid
    mismatches between CSV names and GeoJSON property names (GEN, NAME, etc.).
    """
    import json as _json
    import geopandas as gpd
    from shapely.geometry import Point

    if not GEOJSON_SRC.exists():
        log.warning("municipality_nrw.geojson not found – skipping GeoJSON export")
        return

    # Load polygon GeoJSON
    poly_gdf = gpd.read_file(str(GEOJSON_SRC))
    if poly_gdf.crs is None:
        poly_gdf = poly_gdf.set_crs("EPSG:4326")
    else:
        poly_gdf = poly_gdf.to_crs("EPSG:4326")
    log.info("GeoJSON properties: %s", list(poly_gdf.columns))

    # Build point GeoDataFrame from CSV centroids + forecast results
    rows = []
    for r in results:
        today = (r["forecasts"].get("today") or {})
        rows.append({
            "csv_name":        r["name"],
            "perceived_temp_c": today.get("perceived_temp_c"),
            "sensation":        today.get("sensation"),
            "risk":             today.get("risk"),
            "bg_color":         today.get("bg_color"),
            "forecast_today":           _json.dumps(r["forecasts"].get("today"),   ensure_ascii=False),
            "forecast_tomorrow":        _json.dumps(r["forecasts"].get("tomorrow"), ensure_ascii=False),
            "forecast_day_after":       _json.dumps(r["forecasts"].get("day_after_tomorrow"), ensure_ascii=False),
            "geometry": Point(r["lon"], r["lat"]),
        })
    pts_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Spatial join: each centroid point → enclosing polygon
    joined = gpd.sjoin(poly_gdf, pts_gdf, how="left", predicate="contains")

    # Count matches
    matched = joined["perceived_temp_c"].notna().sum()
    log.info("Spatial join: %d/%d polygons matched to a forecast point",
             matched, len(joined))
    if matched == 0:
        # Fallback: nearest join (handles edge cases where centroid is just outside polygon)
        log.warning("No matches via 'contains' – trying nearest join")
        joined = gpd.sjoin_nearest(poly_gdf, pts_gdf, how="left", max_distance=5000)
        matched = joined["perceived_temp_c"].notna().sum()
        log.info("Nearest join: %d/%d matched", matched, len(joined))

    # Drop columns that are not JSON-serializable (e.g. Timestamps from sjoin)
    for col in list(joined.columns):
        if col == "geometry":
            continue
        try:
            import json as _test_json
            _test_json.dumps(joined[col].iloc[0] if len(joined) else None,
                             default=str)
        except Exception:
            pass
        # Convert any datetime/Timestamp columns to ISO strings
        if hasattr(joined[col], "dt") or str(joined[col].dtype).startswith(
                ("datetime", "timedelta")):
            joined[col] = joined[col].astype(str)

    # Also drop the sjoin index column which can cause issues
    for drop_col in ("index_right", "index_left"):
        if drop_col in joined.columns:
            joined = joined.drop(columns=[drop_col])

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    # Use fiona/pyogrio writer to avoid JSON serialisation issues with mixed types
    joined.to_file(str(OUTPUT_GEOJSON), driver="GeoJSON")
    log.info("GeoJSON saved: %s  (%d features, %d with forecast data)",
             OUTPUT_GEOJSON, len(joined), matched)


def main() -> None:
    try:
        url = get_latest_gft_url()
    except Exception as e:
        log.error("URL discovery failed: %s", e); sys.exit(1)

    run_dt, valid_dt = parse_filename(url)
    log.info("Model run:      %s UTC", run_dt)
    log.info("Validity start: %s UTC", valid_dt)

    ref = (run_dt or datetime.datetime.now(datetime.timezone.utc)).date()
    dates = {
        "today":              ref,
        "tomorrow":           ref + datetime.timedelta(days=1),
        "day_after_tomorrow": ref + datetime.timedelta(days=2),
    }
    log.info("Forecast dates: %s", dates)

    tmp = Path(tempfile.mktemp(suffix=".grib2"))
    try:
        download(url, tmp)
        log.info("Opening GRIB2 file …")
        ds = open_grib(tmp)
        log.info("Variables: %s  |  Coordinates: %s",
                 list(ds.data_vars), list(ds.coords))

        if not CSV_PATH.exists():
            log.error("Municipality CSV not found: %s", CSV_PATH); sys.exit(1)
        df = pd.read_csv(CSV_PATH)
        log.info("%d municipalities loaded", len(df))

        results = process(ds, df, dates)

        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps({
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_run_utc":    run_dt.isoformat() if run_dt else None,
            "valid_start_utc":  valid_dt.isoformat() if valid_dt else None,
            "forecast_dates":   {k: v.isoformat() for k, v in dates.items()},
            "data_source":      "DWD OpenData – Health Forecasts (ICON-EU-Nest, GRIB2)",
            "classification":   "VDI 3787 Part 2 / DWD Klima-Michel model",
            "municipalities":   results,
        }, indent=2, ensure_ascii=False), "utf-8")
        log.info("JSON saved: %s", OUTPUT_JSON)

        # GeoJSON export (municipality polygons + forecast attributes)
        export_geojson(results, dates)

        # Map rendering
        try:
            from generate_map import render_map
            date_str = dates["today"].strftime("%Y-%m-%d")
            run_str  = (run_dt or datetime.datetime.now(datetime.timezone.utc)).isoformat()
            render_map(date_str, run_str)
        except Exception as e:
            import traceback
            log.error("Map rendering failed:\n%s", traceback.format_exc())
            sys.exit(1)   # fail loudly so the workflow shows the error

        update_readme(build_table(
            results, dates,
            run_dt or datetime.datetime.now(datetime.timezone.utc)))
    finally:
        tmp.unlink(missing_ok=True)

    log.info("Done — %d municipalities updated", len(results))

if __name__ == "__main__":
    main()
