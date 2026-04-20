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
# Paths  (all relative to the repo root where this script lives)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).parent
CSV_PATH      = REPO_ROOT / "data" / "municipality_nrw.csv"
OUTPUT_JSON   = REPO_ROOT / "output" / "thermal_index_nrw.json"
README_FILE   = REPO_ROOT / "README.md"
TEMPLATE_FILE = REPO_ROOT / "README_template.md"   # static header

BASE_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/"
TOP_N    = 10

# ---------------------------------------------------------------------------
# DWD thermal classification  (VDI 3787 / Klima-Michel model, 9 levels)
# Colour values match the official DWD map legend shown in the image.
# ---------------------------------------------------------------------------
THERMAL_CLASSES = [
    # (PT upper bound °C, thermal sensation, health risk, bg hex, text hex)
    (-39, "Very cold",      "Very high", "#08306b", "#ffffff"),
    (-26, "Cold",           "High",      "#2171b5", "#ffffff"),
    (-13, "Cool",           "Elevated",  "#6baed6", "#000000"),
    (  0, "Slightly cool",  "Low",       "#c6dbef", "#000000"),
    ( 20, "Comfortable",    "None",      "#41ab5d", "#000000"),
    ( 26, "Slightly warm",  "Low",       "#ffffb2", "#000000"),
    ( 32, "Warm",           "Elevated",  "#fd8d3c", "#000000"),
    ( 38, "Hot",            "High",      "#e31a1c", "#ffffff"),
    (999, "Very hot",       "Very high", "#800026", "#ffffff"),
]

def classify(temp_c: float) -> dict:
    for upper, sensation, risk, bg, fg in THERMAL_CLASSES:
        if temp_c <= upper:
            return {"sensation": sensation, "risk": risk,
                    "bg_color": bg, "fg_color": fg}
    return {"sensation": "Unknown", "risk": "–",
            "bg_color": "#cccccc", "fg_color": "#000000"}

def temp_icon(t: float) -> str:
    if t >= 38:  return "🟣"
    if t >= 32:  return "🔴"
    if t >= 26:  return "🟠"
    if t >= 20:  return "🟡"
    if t >=  0:  return "🟢"
    if t >= -13: return "🔵"
    return "🔷"

def to_celsius(v: float) -> float:
    """Convert Kelvin → Celsius if needed (threshold > 100)."""
    return v - 273.15 if v > 100 else v

# ---------------------------------------------------------------------------
# DWD directory scan → latest GFT file URL
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
# Filename → model run time and validity start
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
# Open GRIB2 file
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
    # PT1M is the real DWD variable name for perceived temperature
    for c in ("PT1M", "PT", "t2m", "2t", "pt", "perceived_temperature"):
        if c in ds.data_vars:
            return c
    first = list(ds.data_vars)[0]
    log.warning("No standard variable found – using '%s'. Available: %s",
                first, list(ds.data_vars))
    return first

# ---------------------------------------------------------------------------
# Vectorised municipality processing via KD-Tree
# (replaces 396× xr.sel() → ~100× faster)
# ---------------------------------------------------------------------------
def process(ds: xr.Dataset, df: pd.DataFrame,
            dates: dict[str, datetime.date]) -> list[dict]:
    """
    Build a KD-Tree over the GRIB2 grid, find the nearest grid point for every
    municipality in a single query, then extract daily maxima via NumPy slicing.
    """
    var = find_var(ds)
    log.info("Using variable '%s'", var)

    # Build KD-Tree over grid points
    grid_lat = ds.latitude.values.ravel()
    grid_lon = ds.longitude.values.ravel()
    tree = cKDTree(np.column_stack([grid_lat, grid_lon]))
    _, nn_idx = tree.query(df[["lat", "lon"]].values)
    log.info("KD-Tree: %d grid points, %d municipalities mapped",
             len(grid_lat), len(df))

    # Time axis
    tc = next((c for c in ("valid_time", "time") if c in ds.coords), None)
    if tc is None:
        log.error("No time coordinate found in dataset.")
        sys.exit(1)
    times = pd.to_datetime(ds[tc].values)
    is_scalar = times.ndim == 0

    # Flatten raw data to (time × grid)
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

    # Pre-compute date masks for all forecast days
    date_masks = {
        key: np.array([t.date() == d for t in times])
        for key, d in dates.items()
    }

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
# README table  (style matches the NRW Ozone project)
# ---------------------------------------------------------------------------
COLOUR_SCALE = [
    ("🔷", "≤ −39 °C",       "Very cold",    "Very high"),
    ("🔵", "−39 to −26 °C",  "Cold",         "High"),
    ("🔵", "−26 to −13 °C",  "Cool",         "Elevated"),
    ("🔵", "−13 to 0 °C",    "Slightly cool","Low"),
    ("🟢", "0 to +20 °C",    "Comfortable",  "None"),
    ("🟡", "+20 to +26 °C",  "Slightly warm","Low"),
    ("🟠", "+26 to +32 °C",  "Warm",         "Elevated"),
    ("🔴", "+32 to +38 °C",  "Hot",          "High"),
    ("🟣", "≥ +38 °C",       "Very hot",     "Very high"),
]

def build_table(results: list[dict], dates: dict,
                run_dt: datetime.datetime) -> str:
    top = sorted(
        results,
        key=lambda r: -(r["forecasts"].get("today") or {}).get("perceived_temp_c", -999)
    )[:TOP_N]

    def fmt(d):
        if d is None:
            return "–"
        return f"{temp_icon(d['perceived_temp_c'])} {d['perceived_temp_c']:.1f} °C"

    run_str   = run_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str  = dates["today"].strftime("%Y-%m-%d")
    d_today   = dates["today"].strftime("%Y-%m-%d")
    d_tom     = dates["tomorrow"].strftime("%Y-%m-%d")
    d_dat     = dates["day_after_tomorrow"].strftime("%Y-%m-%d")

    lines = [
        "<!-- THERMAL_TABLE_START -->",
        "",
        f"## Top 10 — Highest Perceived Temperatures Today ({date_str})",
        "",
        f"*Forecast base: {run_dt.strftime('%Y-%m-%d %H:%M')} UTC · Generated: {run_str}*",
        "",
        f"| | Municipality | Today ({d_today}) | Tomorrow ({d_tom}) | Day after ({d_dat}) | Health risk |",
        "|-|-------------|-------------------|-------------------|---------------------|-------------|",
    ]
    for i, r in enumerate(top, 1):
        fh = r["forecasts"].get("today")
        fm = r["forecasts"].get("tomorrow")
        fu = r["forecasts"].get("day_after_tomorrow")
        risk = fh["risk"] if fh else "–"
        lines.append(
            f"| {i} | **{r['name']}** | {fmt(fh)} · {fh['sensation'] if fh else '–'} "
            f"| {fmt(fm)} · {fm['sensation'] if fm else '–'} "
            f"| {fmt(fu)} · {fu['sensation'] if fu else '–'} "
            f"| {risk} |"
        )

    lines += [
        "",
        "### Colour scale",
        "",
        "| Colour | Perceived temperature | Thermal sensation | Health risk |",
        "|--------|----------------------|-------------------|-------------|",
    ]
    for row in COLOUR_SCALE:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines += ["", "---", "", "<!-- THERMAL_TABLE_END -->"]
    return "\n".join(lines)

def update_readme(table: str) -> None:
    START = "<!-- THERMAL_TABLE_START -->"
    END   = "<!-- THERMAL_TABLE_END -->"

    base = (TEMPLATE_FILE if TEMPLATE_FILE.exists() else README_FILE)
    base = base.read_text("utf-8") if base.exists() else "# NRW Thermal Risk Index\n\n"

    if START in base and END in base:
        content = base[:base.index(START)] + table + base[base.index(END) + len(END):]
    else:
        content = base.rstrip("\n") + "\n\n" + table + "\n"

    README_FILE.write_text(content, "utf-8")
    log.info("README.md updated")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        url = get_latest_gft_url()
    except Exception as e:
        log.error("URL discovery failed: %s", e)
        sys.exit(1)

    run_dt, valid_dt = parse_filename(url)
    log.info("Model run:      %s UTC", run_dt)
    log.info("Validity start: %s UTC", valid_dt)

    ref = (run_dt or datetime.datetime.now(datetime.timezone.utc)).date()
    dates = {
        "today":               ref,
        "tomorrow":            ref + datetime.timedelta(days=1),
        "day_after_tomorrow":  ref + datetime.timedelta(days=2),
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
            log.error("Municipality CSV not found: %s", CSV_PATH)
            sys.exit(1)
        df = pd.read_csv(CSV_PATH)
        log.info("%d municipalities loaded", len(df))

        results = process(ds, df, dates)

        # Save JSON
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_run_utc":    run_dt.isoformat() if run_dt else None,
            "valid_start_utc":  valid_dt.isoformat() if valid_dt else None,
            "forecast_dates":   {k: v.isoformat() for k, v in dates.items()},
            "data_source":      "DWD OpenData – Health Forecasts (ICON-EU-Nest, GRIB2)",
            "classification":   "VDI 3787 Part 2 / DWD Klima-Michel model",
            "municipalities":   results,
        }
        OUTPUT_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        log.info("JSON saved: %s", OUTPUT_JSON)

        # Update README
        update_readme(build_table(
            results, dates,
            run_dt or datetime.datetime.now(datetime.timezone.utc)))

    finally:
        tmp.unlink(missing_ok=True)

    log.info("Done — %d municipalities updated", len(results))


if __name__ == "__main__":
    main()
