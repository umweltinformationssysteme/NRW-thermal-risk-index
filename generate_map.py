"""
NRW Thermal Risk Index – Map Generator
=======================================
Analogous to dwd_heat-health-warning-map_nrw/generate_map.py.

Key differences from heat-warning map:
- Reads output/thermal_index_nrw.geojson (municipality polygons + perceived_temp_c)
- 24-step colour scale (DWD Klima-Michel model / VDI 3787)
- No CRS reprojection of the TIFF – reproject GeoDataFrame to TIFF CRS instead

Output: output/thermal-risk-map-nrw-today.jpg  (1280 × 720 px)
"""

import io
import logging
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

log = logging.getLogger(__name__)

REPO_ROOT       = Path(__file__).parent
GEOJSON_PATH    = REPO_ROOT / "output" / "thermal_index_nrw.geojson"
BACKGROUND_TIFF = REPO_ROOT / "background.tiff"
OUTPUT_MAP      = REPO_ROOT / "output" / "thermal-risk-map-nrw-today.jpg"

IMG_W_PX   = 1280
IMG_H_PX   = 720
NRW_H_FRAC = 680 / 720   # NRW fills ~95% of height
DPI        = 100
POLY_ALPHA = 0.75

# ---------------------------------------------------------------------------
# Full 24-step colour scale from Legende_Thermischer_Gefahrenindex.CSV
# ---------------------------------------------------------------------------
COLOUR_STEPS: list[tuple[float, str]] = [
    ( -39, "#011AFD"), ( -34, "#1967FD"), ( -30, "#2F72FF"), ( -26, "#3D81FE"),
    ( -21, "#5BA0FD"), ( -17, "#6AAFFD"), ( -13, "#81BEFD"),
    (  -8, "#A5DAFA"), (  -4, "#B4E6FF"), (   0, "#CAEFFF"),
    (   4, "#8DFF8D"), (   8, "#01FE03"), (  12, "#00E700"),
    (  16, "#88FF02"), (  20, "#C8FF2F"),
    (  23, "#FFFF7D"), (  26, "#FEE362"),
    (  29, "#FFAF34"), (  32, "#FD7D1A"),
    (  35, "#FF3001"), (  38, "#E11902"),
    (  41, "#AD1AE4"), (  44, "#E04BFF"),
    ( 999, "#FD7EFF"),
]

# Compact 9-group legend
LEGEND_GROUPS: list[tuple[str, str, str, str]] = [
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


def classify_colour(temp_c) -> str:
    try:
        if temp_c is None:
            return "#CCCCCC"
        v = float(temp_c)
        if np.isnan(v):
            return "#CCCCCC"
        for upper, hex_c in COLOUR_STEPS:
            if v <= upper:
                return hex_c
        return "#FD7EFF"
    except (TypeError, ValueError):
        return "#CCCCCC"


def hex_to_rgba(hex_c: str, alpha: float) -> tuple:
    h = hex_c.lstrip("#")
    return int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, alpha


def compute_map_extent(gdf: gpd.GeoDataFrame):
    """Centre NRW in the frame with NRW_H_FRAC of height used."""
    b     = gdf.total_bounds          # minx, miny, maxx, maxy
    map_h = (b[3] - b[1]) / NRW_H_FRAC
    map_w = map_h * (IMG_W_PX / IMG_H_PX)
    cx    = (b[0] + b[2]) / 2
    cy    = (b[1] + b[3]) / 2
    return (cx - map_w/2, cx + map_w/2), (cy - map_h/2, cy + map_h/2)


def render_map(date_str: str, run_str: str) -> None:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"GeoJSON not found: {GEOJSON_PATH}\n"
            "Run fetch_weather.py first.")

    # ── Load & check GeoJSON ────────────────────────────────────────────────
    log.info("Loading GeoJSON: %s", GEOJSON_PATH)
    gdf = gpd.read_file(str(GEOJSON_PATH))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    if "perceived_temp_c" not in gdf.columns:
        raise ValueError("'perceived_temp_c' column missing from GeoJSON.")

    missing = gdf["perceived_temp_c"].isna().sum()
    if missing == len(gdf):
        raise ValueError(f"ALL {missing} municipalities have perceived_temp_c=null.")
    elif missing:
        log.warning("%d/%d municipalities have no forecast data (grey)", missing, len(gdf))
    else:
        log.info("All %d municipalities have forecast data", len(gdf))

    # Assign fill colour
    gdf["fill_hex"] = gdf["perceived_temp_c"].apply(classify_colour)

    # ── Open TIFF – reproject GDF to TIFF CRS (not the other way around) ───
    if not BACKGROUND_TIFF.exists():
        raise FileNotFoundError(f"background.tiff not found: {BACKGROUND_TIFF}")
    os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
    with rasterio.open(str(BACKGROUND_TIFF)) as src:
        tiff_crs    = src.crs
        tiff_bounds = src.bounds
        tiff_data   = src.read()
        log.info("TIFF: CRS=%s  size=%dx%d  bands=%d",
                 tiff_crs, src.width, src.height, src.count)

    # Reproject GDF to TIFF CRS (same approach as heat-warning map)
    gdf_proj = gdf.to_crs(tiff_crs)
    xlim, ylim = compute_map_extent(gdf_proj)

    # ── Build RGB from TIFF (handles uint8, uint16, float) ─────────────────
    n   = tiff_data.shape[0]
    rgb = np.stack([tiff_data[i] for i in range(min(3, n))]
                   if n >= 3 else [tiff_data[0]] * 3, axis=-1)
    if   rgb.dtype == np.uint16: rgb = (rgb / 65535.0).clip(0, 1)
    elif rgb.dtype == np.uint8:  rgb = (rgb / 255.0  ).clip(0, 1)
    else:
        lo, hi = rgb.min(), rgb.max()
        rgb = ((rgb - lo) / (hi - lo + 1e-9)).clip(0, 1)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(IMG_W_PX/DPI, IMG_H_PX/DPI), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()

    # Background satellite image
    ax.imshow(rgb,
              extent=[tiff_bounds.left, tiff_bounds.right,
                      tiff_bounds.bottom, tiff_bounds.top],
              origin="upper", aspect="auto", interpolation="bilinear")

    # Municipality polygons
    log.info("Drawing %d polygons …", len(gdf_proj))
    for _, row in gdf_proj.iterrows():
        color = hex_to_rgba(row["fill_hex"], POLY_ALPHA)
        gpd.GeoDataFrame([row], crs=gdf_proj.crs).plot(
            ax=ax, color=[color], edgecolor="#44444466", linewidth=0.3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ── Legend ──────────────────────────────────────────────────────────────
    # Count municipalities per sensation group
    def get_group(hex_c):
        for sensation, risk, rep_hex, _ in LEGEND_GROUPS:
            if hex_c == rep_hex:
                return sensation
        return None

    # Build legend handles – only include groups that appear in the data
    active_sensations = set(gdf_proj["fill_hex"].map(
        lambda h: next((s for s, _, rh, _ in LEGEND_GROUPS if rh == h), None)
    ).dropna())

    handles = []
    for sensation, risk, hex_c, pt_range in LEGEND_GROUPS:
        cnt = (gdf_proj["fill_hex"] == hex_c).sum()
        # Include row even if count is 0 for completeness
        rgba = hex_to_rgba(hex_c, 1.0)
        handles.append(mpatches.Patch(
            facecolor=rgba[:3], edgecolor="#888",
            label=f"{sensation}  {pt_range}  ({cnt})"
        ))

    # Legend positioned at bottom-right like heat-warning map
    LEGEND_RIGHT_PX  = 1260
    LEGEND_BOTTOM_PX = 10
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(LEGEND_RIGHT_PX / IMG_W_PX, LEGEND_BOTTOM_PX / IMG_H_PX),
        bbox_transform=ax.transAxes,
        fontsize=5.5,
        framealpha=0.88, edgecolor="#bbbbbb", facecolor="#ffffff",
        handlelength=1.2, handleheight=1.0,
        borderpad=0.6, labelspacing=0.3,
        title=f"Perceived Temperature · NRW\n{date_str}",
        title_fontsize=6.5,
    )
    leg.get_title().set_fontweight("bold")

    # Source note
    ax.text(0.01, 0.01,
            "Data: Deutscher Wetterdienst (DWD), CC BY 4.0  |  "
            "Background: Sentinel-2  |  Boundaries: BKG",
            transform=ax.transAxes, fontsize=5, color="white", alpha=0.9,
            va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"))

    # ── Save PNG → JPEG via Pillow ───────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((IMG_W_PX, IMG_H_PX), Image.LANCZOS)
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUTPUT_MAP), format="JPEG", quality=88, optimize=True)
    log.info("Map saved: %s  (%dx%d px,  %.0f KB)",
             OUTPUT_MAP, img.size[0], img.size[1],
             OUTPUT_MAP.stat().st_size / 1024)


if __name__ == "__main__":
    import datetime
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    today   = datetime.date.today().strftime("%Y-%m-%d")
    run_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    render_map(today, run_str)
