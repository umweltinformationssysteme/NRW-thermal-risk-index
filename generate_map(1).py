"""
NRW Thermal Risk Index – Map Generator
=======================================
Renders a choropleth map of perceived temperature by municipality for NRW,
analogous to the DWD Heat Warning Map (dwd_heat-health-warning-map_nrw).

Inputs:
  municipality_nrw.geojson  – NRW municipality polygons with perceived_temp_c
  background.tiff           – Sentinel-2 georeferenced background raster

Output:
  output/thermal-risk-map-nrw-today.jpg  – 1280 × 720 px, committed daily
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import numpy as np
from PIL import Image
import rasterio
from rasterio.plot import show as rasterio_show

log = logging.getLogger(__name__)

REPO_ROOT      = Path(__file__).parent
GEOJSON_PATH   = REPO_ROOT / "municipality_nrw.geojson"
BACKGROUND_TIFF= REPO_ROOT / "background.tiff"
OUTPUT_JSON    = REPO_ROOT / "output" / "thermal_index_nrw.json"
OUTPUT_MAP     = REPO_ROOT / "output" / "thermal-risk-map-nrw-today.jpg"

MAP_WIDTH_PX   = 1280
MAP_HEIGHT_PX  = 720
DPI            = 150
POLY_ALPHA     = 0.75   # polygon fill opacity (like heat-warning project: 70%)

# ---------------------------------------------------------------------------
# Full 24-step colour scale from the official DWD legend CSV
# (upper PT bound °C → hex colour)
# ---------------------------------------------------------------------------
COLOUR_STEPS: list[tuple[float, str]] = [
    ( -39, "#011AFD"),
    ( -34, "#1967FD"),
    ( -30, "#2F72FF"),
    ( -26, "#3D81FE"),
    ( -21, "#5BA0FD"),
    ( -17, "#6AAFFD"),
    ( -13, "#81BEFD"),
    (  -8, "#A5DAFA"),
    (  -4, "#B4E6FF"),
    (   0, "#CAEFFF"),
    (   4, "#8DFF8D"),
    (   8, "#01FE03"),
    (  12, "#00E700"),
    (  16, "#88FF02"),
    (  20, "#C8FF2F"),
    (  23, "#FFFF7D"),
    (  26, "#FEE362"),
    (  29, "#FFAF34"),
    (  32, "#FD7D1A"),
    (  35, "#FF3001"),
    (  38, "#E11902"),
    (  41, "#AD1AE4"),
    (  44, "#E04BFF"),
    ( 999, "#FD7EFF"),
]

# 9-group labels for the compact legend (sensation + risk + representative colour)
LEGEND_GROUPS: list[tuple[str, str, str, str]] = [
    # (sensation_en,   sensation_de,   risk_en,    hex)
    ("Very cold",    "Sehr kalt",    "Very high", "#1967FD"),
    ("Cold",         "Kalt",         "High",      "#3D81FE"),
    ("Cool",         "Kühl",         "Elevated",  "#81BEFD"),
    ("Slightly cool","Leicht kühl",  "Low",       "#B4E6FF"),
    ("Comfortable",  "Behaglich",    "None",      "#00E700"),
    ("Slightly warm","Leicht warm",  "Low",       "#FEE362"),
    ("Warm",         "Warm",         "Elevated",  "#FFAF34"),
    ("Hot",          "Heiß",         "High",      "#E11902"),
    ("Very hot",     "Sehr heiß",    "Very high", "#E04BFF"),
]


def classify_colour(temp_c: float) -> str:
    """Return the hex fill colour for a given perceived temperature."""
    for upper, hex_c in COLOUR_STEPS:
        if temp_c <= upper:
            return hex_c
    return "#FD7EFF"


def hex_to_rgba(hex_c: str, alpha: float = 1.0) -> tuple:
    r = int(hex_c[1:3], 16) / 255
    g = int(hex_c[3:5], 16) / 255
    b = int(hex_c[5:7], 16) / 255
    return (r, g, b, alpha)


def load_geodata(json_path: Path) -> gpd.GeoDataFrame:
    """Load GeoJSON, assign colours, reproject to Web-Mercator for display."""
    gdf = gpd.read_file(str(json_path))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")

    # Assign fill colour from perceived_temp_c (written by fetch_weather.py)
    if "perceived_temp_c" in gdf.columns:
        gdf["fill_hex"] = gdf["perceived_temp_c"].apply(
            lambda t: classify_colour(float(t)) if t is not None else "#CCCCCC"
        )
    else:
        log.warning("'perceived_temp_c' column missing – using neutral grey")
        gdf["fill_hex"] = "#CCCCCC"

    return gdf


def load_background(tiff_path: Path) -> tuple:
    """Open background GeoTIFF and return (rasterio dataset, extent)."""
    src = rasterio.open(str(tiff_path))
    # Reproject extent to EPSG:3857 for matching with GeoDataFrame
    from rasterio.warp import transform_bounds
    bounds_3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
    extent = [bounds_3857[0], bounds_3857[2], bounds_3857[1], bounds_3857[3]]
    return src, extent


def draw_legend(ax: plt.Axes, date_str: str, n_municipalities: int) -> None:
    """Draw compact 9-group legend, flush to the right edge (like heat-warning map)."""
    x0 = 0.765   # legend left edge in axes-fraction coordinates
    y0 = 0.97    # top
    row_h = 0.052
    box_w = 0.030
    box_h = 0.038

    # Shadow / background panel
    panel = mpatches.FancyBboxPatch(
        (x0 - 0.01, y0 - row_h * len(LEGEND_GROUPS) - 0.07),
        0.245, row_h * len(LEGEND_GROUPS) + 0.09,
        boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#888888",
        facecolor="white", alpha=0.88,
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(panel)

    # Title
    ax.text(x0 + 0.10, y0 - 0.01, "Perceived temperature",
            transform=ax.transAxes, fontsize=6.5, fontweight="bold",
            va="top", ha="center", zorder=6)
    ax.text(x0 + 0.10, y0 - 0.04, f"{date_str}  ·  {n_municipalities} municipalities",
            transform=ax.transAxes, fontsize=5.0, color="#555555",
            va="top", ha="center", zorder=6)

    for i, (sensation_en, sensation_de, risk, hex_c) in enumerate(LEGEND_GROUPS):
        y = y0 - 0.07 - i * row_h
        r, g, b, _ = hex_to_rgba(hex_c)
        rect = mpatches.Rectangle(
            (x0, y - box_h / 2), box_w, box_h,
            linewidth=0.3, edgecolor="#666666",
            facecolor=(r, g, b, 0.90),
            transform=ax.transAxes, zorder=6,
        )
        ax.add_patch(rect)
        ax.text(x0 + box_w + 0.008, y, f"{sensation_en}",
                transform=ax.transAxes, fontsize=5.8,
                va="center", ha="left", zorder=6)
        ax.text(x0 + box_w + 0.008, y - 0.020, f"Risk: {risk}",
                transform=ax.transAxes, fontsize=4.5, color="#555555",
                va="center", ha="left", zorder=6)


def render_map(date_str: str, run_str: str) -> None:
    """Main rendering function – produces output/thermal-risk-map-nrw-today.jpg."""
    log.info("Loading GeoJSON …")
    gdf = load_geodata(GEOJSON_PATH)

    fig_w = MAP_WIDTH_PX / DPI
    fig_h = MAP_HEIGHT_PX / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Background raster
    if BACKGROUND_TIFF.exists():
        log.info("Loading background TIFF …")
        try:
            src, extent = load_background(BACKGROUND_TIFF)
            from rasterio.enums import Resampling
            from rasterio.warp import calculate_default_transform, reproject

            # Reproject background to EPSG:3857
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({"crs": "EPSG:3857", "transform": transform,
                           "width": width, "height": height})

            # Read and reproject each band
            bands = []
            for band_idx in range(1, min(src.count + 1, 4)):
                band_data = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=band_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:3857",
                    resampling=Resampling.bilinear,
                )
                bands.append(band_data)

            rgb = np.stack(bands[:3], axis=-1)
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in EPSG:3857
            ax.imshow(rgb, extent=extent, zorder=0, aspect="auto")
        except Exception as e:
            log.warning("Background TIFF failed: %s – using dark background", e)
    else:
        log.warning("background.tiff not found – using dark background")

    # Set map extent to NRW bounds + small margin
    bounds = gdf.total_bounds
    margin_x = (bounds[2] - bounds[0]) * 0.03
    margin_y = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    # Draw municipality polygons
    log.info("Drawing %d municipality polygons …", len(gdf))
    for _, row in gdf.iterrows():
        hex_c = row.get("fill_hex", "#CCCCCC")
        r, g, b, _ = hex_to_rgba(hex_c)
        gdf_single = gpd.GeoDataFrame([row], geometry=[row.geometry], crs=gdf.crs)
        gdf_single.plot(
            ax=ax,
            facecolor=(r, g, b, POLY_ALPHA),
            edgecolor="#333333",
            linewidth=0.15,
            zorder=2,
        )

    # Title
    ax.text(0.02, 0.98,
            "NRW Thermal Risk Index",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="white", va="top", ha="left", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      alpha=0.7, edgecolor="none"))
    ax.text(0.02, 0.91,
            f"Perceived temperature · {date_str}\nSource: DWD OpenData (ICON-EU-Nest)",
            transform=ax.transAxes, fontsize=6, color="#cccccc",
            va="top", ha="left", zorder=7)

    # Legend
    draw_legend(ax, date_str, len(gdf))

    # Clean axes
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # Save
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_MAP), dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="jpeg", quality=92)
    plt.close(fig)
    log.info("Map saved: %s", OUTPUT_MAP)


if __name__ == "__main__":
    import datetime
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    today = datetime.date.today().strftime("%Y-%m-%d")
    render_map(today, datetime.datetime.utcnow().isoformat())
