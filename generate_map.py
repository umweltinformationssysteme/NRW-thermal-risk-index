"""
NRW Thermal Risk Index – Map Generator
=======================================
Renders a choropleth map of perceived temperature by municipality for NRW.

Inputs (all in repo root unless noted):
  output/thermal_index_nrw.geojson  – municipality polygons WITH forecast attrs
  background.tiff                   – Sentinel-2 georeferenced background raster

Output:
  output/thermal-risk-map-nrw-today.jpg  – 1280 × 720 px, committed daily
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PilImage

log = logging.getLogger(__name__)

REPO_ROOT       = Path(__file__).parent
# Read the OUTPUT geojson (has perceived_temp_c attached by fetch_weather.py)
GEOJSON_PATH    = REPO_ROOT / "output" / "thermal_index_nrw.geojson"
BACKGROUND_TIFF = REPO_ROOT / "background.tiff"
OUTPUT_MAP      = REPO_ROOT / "output" / "thermal-risk-map-nrw-today.jpg"

MAP_WIDTH_PX  = 1280
MAP_HEIGHT_PX = 720
DPI           = 150
POLY_ALPHA    = 0.75

# ---------------------------------------------------------------------------
# Full 24-step colour scale from the official DWD legend CSV
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

# 9-group legend  (sensation, risk, representative hex, pt range)
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
    """Map perceived temperature to hex colour. Returns grey for missing data."""
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


def hex_to_rgb(hex_c: str) -> tuple[float, float, float]:
    h = hex_c.lstrip("#")
    return int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255


def load_background(ax: plt.Axes, target_crs: str) -> bool:
    """
    Load background.tiff and display it on ax in target_crs.
    Returns True on success, False on any failure.
    Suppresses PROJ CRS warnings via GTIFF_SRS_SOURCE=EPSG.
    """
    if not BACKGROUND_TIFF.exists():
        log.warning("background.tiff not found – using dark background")
        return False
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        # Tell GDAL to use EPSG definitions, not the potentially broken GeoTIFF keys
        os.environ["GTIFF_SRS_SOURCE"] = "EPSG"

        with rasterio.open(str(BACKGROUND_TIFF)) as src:
            log.info("TIFF CRS: %s  |  size: %dx%d", src.crs, src.width, src.height)

            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            n_bands = min(src.count, 3)
            bands = []
            for b in range(1, n_bands + 1):
                dst = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, b),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                bands.append(dst)

            if len(bands) == 1:
                rgb = np.stack([bands[0]]*3, axis=-1)
            else:
                rgb = np.stack(bands[:3], axis=-1)

            # Compute extent in target CRS
            from rasterio.transform import array_bounds
            b = array_bounds(height, width, transform)  # (bottom, left, top, right) NO
            # array_bounds returns (top, left, bottom, right) in rasterio conventions
            # imshow extent: [left, right, bottom, top]
            from rasterio.crs import CRS
            extent = [b[1], b[3], b[0], b[2]]  # [left, right, bottom, top]

            ax.imshow(rgb, extent=extent, zorder=0, aspect="auto", origin="upper")
            log.info("Background TIFF loaded OK  extent=%s", [f'{v:.0f}' for v in extent])
            return True

    except Exception as e:
        log.warning("Background TIFF load failed: %s – using dark background", e)
        return False


def draw_legend(ax: plt.Axes, date_str: str, n_mun: int) -> None:
    """Draw compact 9-group legend, flush right edge like heat-warning map."""
    x0    = 0.765
    y0    = 0.97
    row_h = 0.057
    bw    = 0.028
    bh    = 0.036

    # Background panel
    panel = mpatches.FancyBboxPatch(
        (x0 - 0.012, y0 - row_h * len(LEGEND_GROUPS) - 0.072),
        0.248, row_h * len(LEGEND_GROUPS) + 0.092,
        boxstyle="round,pad=0.01", linewidth=0.5,
        edgecolor="#888888", facecolor="white", alpha=0.88,
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(panel)

    ax.text(x0 + 0.10, y0 - 0.012, "Perceived temperature",
            transform=ax.transAxes, fontsize=6.5, fontweight="bold",
            va="top", ha="center", zorder=6)
    ax.text(x0 + 0.10, y0 - 0.042,
            f"{date_str}  ·  {n_mun} municipalities",
            transform=ax.transAxes, fontsize=4.8, color="#555555",
            va="top", ha="center", zorder=6)

    for i, (sensation, risk, hex_c, pt_range) in enumerate(LEGEND_GROUPS):
        y  = y0 - 0.075 - i * row_h
        r, g, b = hex_to_rgb(hex_c)
        rect = mpatches.Rectangle(
            (x0, y - bh/2), bw, bh,
            linewidth=0.3, edgecolor="#666666",
            facecolor=(r, g, b, 0.90),
            transform=ax.transAxes, zorder=6,
        )
        ax.add_patch(rect)
        ax.text(x0 + bw + 0.009, y + 0.006,
                f"{sensation}  {pt_range}",
                transform=ax.transAxes, fontsize=5.6,
                va="center", ha="left", zorder=6)
        ax.text(x0 + bw + 0.009, y - 0.016,
                f"Risk: {risk}",
                transform=ax.transAxes, fontsize=4.4, color="#555555",
                va="center", ha="left", zorder=6)


def render_map(date_str: str, run_str: str) -> None:
    """Main entry point – produces output/thermal-risk-map-nrw-today.jpg."""
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"GeoJSON not found: {GEOJSON_PATH}\n"
            "Run fetch_weather.py first to generate output/thermal_index_nrw.geojson"
        )

    log.info("Loading GeoJSON: %s", GEOJSON_PATH)
    gdf = gpd.read_file(str(GEOJSON_PATH))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    target_crs = "EPSG:3857"
    gdf = gdf.to_crs(target_crs)

    # Assign colours from perceived_temp_c
    if "perceived_temp_c" not in gdf.columns:
        raise ValueError(
            "'perceived_temp_c' column missing from GeoJSON.\n"
            "The file must be output/thermal_index_nrw.geojson generated by fetch_weather.py."
        )
    gdf["fill_hex"] = gdf["perceived_temp_c"].apply(classify_colour)

    missing = gdf["perceived_temp_c"].isna().sum()
    if missing == len(gdf):
        raise ValueError(
            f"ALL {missing} municipalities have perceived_temp_c=null.\n"
            "Check that output/thermal_index_nrw.geojson was written by "
            "fetch_weather.py with forecast data attached."
        )
    elif missing:
        log.warning("%d/%d municipalities have no forecast data (shown grey)",
                    missing, len(gdf))
    else:
        log.info("All %d municipalities have forecast data ✓", len(gdf))

    log.info("GeoJSON: %d municipalities, CRS=%s", len(gdf), gdf.crs)

    # Figure
    fig_w = MAP_WIDTH_PX / DPI
    fig_h = MAP_HEIGHT_PX / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    bg_color = "#1a1a2e"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Set extent first (needed for imshow placement)
    bounds  = gdf.total_bounds          # [minx, miny, maxx, maxy]
    pad_x   = (bounds[2] - bounds[0]) * 0.03
    pad_y   = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_aspect("equal")

    # Background raster
    load_background(ax, target_crs)

    # Municipality polygons
    log.info("Drawing %d polygons …", len(gdf))
    for _, row in gdf.iterrows():
        r, g, b = hex_to_rgb(row["fill_hex"])
        gpd.GeoDataFrame(
            geometry=[row.geometry], crs=gdf.crs
        ).plot(
            ax=ax,
            facecolor=(r, g, b, POLY_ALPHA),
            edgecolor="#00000044",
            linewidth=0.15,
            zorder=2,
        )

    # Title
    ax.text(0.015, 0.985, "NRW Thermal Risk Index",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="white", va="top", ha="left", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_color,
                      alpha=0.75, edgecolor="none"))
    ax.text(0.015, 0.915,
            f"Max. perceived temperature · {date_str}\n"
            "Source: DWD OpenData · ICON-EU-Nest · Klima-Michel model",
            transform=ax.transAxes, fontsize=5.5, color="#cccccc",
            va="top", ha="left", zorder=7)

    draw_legend(ax, date_str, len(gdf))
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # Save PNG → convert to JPEG via Pillow (quality control)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    img = PilImage.open(buf).convert("RGB")
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUTPUT_MAP), "JPEG", quality=92, optimize=True)
    log.info("Map saved: %s  (%.0f KB)", OUTPUT_MAP, OUTPUT_MAP.stat().st_size / 1024)


if __name__ == "__main__":
    import datetime, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    today   = datetime.date.today().strftime("%Y-%m-%d")
    run_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    render_map(today, run_str)
