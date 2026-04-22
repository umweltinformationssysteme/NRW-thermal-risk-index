# NRW Thermal Risk Index

Daily 3-day perceived temperature forecast for all municipalities in North Rhine-Westphalia (NRW), Germany.  
Data source: [DWD OpenData – Health Forecasts](https://opendata.dwd.de/climate_environment/health/forecasts/) — Klima-Michel model, ICON-EU-Nest, variable `PT1M`.

---

<!-- THERMAL_TABLE_START -->
<!-- THERMAL_TABLE_END -->

---

## What it does

A GitHub Actions workflow runs every morning at **09:00 UTC**. You can trigger a run at any time via **Actions → NRW Thermal Risk Index → Run workflow**. It:

1. Scans the DWD OpenData directory and downloads the latest perceived-temperature GRIB2 forecast file (ICON-EU-Nest, variable `PT1M`).
2. Builds a KD-Tree over the ~905,000 ICON-EU-Nest grid points and maps each of the 396 municipality centroids to the nearest grid point in a single vectorised query.
3. Extracts the **daily maximum perceived temperature** for today, tomorrow and the day after tomorrow.
4. Classifies each value according to the 24-step DWD colour scale (VDI 3787 Part 2).
5. Exports a GeoJSON file with forecast attributes attached to each municipality polygon.
6. Renders a choropleth map (`output/thermal-risk-map-nrw-today.jpg`) using the municipality polygons coloured by perceived temperature, overlaid on a Sentinel-2 background.
7. Updates the Top-10 table and the map in this README.
8. Writes full results to `output/thermal_index_nrw.json` and pushes everything back to this repository.

---

## Output format

`output/thermal_index_nrw.json`

```json
{
  "generated_at_utc": "2026-04-20T09:12:34Z",
  "model_run_utc": "2026-04-20T03:22:07Z",
  "forecast_dates": {
    "today": "2026-04-20",
    "tomorrow": "2026-04-21",
    "day_after_tomorrow": "2026-04-22"
  },
  "municipalities": [
    {
      "name": "Köln",
      "lat": 50.938107,
      "lon": 6.957068,
      "forecasts": {
        "today": {
          "perceived_temp_c": 24.3,
          "sensation": "Slightly warm",
          "risk": "Low",
          "bg_color": "#FEE362"
        },
        "tomorrow": { "perceived_temp_c": 18.1, "sensation": "Comfortable", "risk": "None" },
        "day_after_tomorrow": null
      }
    }
  ]
}
```

`output/thermal_index_nrw.geojson` — same data merged into municipality polygon geometries for direct use in GIS tools or web maps.

---

## Repository structure

```
NRW-thermal-risk-index/
├── .github/
│   └── workflows/
│       └── update_thermal_index.yml      ← GitHub Actions (daily 09:00 UTC)
├── data/
│   └── municipality_nrw.csv             ← municipality centroids (396 entries)
├── output/
│   ├── thermal_index_nrw.json           ← forecast data (auto-committed daily)
│   ├── thermal_index_nrw.geojson        ← polygon forecast (auto-committed daily)
│   └── thermal-risk-map-nrw-today.jpg   ← choropleth map (auto-committed daily)
├── fetch_weather.py                     ← GRIB2 download, processing, README update
├── generate_map.py                      ← map rendering (matplotlib + rasterio)
├── municipality_nrw.geojson             ← NRW municipality polygon boundaries (BKG)
├── background.tiff                      ← Sentinel-2 georeferenced background
├── requirements.txt
├── README_template.md                   ← static sections (do not edit the block above)
└── README.md                            ← auto-generated daily
```

---

## Licenses and Data Sources

### 1. Meteorological Data (Perceived Temperature)
- **Data source:** [DWD OpenData – Health Forecasts](https://opendata.dwd.de/climate_environment/health/forecasts/)
- **Model:** ICON-EU-Nest · Variable: `PT1M` (Perceived Temperature)
- **License:** [DWD Open Data License](https://www.dwd.de/EN/service/copyright/copyright_node.html)
- **Attribution:** *Contains data from Deutscher Wetterdienst (DWD), OpenData.*

### 2. Classification
The thermal classification follows the **DWD Klima-Michel model** (Staiger et al., 2012) and VDI guideline 3787 Part 2. The perceived temperature integrates air temperature, wind speed, humidity and solar radiation into a single biometeorological index.

### 3. Administrative Boundaries
- **Data source:** © GeoBasis-DE / BKG (data modified).

### 4. Satellite Background
- **Data source:** Sentinel-2 Quarterly Mosaics True Color Cloudless, via Sentinel Hub
- **License:** [Copernicus Data License](https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/TermsConditions)

---

## Notes

- The DWD GRIB2 file is updated once per day (model run ~03:00–04:00 UTC).
- All timestamps are in **UTC**.
- The ICON-EU-Nest grid resolution is approximately **2 × 2 km**.
- `perceived_temp_c` is `null` when no forecast data is available for that day.
