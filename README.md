# NRW Perceived Temperature Tracker (ICON-EU) Thermal Risk Index

This automated tool tracks the **Perceived Temperature** (Gefühlte Temperatur) for all 396 municipalities in North Rhine-Westphalia (NRW) using data from the German Weather Service (DWD).

## How it works
1. **Source:** Downloads GRIB2 data from the DWD Open Data Server (ICON-EU model).
2. **Processing:** Maps the 6.5km grid points to the specific centroids of NRW municipalities.
3. **Daily Maxima:** Calculates the peak perceived temperature for **Today**, **Tomorrow**, and **Day After**.
4. **Automation:** A GitHub Action runs every morning to update the `output/weather_data.json`.

## Hazard Classification
We use the official thermal hazard categories:

| Perceived Temp | Feeling | Hazard Level | Color Code |
| :--- | :--- | :--- | :--- |
| > 38 °C | Very Hot | **Very High** | Dark Red (#800080) |
| 32 - 38 °C | Hot | **High** | Red (#FF0000) |
| 26 - 32 °C | Warm | **Medium** | Orange (#FFA500) |
| 20 - 26 °C | Slightly Warm | **Low** | Yellow (#FFFF00) |
| 0 - 20 °C | Pleasant | **None** | Green (#008000) |

## Sample Data: Top 10 Hottest Locations (Example)
*Based on simulated peak forecast data.*

| Municipality | Peak Temp | Feeling | Hazard | Map Color |
| :--- | :--- | :--- | :--- | :--- |
| **Düsseldorf** | 38.2 °C | Very Hot | Very High | 🟣 |
| **Duisburg** | 37.9 °C | Hot | High | 🔴 |
| **Köln** | 37.5 °C | Hot | High | 🔴 |
| **Essen** | 36.8 °C | Hot | High | 🔴 |
| **Bonn** | 36.5 °C | Hot | High | 🔴 |
| **Gelsenkirchen**| 35.9 °C | Hot | High | 🔴 |
| **Münster** | 33.2 °C | Hot | High | 🔴 |
| **Aachen** | 31.5 °C | Warm | Medium | 🟠 |
| **Bielefeld** | 30.1 °C | Warm | Medium | 🟠 |
| **Winterberg** | 25.4 °C | Slightly Warm| Low | 🟡 |

## Setup
To run this locally:
1. Install `eccodes` on your system.
2. Run `pip install -r requirements.txt`.
3. Execute `python scripts/fetch_weather.py`.


.
├── .github/
│   └── workflows/
│       └── update_data.yml    # The automation engine
├── data/
│   └── municipality_nrw.csv   # Your uploaded CSV file
├── output/
│   └── weather_data.json      # Generated output for the dashboard
├── scripts/
│   └── fetch_weather.py       # The Python logic
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (English)
