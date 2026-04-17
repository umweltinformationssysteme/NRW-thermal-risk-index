import pandas as pd
import xarray as xr
import requests
import datetime
import os
import sys
import re

# Configuration
BASE_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/"
CSV_PATH = "data/municipality_nrw.csv"
OUTPUT_PATH = "output/weather_data.json"

def get_thermal_info(temp_celsius):
    """Returns feeling, hazard level and hex color based on perceived temperature."""
    if temp_celsius > 38: return "very hot", "very high", "#800080"
    if temp_celsius > 32: return "hot", "high", "#FF0000"
    if temp_celsius > 26: return "warm", "medium", "#FFA500"
    if temp_celsius > 20: return "slightly warm", "low", "#FFFF00"
    return "pleasant", "none", "#008000"

def get_latest_grib_url():
    """Scans the DWD directory for the latest file."""
    response = requests.get(BASE_URL)
    response.raise_for_status()
    pattern = r'href="([^"]*icreu_gft[^"]*\.(?:bin|grib2))"'
    files = re.findall(pattern, response.text)
    if not files:
        raise Exception("No weather files found.")
    files.sort()
    return BASE_URL + files[-1]

def main():
    # 1. Download
    try:
        target_url = get_latest_grib_url()
        r = requests.get(target_url, stream=True)
        r.raise_for_status()
        with open("temp.grib2", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Download error: {e}")
        sys.exit(1)

    # 2. Open and Prepare Data
    try:
        # Load and sort to ensure slice works
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", backend_kwargs={'indexpath': ''})
        if ds.latitude[0] > ds.latitude[-1]:
            ds = ds.sortby("latitude")
        
        # Crop and load to RAM for speed
        ds_nrw = ds.sel(latitude=slice(50.2, 52.6), longitude=slice(5.7, 9.6)).load()
        
        # Identify time dimension (DWD uses 'step' for forecast hours)
        time_dim = 'step' if 'step' in ds_nrw.coords else 'time'
    except Exception as e:
        print(f"Data error: {e}")
        sys.exit(1)
    
    # 3. Process Municipalities
    df_coords = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df_coords.iterrows():
        try:
            point = ds_nrw.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
            
            # Forecast Logic: Use 'step' (hours from now) if available
            # Today: 0-24h, Tomorrow: 24-48h, Day After: 48-72h
            forecasts = {}
            time_windows = {
                "today": (0, 24),
                "tomorrow": (24, 48),
                "day_after": (48, 72)
            }

            for key, (start_h, end_h) in time_windows.items():
                # Filter by forecast step (hours)
                if time_dim == 'step':
                    # Convert start/end to timedeltas for 'step' coordinate
                    s = datetime.timedelta(hours=start_h)
                    e = datetime.timedelta(hours=end_h)
                    window_data = point.where((point.step >= s) & (point.step < e), drop=True)
                else:
                    # Fallback for absolute time
                    now = datetime.datetime.now(datetime.timezone.utc)
                    s = now + datetime.timedelta(hours=start_h)
                    e = now + datetime.timedelta(hours=end_h)
                    window_data = point.where((point.time >= s) & (point.time < e), drop=True)

                if window_data[time_dim].size > 0:
                    raw_max = float(window_data.PT1M.max())
                    if not pd.isna(raw_max):
                        temp_c = raw_max - 273.15 if raw_max > 100 else raw_max
                        feeling, hazard, color = get_thermal_info(temp_c)
                        forecasts[key] = {
                            "temp_c": round(temp_c, 1),
                            "feeling": feeling,
                            "hazard": hazard,
                            "color": color
                        }
                    else: forecasts[key] = None
                else: forecasts[key] = None

            results.append({
                "city": row['name'],
                "lat": row['lat'], "lon": row['lon'],
                "forecasts": forecasts
            })
        except: continue

    # 4. Export
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Success.")

if __name__ == "__main__":
    main()
