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
    print("Scanning DWD directory...")
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
        print(f"Downloading {target_url}...")
        r = requests.get(target_url, stream=True)
        r.raise_for_status()
        with open("temp.grib2", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # 2. Open Data (No slicing to prevent empty datasets)
    print("Opening dataset...")
    try:
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", backend_kwargs={'indexpath': ''})
        # Check which time coordinate exists
        if 'step' in ds.coords:
            time_dim = 'step'
        elif 'valid_time' in ds.coords:
            time_dim = 'valid_time'
        else:
            time_dim = 'time'
        print(f"Time dimension identified: {time_dim}")
    except Exception as e:
        print(f"Processing error: {e}")
        sys.exit(1)
    
    # 3. Process Municipalities
    df_coords = pd.read_csv(CSV_PATH)
    results = []

    print(f"Processing {len(df_coords)} cities...")
    for _, row in df_coords.iterrows():
        try:
            # We select the point directly from the full dataset
            point = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
            
            # Forecast Logic based on hours (steps)
            forecasts = {}
            # 0-24h (Today), 24-48h (Tomorrow), 48-72h (Day After)
            windows = [("today", 0, 24), ("tomorrow", 24, 48), ("day_after", 48, 72)]

            for key, start, end in windows:
                if time_dim == 'step':
                    # Filter by hours from start
                    data_slice = point.where(
                        (point.step >= datetime.timedelta(hours=start)) & 
                        (point.step < datetime.timedelta(hours=end)), 
                        drop=True
                    )
                else:
                    # Fallback for absolute timestamps
                    now = datetime.datetime.now(datetime.timezone.utc)
                    t_start = now + datetime.timedelta(hours=start)
                    t_end = now + datetime.timedelta(hours=end)
                    data_slice = point.where(
                        (point[time_dim] >= t_start) & (point[time_dim] < t_end), 
                        drop=True
                    )

                if data_slice[time_dim].size > 0:
                    val = float(data_slice.PT1M.max())
                    if not pd.isna(val):
                        temp_c = val - 273.15 if val > 100 else val
                        f_info = get_thermal_info(temp_c)
                        forecasts[key] = {
                            "temp_c": round(temp_c, 1),
                            "feeling": f_info[0],
                            "hazard": f_info[1],
                            "color": f_info[2]
                        }
                    else: forecasts[key] = None
                else: forecasts[key] = None

            results.append({
                "city": row['name'],
                "lat": row['lat'], "lon": row['lon'],
                "forecasts": forecasts
            })
        except Exception:
            continue

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Done. Processed {len(results)} municipalities.")

if __name__ == "__main__":
    main()
