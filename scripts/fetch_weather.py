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
    """Returns feeling, hazard level and hex color based on perceived temperature in Celsius."""
    if temp_celsius > 38: return "very hot", "very high", "#800080"
    if temp_celsius > 32: return "hot", "high", "#FF0000"
    if temp_celsius > 26: return "warm", "medium", "#FFA500"
    if temp_celsius > 20: return "slightly warm", "low", "#FFFF00"
    return "pleasant", "none", "#008000"

def get_latest_grib_url():
    """Scans the DWD directory to find the most recent GFT (Perceived Temp) file."""
    print(f"Scanning {BASE_URL} for the latest forecast file...")
    response = requests.get(BASE_URL)
    response.raise_for_status()
    pattern = r'href="([^"]*icreu_gft[^"]*\.(?:bin|grib2))"'
    files = re.findall(pattern, response.text)
    if not files:
        raise Exception("No matching weather files found on DWD server.")
    files.sort()
    return BASE_URL + files[-1]

def main():
    # 1. Download
    try:
        target_url = get_latest_grib_url()
        print(f"Downloading: {target_url}")
        with requests.get(target_url, stream=True) as r:
            r.raise_for_status()
            with open("temp.grib2", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Critical Error during download: {e}")
        sys.exit(1)

    # 2. Open dataset
    print("Opening and processing dataset...")
    try:
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # FIX: Ensure latitudes are ascending for easier slicing
        if ds.latitude[0] > ds.latitude[-1]:
            ds = ds.sortby("latitude")
            
        # Performance: Crop to NRW bounding box (with a small buffer)
        ds_nrw = ds.sel(
            latitude=slice(50.2, 52.6), 
            longitude=slice(5.7, 9.6)
        ).load()
        
        time_var = 'time' if 'time' in ds_nrw.coords else 'valid_time'
        print(f"Using time coordinate: {time_var}")
    except Exception as e:
        print(f"Failed to parse data: {e}")
        sys.exit(1)
    
    # 3. Process municipalities
    df_coords = pd.read_csv(CSV_PATH)
    today_dt = datetime.datetime.now(datetime.timezone.utc).date()
    forecast_days = [today_dt, today_dt + datetime.timedelta(days=1), today_dt + datetime.timedelta(days=2)]
    
    results = []
    print(f"Processing {len(df_coords)} municipalities...")
    
    for _, row in df_coords.iterrows():
        try:
            # Select nearest point
            point = ds_nrw.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
            
            daily_forecasts = {}
            for i, key in enumerate(["today", "tomorrow", "day_after"]):
                target_date = forecast_days[i]
                day_data = point.where(point[time_var].dt.date == target_date, drop=True)
                
                if day_data[time_var].size > 0:
                    raw_max = float(day_data.PT1M.max())
                    if not pd.isna(raw_max):
                        # Kelvin to Celsius conversion
                        temp_c = raw_max - 273.15 if raw_max > 100 else raw_max
                        feeling, hazard, color = get_thermal_info(temp_c)
                        daily_forecasts[key] = {
                            "temp_c": round(temp_c, 1),
                            "feeling": feeling,
                            "hazard": hazard,
                            "color": color
                        }
                    else:
                        daily_forecasts[key] = None
                else:
                    daily_forecasts[key] = None
            
            results.append({
                "city": row['name'],
                "lat": row['lat'],
                "lon": row['lon'],
                "forecasts": daily_forecasts
            })
        except Exception as e:
            print(f"Skipping {row['name']} due to error: {e}")

    # 4. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Update successful. Processed {len(results)} municipalities.")

if __name__ == "__main__":
    main()
