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

def get_thermal_info(temp):
    """Returns feeling, hazard level and hex color based on perceived temperature."""
    if temp > 38: return "very hot", "very high", "#800080"
    if temp > 32: return "hot", "high", "#FF0000"
    if temp > 26: return "warm", "medium", "#FFA500"
    if temp > 20: return "slightly warm", "low", "#FFFF00"
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
        print(f"Latest file identified: {target_url}")
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
    print("Opening dataset...")
    try:
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # Determine the time coordinate name (could be 'time', 'valid_time' or 'step')
        # In GFT files it is usually 'time' or 'valid_time'
        time_var = 'time' if 'time' in ds.coords else 'valid_time'
        print(f"Using time coordinate: {time_var}")
        
    except Exception as e:
        print(f"Failed to parse data: {e}")
        sys.exit(1)
    
    # 3. Process municipalities
    df_coords = pd.read_csv(CSV_PATH)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    forecast_days = [today, today + datetime.timedelta(days=1), today + datetime.timedelta(days=2)]
    
    results = []
    for _, row in df_coords.iterrows():
        # Select the grid point nearest to the municipality
        # .sel results in a data structure where we need to check existence differently
        point = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
        
        daily_forecasts = {}
        for i, key in enumerate(["today", "tomorrow", "day_after"]):
            target_date = forecast_days[i]
            
            # Filter by date
            day_data = point.where(point[time_var].dt.date == target_date, drop=True)
            
            # Robust check if data for this day exists
            if day_data.variables and day_data[time_var].size > 0:
                # Get max value and ensure it's a float
                max_val = float(day_data.PT1M.max())
                
                # Check if max_val is NaN (can happen if no data in slice)
                if not pd.isna(max_val):
                    feeling, hazard, color = get_thermal_info(max_val)
                    daily_forecasts[key] = {
                        "temp": round(max_val, 1),
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

    # 4. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully processed {len(results)} municipalities.")

if __name__ == "__main__":
    main()
