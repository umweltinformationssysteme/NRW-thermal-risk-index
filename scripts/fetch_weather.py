import pandas as pd
import xarray as xr
import requests
import datetime
import os
import sys

# Configuration
# Note: Using the specific URL for ICON-EU Perceived Temperature
GRIB_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/Z__C_EDZW_latest_grb02_icreu_gft.grib2"
CSV_PATH = "data/municipality_nrw.csv"
OUTPUT_PATH = "output/weather_data.json"

def get_thermal_info(temp):
    """Returns feeling and color based on perceived temperature in Celsius."""
    if temp > 38: return "very hot", "very high", "#800080"
    if temp > 32: return "hot", "high", "#FF0000"
    if temp > 26: return "warm", "medium", "#FFA500"
    if temp > 20: return "slightly warm", "low", "#FFFF00"
    return "pleasant", "none", "#008000"

def main():
    # 1. Download with stream=True to ensure the file is written correctly
    print(f"Downloading GRIB2 data from {GRIB_URL}...")
    try:
        with requests.get(GRIB_URL, stream=True) as r:
            r.raise_for_status()
            with open("temp.grib2", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Check if file is empty
        if os.path.getsize("temp.grib2") < 1000:
            print("Error: Downloaded file is too small (corrupted).")
            sys.exit(1)
            
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # 2. Open dataset
    # indexpath='' prevents cfgrib from trying to create a .idx file in the repo
    print("Processing GRIB2 data...")
    try:
        ds = xr.open_dataset(
            "temp.grib2", 
            engine="cfgrib", 
            backend_kwargs={'indexpath': ''}
        )
    except Exception as e:
        print(f"Failed to read GRIB2: {e}")
        sys.exit(1)
    
    # 3. Load municipality coordinates
    df_coords = pd.read_csv(CSV_PATH)
    
    # Define days (Today, Tomorrow, Day After)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    days = [today, today + datetime.timedelta(days=1), today + datetime.timedelta(days=2)]
    
    results = []

    for _, row in df_coords.iterrows():
        # Select nearest grid point
        point = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
        
        forecasts = {}
        for i, day_key in enumerate(["today", "tomorrow", "day_after"]):
            day_date = days[i]
            # Use .dt.date to filter for the specific day
            day_data = point.where(point.time.dt.date == day_date, drop=True)
            
            if len(day_data.time) > 0:
                # Get max perceived temperature (PT1M)
                max_temp = float(day_data.PT1M.max())
                feeling, hazard, color = get_thermal_info(max_temp)
                forecasts[day_key] = {
                    "temp": round(max_temp, 1),
                    "feeling": feeling,
                    "hazard": hazard,
                    "color": color
                }
            else:
                forecasts[day_key] = None # No data for this day yet
        
        results.append({
            "city": row['name'],
            "lat": row['lat'],
            "lon": row['lon'],
            "forecasts": forecasts
        })

    # 4. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully updated: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
