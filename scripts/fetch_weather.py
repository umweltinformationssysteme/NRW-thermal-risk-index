import pandas as pd
import xarray as xr
import requests
import datetime
import os

# Configuration
GRIB_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/Z__C_EDZW_latest_grb02_icreu_gft.grib2"
CSV_PATH = "data/municipality_nrw.csv"
OUTPUT_PATH = "output/weather_data.json"

def get_thermal_info(temp):
    """Returns feeling and color based on perceived temperature."""
    if temp > 38: return "very hot", "very high", "#800080"
    if temp > 32: return "hot", "high", "#FF0000"
    if temp > 26: return "warm", "medium", "#FFA500"
    if temp > 20: return "slightly warm", "low", "#FFFF00"
    return "pleasant", "none", "#008000"

def main():
    # Download the latest GRIB2 file from DWD
    print("Downloading GRIB2 data...")
    response = requests.get(GRIB_URL)
    with open("temp.grib2", "wb") as f:
        f.write(response.content)

    # Open dataset with cfgrib
    ds = xr.open_dataset("temp.grib2", engine="cfgrib")
    
    # Load your municipality coordinates
    df_coords = pd.read_csv(CSV_PATH)
    
    # Define days (Today, Tomorrow, Day After)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    days = [today, today + datetime.timedelta(days=1), today + datetime.timedelta(days=2)]
    
    results = []

    for _, row in df_coords.iterrows():
        # Select nearest grid point from ICON-EU model
        point = ds.sel(latitude=row['lat'], longitude=row['lon'], method='nearest')
        
        forecasts = {}
        for i, day in enumerate(["today", "tomorrow", "day_after"]):
            # Filter data for the specific day and find maximum PT1M (Perceived Temp)
            day_date = days[i]
            day_data = point.where(point.time.dt.date == day_date, drop=True)
            
            if len(day_data.time) > 0:
                max_temp = float(day_data.PT1M.max())
                feeling, hazard, color = get_thermal_info(max_temp)
                forecasts[day] = {
                    "temp": round(max_temp, 1),
                    "feeling": feeling,
                    "hazard": hazard,
                    "color": color
                }
        
        results.append({
            "city": row['name'],
            "lat": row['lat'],
            "lon": row['lon'],
            "forecasts": forecasts
        })

    # Save as JSON for the dashboard frontend
    import json
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Update successful. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
