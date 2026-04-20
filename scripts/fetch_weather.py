"""
NRW Thermischer Gefahrenindex
Lädt aktuelle Gefühlte-Temperatur-Vorhersagen des DWD (GRIB2) und
erstellt eine Gemeindeübersicht für NRW inkl. tagesaktueller README-Tabelle.

Datenquelle : DWD OpenData – Climate & Environment – Health Forecasts
Klassifikation: VDI 3787 Blatt 2 / DWD Klima-Michel-Modell
"""

import json
import logging
import os
import re
import sys
import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pfade  (alle relativ zum Repo-Root, in dem das Skript liegt)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).parent
CSV_PATH      = REPO_ROOT / "data" / "municipality_nrw.csv"
OUTPUT_JSON   = REPO_ROOT / "output" / "weather_data.json"
README_FILE   = REPO_ROOT / "README.md"
TEMPLATE_FILE = REPO_ROOT / "README_template.md"   # optionaler statischer Header

BASE_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/"
TOP_N    = 10   # Zeilen in der README-Tabelle

# ---------------------------------------------------------------------------
# DWD-Klassifikation  (VDI 3787 / Klima-Michel-Modell, 9 Stufen)
# ---------------------------------------------------------------------------
THERMAL_CLASSES = [
    # (GT-Obergrenze °C, Empfinden, Gefährdung, Hintergrund-Hex, Text-Hex)
    (-40, "Extremer Kältestress",    "Sehr hoch", "#08306b", "#ffffff"),
    (-27, "Starker Kältestress",     "Hoch",      "#2171b5", "#ffffff"),
    (-13, "Mäßiger Kältestress",     "Erhöht",    "#6baed6", "#000000"),
    (  0, "Schwacher Kältestress",   "Gering",    "#c6dbef", "#000000"),
    ( 20, "Kein Stress (Behagl.)",   "Keine",     "#41ab5d", "#000000"),
    ( 26, "Leichte Wärmebelastung",  "Gering",    "#ffffb2", "#000000"),
    ( 32, "Mäßige Wärmebelastung",   "Erhöht",    "#fd8d3c", "#000000"),
    ( 38, "Starke Wärmebelastung",   "Hoch",      "#e31a1c", "#ffffff"),
    (999, "Extreme Wärmebelastung",  "Sehr hoch", "#800026", "#ffffff"),
]

def classify(temp_c: float) -> dict:
    for upper, empfinden, gefaehrdung, bg, fg in THERMAL_CLASSES:
        if temp_c <= upper:
            return {"empfinden": empfinden, "gefaehrdung": gefaehrdung,
                    "bg_color": bg, "fg_color": fg}
    return {"empfinden": "Unbekannt", "gefaehrdung": "–",
            "bg_color": "#cccccc", "fg_color": "#000000"}

def temp_icon(t: float) -> str:
    if t >= 38: return "🟣"
    if t >= 32: return "🔴"
    if t >= 26: return "🟠"
    if t >= 20: return "🟡"
    if t >=  0: return "🟢"
    if t >= -13: return "🔵"
    return "🔷"

# ---------------------------------------------------------------------------
# DWD-Verzeichnis  →  neueste GFT-URL
# ---------------------------------------------------------------------------
def get_latest_gft_url() -> str:
    log.info("Suche GFT-Dateien unter %s", BASE_URL)
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
    # Punkt vor Erweiterung korrekt escapen
    files = re.findall(r'href="([^"]*icreu_gft[^"]*\.(?:bin|grib2))"', r.text)
    if not files:
        raise RuntimeError("Keine GFT-Dateien im DWD-Verzeichnis gefunden.")
    files.sort()
    url = BASE_URL + files[-1]
    log.info("Neueste Datei: %s", files[-1])
    return url

# ---------------------------------------------------------------------------
# Dateinamen parsen  →  Lauf- und Gültigkeitsdatum
# ---------------------------------------------------------------------------
def parse_filename(url: str):
    fname = url.split("/")[-1]
    run_dt, valid_dt = None, None

    m = re.search(r"EDZW_(\d{14})", fname)
    if m:
        run_dt = datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=datetime.timezone.utc)

    m = re.search(r"_(\d{10})_HPC", fname)
    if m:
        valid_dt = datetime.datetime.strptime("20" + m.group(1), "%Y%m%d%H%M").replace(
            tzinfo=datetime.timezone.utc)

    return run_dt, valid_dt

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(url: str, dest: Path) -> None:
    log.info("Download läuft …")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        dest.write_bytes(b"".join(r.iter_content(65536)))
    log.info("  → %.1f MB gespeichert", dest.stat().st_size / 1e6)

# ---------------------------------------------------------------------------
# GRIB2 öffnen
# ---------------------------------------------------------------------------
def open_grib(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(str(path), engine="cfgrib",
                               backend_kwargs={"indexpath": ""})
    except Exception as e:
        log.warning("open_dataset fehlgeschlagen (%s) – versuche open_datasets", e)

    import cfgrib
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not datasets:
        raise RuntimeError("GRIB2-Datei konnte nicht gelesen werden.")
    log.info("open_datasets: %d Message(s) gefunden", len(datasets))
    return datasets[0]

def find_var(ds: xr.Dataset) -> str:
    for c in ("t2m", "2t", "pt", "PT", "perceived_temperature"):
        if c in ds.data_vars:
            return c
    first = list(ds.data_vars)[0]
    log.warning("Keine Standardvariable gefunden – nehme '%s'. Vorhanden: %s",
                first, list(ds.data_vars))
    return first

# ---------------------------------------------------------------------------
# Vorhersage je Gemeinde
# ---------------------------------------------------------------------------
def to_celsius(v: float) -> float:
    return v - 273.15 if v > 100 else v

def day_max(point: xr.Dataset, var: str, target: datetime.date):
    """Tagesmaximum der GT für ein bestimmtes Datum; None wenn keine Daten."""
    tc = next((c for c in ("valid_time", "time") if c in point.coords), None)
    if tc is None:
        return None

    times = pd.to_datetime(point[tc].values)
    arr   = point[var].values

    # Skalarer Fall (ein einziger Zeitschritt)
    if times.ndim == 0:
        if pd.Timestamp(times).date() == target:
            v = float(arr) if arr.ndim == 0 else float(arr.flat[0])
            return to_celsius(v) if not np.isnan(v) else None
        return None

    mask = np.array([t.date() == target for t in times])
    vals = arr[mask]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    return to_celsius(float(np.max(vals)))

def process(ds: xr.Dataset, df: pd.DataFrame,
            dates: dict[str, datetime.date]) -> list[dict]:
    var = find_var(ds)
    log.info("Verwende Variable '%s'", var)
    out = []
    for _, row in df.iterrows():
        try:
            pt = ds.sel(latitude=row["lat"], longitude=row["lon"], method="nearest")
        except Exception as e:
            log.warning("Interpolation für %s: %s", row["name"], e)
            continue

        forecasts = {}
        for key, d in dates.items():
            t = day_max(pt, var, d)
            forecasts[key] = ({"temp_c": round(t, 1), **classify(t)} if t is not None else None)

        out.append({"name": row["name"], "lat": row["lat"],
                    "lon": row["lon"], "forecasts": forecasts})
    log.info("%d Gemeinden verarbeitet", len(out))
    return out

# ---------------------------------------------------------------------------
# README-Tabelle
# ---------------------------------------------------------------------------
LEGEND = [
    ("🔷", "≤ −40 °C",       "Extremer Kältestress",   "Sehr hoch"),
    ("🔵", "−40 bis −27 °C", "Starker Kältestress",    "Hoch"),
    ("🔵", "−27 bis −13 °C", "Mäßiger Kältestress",    "Erhöht"),
    ("🔵", "−13 bis 0 °C",   "Schwacher Kältestress",  "Gering"),
    ("🟢", "0 bis +20 °C",   "Kein Stress (Behaglichkeit)", "Keine"),
    ("🟡", "+20 bis +26 °C", "Leichte Wärmebelastung", "Gering"),
    ("🟠", "+26 bis +32 °C", "Mäßige Wärmebelastung",  "Erhöht"),
    ("🔴", "+32 bis +38 °C", "Starke Wärmebelastung",  "Hoch"),
    ("🟣", "≥ +38 °C",       "Extreme Wärmebelastung", "Sehr hoch"),
]

def build_table(results: list[dict], dates: dict, run_dt: datetime.datetime) -> str:
    def key_fn(r):
        f = r["forecasts"].get("heute")
        return -(f["temp_c"] if f else -999)

    top = sorted(results, key=key_fn)[:TOP_N]

    def fmt(d):
        if d is None:
            return "–"
        return f"{temp_icon(d['temp_c'])} {d['temp_c']:.1f} °C · {d['empfinden']}"

    ts = run_dt.strftime("%d.%m.%Y %H:%M")
    d_h = dates["heute"].strftime("%d.%m.%Y")
    d_m = dates["morgen"].strftime("%d.%m.%Y")
    d_u = dates["uebermorgen"].strftime("%d.%m.%Y")

    lines = [
        "<!-- THERMAL_TABLE_START -->",
        "",
        f"> **Stand:** {ts} UTC &nbsp;·&nbsp; Quelle: DWD OpenData &nbsp;·&nbsp; Klima-Michel-Modell (VDI 3787)",
        "",
        "## 🌡️ Top 10 – Höchste gefühlte Temperaturen in NRW",
        "",
        f"| # | Gemeinde | Heute ({d_h}) | Morgen ({d_m}) | Übermorgen ({d_u}) | Gefährdung |",
        "|---|----------|--------------|----------------|-------------------|------------|",
    ]
    for i, r in enumerate(top, 1):
        fh = r["forecasts"].get("heute")
        fm = r["forecasts"].get("morgen")
        fu = r["forecasts"].get("uebermorgen")
        gef = fh["gefaehrdung"] if fh else "–"
        lines.append(f"| {i} | **{r['name']}** | {fmt(fh)} | {fmt(fm)} | {fmt(fu)} | {gef} |")

    lines += [
        "",
        "### Farbskala / Legende",
        "",
        "| Symbol | Gefühlte Temperatur | Thermisches Empfinden | Gesundheitliche Gefährdung |",
        "|--------|---------------------|----------------------|---------------------------|",
    ]
    for row in LEGEND:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines += ["", "<!-- THERMAL_TABLE_END -->"]
    return "\n".join(lines)

def update_readme(table: str) -> None:
    START = "<!-- THERMAL_TABLE_START -->"
    END   = "<!-- THERMAL_TABLE_END -->"

    if TEMPLATE_FILE.exists():
        base = TEMPLATE_FILE.read_text("utf-8")
    elif README_FILE.exists():
        base = README_FILE.read_text("utf-8")
    else:
        base = "# NRW Thermischer Gefahrenindex\n\n"

    if START in base and END in base:
        before = base[: base.index(START)]
        after  = base[base.index(END) + len(END):]
        content = before + table + after
    else:
        content = base.rstrip("\n") + "\n\n" + table + "\n"

    README_FILE.write_text(content, "utf-8")
    log.info("README.md aktualisiert")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # 1 – URL ermitteln
    try:
        url = get_latest_gft_url()
    except Exception as e:
        log.error("URL-Ermittlung: %s", e)
        sys.exit(1)

    run_dt, valid_dt = parse_filename(url)
    log.info("Modell-Lauf: %s UTC", run_dt)
    log.info("Gültigkeitsstart: %s UTC", valid_dt)

    ref = (run_dt or datetime.datetime.now(datetime.timezone.utc)).date()
    dates = {
        "heute":       ref,
        "morgen":      ref + datetime.timedelta(days=1),
        "uebermorgen": ref + datetime.timedelta(days=2),
    }
    log.info("Vorhersagedaten: %s", dates)

    # 2 – Download
    tmp = Path(tempfile.mktemp(suffix=".grib2"))
    try:
        download(url, tmp)

        # 3 – GRIB2 öffnen
        ds = open_grib(tmp)
        log.info("Variablen: %s  |  Koordinaten: %s",
                 list(ds.data_vars), list(ds.coords))

        # 4 – Gemeinden laden
        if not CSV_PATH.exists():
            log.error("CSV nicht gefunden: %s", CSV_PATH)
            sys.exit(1)
        df = pd.read_csv(CSV_PATH)
        log.info("%d Gemeinden aus CSV geladen", len(df))

        # 5 – Vorhersage berechnen
        results = process(ds, df, dates)

        # 6 – JSON speichern
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_run":      run_dt.isoformat() if run_dt else None,
            "valid_start":    valid_dt.isoformat() if valid_dt else None,
            "forecast_dates": {k: v.isoformat() for k, v in dates.items()},
            "municipalities": results,
        }
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        log.info("JSON gespeichert: %s", OUTPUT_JSON)

        # 7 – README aktualisieren
        table = build_table(results, dates,
                            run_dt or datetime.datetime.now(datetime.timezone.utc))
        update_readme(table)

    finally:
        tmp.unlink(missing_ok=True)

    log.info("Fertig – %d Gemeinden aktualisiert", len(results))

if __name__ == "__main__":
    main()
