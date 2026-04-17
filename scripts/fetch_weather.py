"""
NRW Thermischer Gefahrenindex
Lädt aktuelle Gefühlte-Temperatur-Vorhersagen des DWD (GRIB2)
und erstellt eine Gemeindeübersicht für NRW inkl. README-Tabelle.

Datenquelle: DWD OpenData – Climate & Environment – Health Forecasts
Klassifikation: VDI 3787 Blatt 2 / DWD Klima-Michel-Modell
"""

import json
import os
import re
import sys
import datetime
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
BASE_URL = "https://opendata.dwd.de/climate_environment/health/forecasts/"
CSV_PATH = Path("data/municipality_nrw.csv")
OUTPUT_JSON = Path("output/weather_data.json")
OUTPUT_README = Path("README.md")
README_TEMPLATE = Path("README_template.md")  # optionaler statischer Header
TOP_N = 10  # Anzahl Gemeinden in der README-Tabelle

# ---------------------------------------------------------------------------
# DWD-Klassifikation (VDI 3787 / Klima-Michel-Modell)
# Beide Richtungen: Kältestress + Wärmebelastung
# ---------------------------------------------------------------------------
THERMAL_CLASSES = [
    # (obere GT-Grenze °C, Empfinden, Gefährdung, Hintergrundfarbe, Textfarbe)
    (-40, "Extremer Kältestress",   "Sehr hoch", "#08306b", "#ffffff"),
    (-27, "Starker Kältestress",    "Hoch",      "#2171b5", "#ffffff"),
    (-13, "Mäßiger Kältestress",    "Erhöht",    "#6baed6", "#000000"),
    (  0, "Schwacher Kältestress",  "Gering",    "#c6dbef", "#000000"),
    ( 20, "Kein Stress (Behagl.)",  "Keine",     "#41ab5d", "#000000"),
    ( 26, "Leichte Wärmebelastung", "Gering",    "#ffffb2", "#000000"),
    ( 32, "Mäßige Wärmebelastung",  "Erhöht",    "#fd8d3c", "#000000"),
    ( 38, "Starke Wärmebelastung",  "Hoch",      "#e31a1c", "#ffffff"),
    (999, "Extreme Wärmebelastung", "Sehr hoch", "#800026", "#ffffff"),
]


def classify(temp_c: float) -> dict:
    """Gibt Empfinden, Gefährdungsstufe und Farben nach DWD/VDI zurück."""
    for upper, empfinden, gefaehrdung, bg_color, fg_color in THERMAL_CLASSES:
        if temp_c <= upper:
            return {
                "empfinden": empfinden,
                "gefaehrdung": gefaehrdung,
                "bg_color": bg_color,
                "fg_color": fg_color,
            }
    # Fallback (sollte nie eintreten)
    return {"empfinden": "Unbekannt", "gefaehrdung": "–", "bg_color": "#cccccc", "fg_color": "#000000"}


# ---------------------------------------------------------------------------
# URL-Ermittlung
# ---------------------------------------------------------------------------
def get_gft_urls(base_url: str) -> list[str]:
    """
    Durchsucht das DWD-Verzeichnis nach GFT-Dateien (Gefühlte Temperatur).
    Gibt alle gefundenen URLs zurück, sortiert (neueste zuletzt).
    """
    log.info("Suche nach GFT-Dateien unter %s", base_url)
    r = requests.get(base_url, timeout=30)
    r.raise_for_status()
    # Korrekte Regex: Punkt vor Erweiterung escapen
    pattern = re.compile(r'href="([^"]*icreu_gft[^"]*\.(?:bin|grib2))"')
    files = pattern.findall(r.text)
    if not files:
        raise RuntimeError("Keine GFT-Dateien im DWD-Verzeichnis gefunden.")
    files.sort()
    log.info("%d GFT-Datei(en) gefunden.", len(files))
    return [base_url + f for f in files]


def parse_valid_times_from_filename(filename: str) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """
    Extrahiert Laufzeit und Gültigkeitszeitpunkt aus DWD-Dateinamen.

    Format: Z__C_EDZW_{YYYYMMDDHHMMSS}_grb02+icreu_gft_icreu__{step_from}_{step_to}_{YYMMDDHHMM}_HPC.bin
    """
    run_dt = None
    valid_start = None

    m_run = re.search(r"EDZW_(\d{14})", filename)
    if m_run:
        run_dt = datetime.datetime.strptime(m_run.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=datetime.timezone.utc
        )

    # Gültigkeitsdatum: 10-stellige Zahl vor _HPC (YYMMDDHHMM)
    m_valid = re.search(r"_(\d{10})_HPC", filename)
    if m_valid:
        valid_start = datetime.datetime.strptime(
            "20" + m_valid.group(1), "%Y%m%d%H%M"
        ).replace(tzinfo=datetime.timezone.utc)

    return run_dt, valid_start


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_file(url: str, dest: Path) -> None:
    """Lädt eine Datei herunter und speichert sie unter dest."""
    log.info("Lade %s ...", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dest.write_bytes(b"".join(r.iter_content(chunk_size=65536)))
    log.info("Download abgeschlossen: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# GRIB2 verarbeiten
# ---------------------------------------------------------------------------
def open_grib(path: Path) -> xr.Dataset:
    """
    Öffnet eine GRIB2-Datei mit cfgrib.
    Falls mehrere GRIB-Messages enthalten sind, wird die erste zurückgegeben.
    """
    try:
        ds = xr.open_dataset(
            str(path),
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},  # kein .idx-File anlegen
        )
        return ds
    except Exception as e:
        log.warning("Direktes Öffnen fehlgeschlagen (%s) – versuche open_datasets.", e)

    # Fallback: alle Messages einzeln öffnen und erste nehmen
    import cfgrib
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not datasets:
        raise RuntimeError("GRIB2-Datei konnte nicht geöffnet werden.")
    log.info("open_datasets: %d Dataset(s) gefunden, nehme ersten.", len(datasets))
    return datasets[0]


def get_variable_name(ds: xr.Dataset) -> str:
    """Ermittelt den Namen der Temperaturvariablen im Dataset."""
    # DWD GFT-Dateien verwenden typischerweise 't2m', '2t', 'pt' o.ä.
    candidates = ["t2m", "2t", "pt", "PT", "perceived_temperature"]
    for c in candidates:
        if c in ds.data_vars:
            return c
    # Fallback: erste Datenvariable
    vars_ = list(ds.data_vars)
    if not vars_:
        raise RuntimeError(f"Keine Datenvariablen in Dataset: {ds}")
    log.warning("Keine bekannte Temperaturvariable gefunden, nehme '%s'. Verfügbar: %s", vars_[0], vars_)
    return vars_[0]


def kelvin_to_celsius(value: float) -> float:
    """Konvertiert Kelvin→Celsius, falls nötig (Schwelle: > 100 K)."""
    return value - 273.15 if value > 100 else value


# ---------------------------------------------------------------------------
# Vorhersage je Gemeinde berechnen
# ---------------------------------------------------------------------------
def extract_forecast_for_date(
    point: xr.Dataset, var_name: str, target_date: datetime.date
) -> dict | None:
    """
    Extrahiert Tagesmaximum der Gefühlten Temperatur für ein Datum.
    Gibt None zurück wenn keine Daten vorhanden.
    """
    # Zeitkoordinate ermitteln
    time_coord = None
    for tc in ("valid_time", "time", "step"):
        if tc in point.coords:
            time_coord = tc
            break
    if time_coord is None:
        log.warning("Keine Zeitkoordinate gefunden.")
        return None

    times = pd.to_datetime(point[time_coord].values)

    if times.ndim == 0:
        # Skalarer Zeitwert (einzelner Zeitschritt)
        if pd.Timestamp(times).date() == target_date:
            val = float(point[var_name].values)
            if np.isnan(val):
                return None
            temp_c = kelvin_to_celsius(val)
            return {"temp_c": round(temp_c, 1), **classify(temp_c)}
        return None

    mask = np.array([t.date() == target_date for t in times])
    if not mask.any():
        return None

    day_vals = point[var_name].values[mask] if point[var_name].ndim > 0 else np.array([float(point[var_name])])
    day_vals = day_vals[~np.isnan(day_vals)]
    if len(day_vals) == 0:
        return None

    # Tagesmaximum der Gefühlten Temperatur (Nachmittagswert relevant für Hitzebewertung)
    max_val = float(np.max(day_vals))
    temp_c = kelvin_to_celsius(max_val)
    return {"temp_c": round(temp_c, 1), **classify(temp_c)}


def process_municipalities(
    ds: xr.Dataset,
    df_municipalities: pd.DataFrame,
    forecast_dates: dict[str, datetime.date],
) -> list[dict]:
    """
    Interpoliert GRIB2-Gitter auf Gemeindezentroide und klassifiziert die GT.
    forecast_dates: {"heute": date, "morgen": date, "uebermorgen": date}
    """
    var_name = get_variable_name(ds)
    log.info("Temperaturvariable: '%s'", var_name)

    results = []
    for _, row in df_municipalities.iterrows():
        try:
            point = ds.sel(latitude=row["lat"], longitude=row["lon"], method="nearest")
        except Exception as e:
            log.warning("Interpolation fehlgeschlagen für %s: %s", row["name"], e)
            continue

        forecasts = {}
        for key, target_date in forecast_dates.items():
            forecasts[key] = extract_forecast_for_date(point, var_name, target_date)

        results.append(
            {
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "forecasts": forecasts,
            }
        )

    log.info("%d Gemeinden verarbeitet.", len(results))
    return results


# ---------------------------------------------------------------------------
# README-Tabelle generieren
# ---------------------------------------------------------------------------
GEFAEHRDUNG_SORT = {
    "Sehr hoch": 0,
    "Hoch": 1,
    "Erhöht": 2,
    "Gering": 3,
    "Keine": 4,
    "–": 5,
}


def make_badge(empfinden: str, gefaehrdung: str, bg: str, fg: str) -> str:
    """Erstellt ein GitHub-Markdown-kompatibles farbiges Badge via shields.io."""
    label = empfinden.replace(" ", "_")
    color = bg.lstrip("#")
    return f"![{empfinden}](https://img.shields.io/badge/{label}-{color}?style=flat-square&labelColor={color}&color={color}&fontColor={fg.lstrip('#')})"


def cell(data: dict | None, key: str = "heute") -> str:
    """Formatiert eine Tabellenzelle mit Farbe und Wert."""
    if data is None:
        return "–"
    d = data.get(key)
    if d is None:
        return "–"
    temp = d["temp_c"]
    empf = d["empfinden"]
    bg = d["bg_color"]
    fg = d["fg_color"]
    # Markdown table: kein HTML-Block erlaubt, daher Unicode-Emoji-Ampel + Wert
    icon = temp_icon(temp)
    return f"{icon} **{temp:.1f} °C**<br><sub>{empf}</sub>"


def temp_icon(temp_c: float) -> str:
    """Gibt ein passendes Unicode-Symbol zurück."""
    if temp_c >= 38: return "🟣"
    if temp_c >= 32: return "🔴"
    if temp_c >= 26: return "🟠"
    if temp_c >= 20: return "🟡"
    if temp_c >= 0:  return "🟢"
    if temp_c >= -13: return "🔵"
    return "🔷"


def build_readme_table(results: list[dict], forecast_dates: dict, run_dt: datetime.datetime) -> str:
    """Erstellt den Markdown-Block mit Top-N-Tabelle für die README."""

    # Sortierung: höchste GT heute absteigend
    def sort_key(r):
        f = r["forecasts"].get("heute")
        return -(f["temp_c"] if f else -999)

    sorted_results = sorted(results, key=sort_key)

    # Top N mit höchster GT + Top N mit niedrigster GT
    top_warm = sorted_results[:TOP_N]

    lines = []
    lines.append("<!-- THERMAL_TABLE_START -->")
    lines.append("")
    lines.append(f"> **Stand:** {run_dt.strftime('%d.%m.%Y %H:%M')} UTC · Quelle: DWD OpenData · Klima-Michel-Modell (VDI 3787)")
    lines.append("")
    lines.append("## 🌡️ Top 10 – Höchste gefühlte Temperaturen in NRW")
    lines.append("")
    lines.append("| # | Gemeinde | Heute | Morgen | Übermorgen | Gefährdung |")
    lines.append("|---|----------|-------|--------|-----------|-----------|")

    for rank, r in enumerate(top_warm, 1):
        f_h = r["forecasts"].get("heute")
        f_m = r["forecasts"].get("morgen")
        f_u = r["forecasts"].get("uebermorgen")

        def fmt(d):
            if d is None:
                return "–"
            return f"{temp_icon(d['temp_c'])} {d['temp_c']:.1f} °C · {d['empfinden']}"

        gefahr = f_h["gefaehrdung"] if f_h else "–"
        lines.append(
            f"| {rank} | **{r['name']}** | {fmt(f_h)} | {fmt(f_m)} | {fmt(f_u)} | {gefahr} |"
        )

    lines.append("")
    lines.append("### Farbskala / Legende")
    lines.append("")
    lines.append("| Farbe | Gefühlte Temperatur | Thermisches Empfinden | Gesundheitliche Gefährdung |")
    lines.append("|-------|---------------------|----------------------|---------------------------|")

    legend_rows = [
        ("🔷", "≤ −40 °C", "Extremer Kältestress", "Sehr hoch"),
        ("🔵", "−40 bis −27 °C", "Starker Kältestress", "Hoch"),
        ("🔵", "−27 bis −13 °C", "Mäßiger Kältestress", "Erhöht"),
        ("🔵", "−13 bis 0 °C", "Schwacher Kältestress", "Gering"),
        ("🟢", "0 bis +20 °C", "Kein Stress (Behaglichkeit)", "Keine"),
        ("🟡", "+20 bis +26 °C", "Leichte Wärmebelastung", "Gering"),
        ("🟠", "+26 bis +32 °C", "Mäßige Wärmebelastung", "Erhöht"),
        ("🔴", "+32 bis +38 °C", "Starke Wärmebelastung", "Hoch"),
        ("🟣", "≥ +38 °C", "Extreme Wärmebelastung", "Sehr hoch"),
    ]
    for row in legend_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines.append("")
    lines.append(f"*Vorhersage: Heute = {forecast_dates['heute'].strftime('%d.%m.%Y')}, "
                 f"Morgen = {forecast_dates['morgen'].strftime('%d.%m.%Y')}, "
                 f"Übermorgen = {forecast_dates['uebermorgen'].strftime('%d.%m.%Y')}*")
    lines.append("")
    lines.append("<!-- THERMAL_TABLE_END -->")

    return "\n".join(lines)


def update_readme(table_md: str) -> None:
    """
    Schreibt den Tabellen-Block in die README.md.
    Ersetzt den Bereich zwischen THERMAL_TABLE_START und THERMAL_TABLE_END,
    oder hängt ihn ans Ende an, wenn kein Marker vorhanden ist.
    """
    if README_TEMPLATE.exists():
        base = README_TEMPLATE.read_text(encoding="utf-8")
    elif OUTPUT_README.exists():
        base = OUTPUT_README.read_text(encoding="utf-8")
    else:
        base = "# NRW Thermischer Gefahrenindex\n\n"

    start_marker = "<!-- THERMAL_TABLE_START -->"
    end_marker = "<!-- THERMAL_TABLE_END -->"

    if start_marker in base and end_marker in base:
        before = base[: base.index(start_marker)]
        after = base[base.index(end_marker) + len(end_marker):]
        new_content = before + table_md + after
    else:
        new_content = base.rstrip("\n") + "\n\n" + table_md + "\n"

    OUTPUT_README.write_text(new_content, encoding="utf-8")
    log.info("README.md aktualisiert.")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. GFT-Dateien ermitteln
    try:
        urls = get_gft_urls(BASE_URL)
    except Exception as e:
        log.error("URL-Ermittlung fehlgeschlagen: %s", e)
        sys.exit(1)

    # Neueste Datei verwenden (höchste Laufzeit = letzter Eintrag nach Sortierung)
    latest_url = urls[-1]
    filename = latest_url.split("/")[-1]
    run_dt, valid_start = parse_valid_times_from_filename(filename)

    log.info("Neueste Datei: %s", filename)
    log.info("Modell-Laufzeit: %s UTC", run_dt)
    log.info("Gültigkeitsbeginn: %s UTC", valid_start)

    # Vorhersagetage relativ zur Modell-Laufzeit bestimmen
    reference_date = (run_dt or datetime.datetime.now(datetime.timezone.utc)).date()
    forecast_dates = {
        "heute":       reference_date,
        "morgen":      reference_date + datetime.timedelta(days=1),
        "uebermorgen": reference_date + datetime.timedelta(days=2),
    }
    log.info("Vorhersagedaten: %s", forecast_dates)

    # 2. Download in temporäre Datei
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file(latest_url, tmp_path)

        # 3. GRIB2 öffnen
        log.info("Öffne GRIB2-Datei ...")
        ds = open_grib(tmp_path)
        log.info("Dataset-Variablen: %s", list(ds.data_vars))
        log.info("Dataset-Koordinaten: %s", list(ds.coords))

        # 4. Gemeinden laden
        if not CSV_PATH.exists():
            log.error("Gemeinde-CSV nicht gefunden: %s", CSV_PATH)
            sys.exit(1)
        df_mun = pd.read_csv(CSV_PATH)
        log.info("%d Gemeinden geladen.", len(df_mun))

        # 5. Vorhersage je Gemeinde
        results = process_municipalities(ds, df_mun, forecast_dates)

        # 6. JSON-Ausgabe
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "model_run": run_dt.isoformat() if run_dt else None,
                    "valid_start": valid_start.isoformat() if valid_start else None,
                    "forecast_dates": {k: v.isoformat() for k, v in forecast_dates.items()},
                    "municipalities": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        log.info("JSON gespeichert: %s", OUTPUT_JSON)

        # 7. README-Tabelle
        table_md = build_readme_table(results, forecast_dates, run_dt or datetime.datetime.now(datetime.timezone.utc))
        update_readme(table_md)

    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Fertig. %d Gemeinden aktualisiert.", len(results))


if __name__ == "__main__":
    main()
