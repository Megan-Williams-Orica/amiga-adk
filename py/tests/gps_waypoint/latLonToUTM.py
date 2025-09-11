#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from pyproj import Transformer
import csv

# ---------------- Base stations (add yours here) ----------------
BASE_STATIONS: Dict[str, Dict[str, float]] = {
    "SPPT": {"E": 371365.70474972005, "N": 6352279.06411331, "alt": 51.1},
    "CESS":{"E": 346184.37336402666, "N": 6365780.99447662, "alt": 110.2},
    "MTLD":{"E": 365105.6363451328, "N": 6376954.683843763, "alt": 49.296},
    "RAYM":{"E": 382486.8256674537, "N": 6374304.294462996, "alt": 43.486},
}

# ---------------- Helpers ----------------
def _project_lonlat_to_en(lon: float, lat: float, base_crs: str, projection: str) -> Tuple[float, float]:
    tf = Transformer.from_crs(base_crs, projection, always_xy=True)
    E, N = tf.transform(lon, lat)
    return float(E), float(N)

def _resolve_base_EN_alt(
    *, station: Optional[str], base_en_alt: Optional[Tuple[float, float, Optional[float]]],
    base_lonlat_alt: Optional[Tuple[float, float, Optional[float]]], base_crs: str, projection: str
) -> Tuple[float, float, float]:
    if base_en_alt is not None:
        E, N, alt = base_en_alt
        return float(E), float(N), float(alt if alt is not None else 0.0)

    if station:
        info = BASE_STATIONS.get(station)
        if info is None:
            raise ValueError(f"Unknown station '{station}'. Add it to BASE_STATIONS.")
        alt = float(info.get("alt", 0.0))
        if "E" in info and "N" in info:
            return float(info["E"]), float(info["N"]), alt
        if "lon" in info and "lat" in info:
            E, N = _project_lonlat_to_en(float(info["lon"]), float(info["lat"]), base_crs, projection)
            return E, N, alt
        raise ValueError(f"Station '{station}' needs either (E,N[,alt]) or (lon,lat[,alt]).")

    if base_lonlat_alt is not None:
        lon, lat, alt = base_lonlat_alt
        E, N = _project_lonlat_to_en(float(lon), float(lat), base_crs, projection)
        return E, N, float(alt if alt is not None else 0.0)

    raise ValueError("No base provided. Use --station or --base-en or --base-lonlat.")

# ---------------- Core: fill dx/dy in place ----------------
def fill_dx_dy_inplace_csv(
    input_csv: Path,
    *,
    e_col: str = "X",      # column with MGA Easting
    n_col: str = "Y",      # column with MGA Northing
    base_E: float,
    base_N: float,
    xy_crs: str = None,    # set if X/Y are NOT already EPSG:7856
    projection: str = "EPSG:7856",
    overwrite: bool = False,  # False -> fill blanks only; True -> overwrite all
) -> None:
    """
    Fill dx/dy cells in-place. Do not change any other field.
    Keeps header & ordering exactly as-is.
    """
    tf = None
    if xy_crs and xy_crs != projection:
        tf = Transformer.from_crs(xy_crs, projection, always_xy=True)

    with open(input_csv, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    header = rows[0]
    try:
        idx_E  = header.index(e_col)
        idx_N  = header.index(n_col)
        idx_dx = header.index("dx")
        idx_dy = header.index("dy")
    except ValueError as e:
        raise SystemExit(f"Required column missing: {e}")

    for i in range(1, len(rows)):
        row = rows[i]
        if len(row) <= max(idx_dx, idx_dy, idx_E, idx_N):
            continue

        blank_dx = (row[idx_dx].strip() == "")
        blank_dy = (row[idx_dy].strip() == "")

        # Skip if we're only filling blanks and both already have values
        if not overwrite and not (blank_dx or blank_dy):
            continue

        try:
            E = float(row[idx_E])
            N = float(row[idx_N])
        except ValueError:
            # Non-numeric E/N -> leave row untouched
            continue

        if tf is not None:
            E, N = tf.transform(E, N)

        if overwrite or blank_dx:
            row[idx_dx] = str(E - base_E)
        if overwrite or blank_dy:
            row[idx_dy] = str(N - base_N)

    with open(input_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Append/Fill dx,dy (metres from base) IN PLACE in the CSV.")
    p.add_argument("--input-csv", required=True, type=Path, help="CSV to modify in place.")
    p.add_argument("--projection", default="EPSG:7856",
                   help="Working CRS for offsets (default: EPSG:7856 GDA2020/MGA56).")
    p.add_argument("--xy-crs", default=None,
                   help="CRS of input easting/northing columns if not already in --projection.")
    p.add_argument("--base-crs", default="EPSG:7844",
                   help="CRS of lon/lat if using --base-lonlat (default: EPSG:7844 GDA2020 geographic).")

    # Column mapping (your file: X = Easting, Y = Northing)
    p.add_argument("--e-col", default="X", help="Column containing MGA Easting (m).")
    p.add_argument("--n-col", default="Y", help="Column containing MGA Northing (m).")

    # Base selection (priority: --base-en > --station > --base-lonlat)
    p.add_argument("--station", choices=sorted(BASE_STATIONS.keys()),
                   help="Named base from the script dictionary.")
    p.add_argument("--base-en", nargs=2, type=float, metavar=("E", "N"),
                   help="Base Easting, Northing in --projection.")
    p.add_argument("--base-lonlat", nargs=2, type=float, metavar=("LON", "LAT"),
                   help="Base longitude, latitude in --base-crs.")
    p.add_argument("--base-alt", type=float, default=None,
                   help="Optional base altitude (unused here; dx/dy only).")

    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite any existing dx/dy instead of filling blanks only.")
    p.add_argument("--inplace", action="store_true",
                   help="Fill ONLY dx,dy in the input CSV and write the same file back. No other outputs.")

    args = p.parse_args()

    base_en_alt = (args.base_en[0], args.base_en[1], args.base_alt) if args.base_en else None
    base_lonlat_alt = (args.base_lonlat[0], args.base_lonlat[1], args.base_alt) if args.base_lonlat else None

    E0, N0, _ = _resolve_base_EN_alt(
        station=args.station,
        base_en_alt=base_en_alt,
        base_lonlat_alt=base_lonlat_alt,
        base_crs=args.base_crs,
        projection=args.projection,
    )

    if args.inplace:
        fill_dx_dy_inplace_csv(
            args.input_csv,
            e_col=args.e_col,
            n_col=args.n_col,
            base_E=E0,
            base_N=N0,
            xy_crs=args.xy_crs,
            projection=args.projection,
            overwrite=args.overwrite,
        )
        print(f"Updated '{args.input_csv}' (dx,dy {'overwritten' if args.overwrite else 'filled if blank'}).")
    else:
        print("Nothing to do. Pass --inplace to modify the CSV in place.")

if __name__ == "__main__":
    main()

