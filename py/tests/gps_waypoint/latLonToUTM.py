#!/usr/bin/env python3
import sys
import pandas as pd
from pyproj import Transformer
import json
import math
import argparse

# Define known base stations (latitude, longitude)
BASE_STATIONS = {
    "SPPT": {"lat": -32.96025, "lon": 151.62366, "alt": 51},
    "CESS": {"lat": -32.83525, "lon": 151.35657, "alt": 110},
    "MTLD": {"lat": -32.73698, "lon": 151.56030, "alt": 49},
    "RAYM": {"lat": -32.76287, "lon": 151.74543, "alt": 43},
}

# Desired constant speed (m/s) for linear velocity
DESIRED_SPEED = 2.0


def convert_to_relative(df, base_lon, base_lat, projection="EPSG:28356"):
    """
    Convert coordinates in df to metre offsets (dx, dy) from base station.
    Supports:
      - Projected coords (columns 'X', 'Y') in same CRS.
      - Geographic coords ('longitude', 'latitude').
    """
    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    E0, N0 = transformer.transform(base_lon, base_lat)

    if {"X", "Y"}.issubset(df.columns):
        df["dx"] = df["X"] - E0
        df["dy"] = df["Y"] - N0
    elif {"longitude", "latitude"}.issubset(df.columns):
        Es, Ns = [], []
        for lon, lat in zip(df["longitude"], df["latitude"]):
            e, n = transformer.transform(lon, lat)
            Es.append(e)
            Ns.append(n)
        df["dx"] = [e - E0 for e in Es]
        df["dy"] = [n - N0 for n in Ns]
    else:
        raise ValueError(
            "CSV must have columns ['X','Y'] or ['longitude','latitude']")
    return df


def convert_csv_to_relative(input_csv, base_lon, base_lat, projection="EPSG:28356"):
    df = pd.read_csv(input_csv)
    return convert_to_relative(df, base_lon, base_lat, projection)


def write_csv(df, output_csv):
    df.to_csv(output_csv, index=False)


def write_json_poses(df, output_json, z=0.0):
    """
    Write a JSON track file with the structure:
    {
      "waypoints": [
        {
          "aFromB": { … },
          "frameA": "world",
          "frameB": "robot",
          "tangentOfBInA": { … }
        },
        …
      ]
    }
    """
    dxs = df["dx"].tolist()
    dys = df["dy"].tolist()
    n = len(dxs)

    # Compute headings (yaws)
    yaws = []
    for i in range(n):
        if i < n - 1:
            yaw = math.atan2(dys[i + 1] - dys[i], dxs[i + 1] - dxs[i])
        else:
            yaw = yaws[-1] if yaws else 0.0
        yaws.append(yaw)

    # Compute distances and dt for constant DESIRED_SPEED
    distances = []
    dt_intervals = []
    for i in range(n):
        if i < n - 1:
            dx = dxs[i + 1] - dxs[i]
            dy = dys[i + 1] - dys[i]
            dist = math.hypot(dx, dy)
            dt = dist / DESIRED_SPEED if DESIRED_SPEED > 0 else 0.0
        else:
            dist = distances[-1] if distances else 0.0
            dt = dt_intervals[-1] if dt_intervals else 0.0
        distances.append(dist)
        dt_intervals.append(dt)

    waypoints = []
    for i, (x, y, yaw) in enumerate(zip(dxs, dys, yaws)):
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        lin = DESIRED_SPEED if distances[i] > 0 else 0.0
        ang = (yaws[i] - yaws[i - 1]) / \
            dt_intervals[i] if i > 0 and dt_intervals[i] > 0 else 0.0

        pose_dict = {
            "aFromB": {
                "rotation": {
                    "unitQuaternion": {"imag": {"z": qz}, "real": qw}
                },
                "translation": {"x": x, "y": y}
            },
            "frameA": "world",
            "frameB": "robot",
            "tangentOfBInA": {
                "linearVelocity": {"x": lin},
                "angularVelocity": {"z": ang}
            }
        }
        waypoints.append(pose_dict)

    track = {"waypoints": waypoints}
    # write out the JSON
    with open(output_json, "w") as f:
        json.dump(track, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert waypoints CSV to relative CSV and poses JSON"
    )
    parser.add_argument("--input-csv", help="Path to input CSV of waypoints")
    parser.add_argument("--output-csv", help="Path for output relative CSV")
    parser.add_argument("--output-json", help="Path for output poses JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--station", choices=BASE_STATIONS.keys(),
                       help="Use a predefined base station name")
    group.add_argument("--base", nargs=2, metavar=("LON", "LAT"),
                       help="Provide base longitude and latitude manually")

    args = parser.parse_args()

    if args.station:
        base_info = BASE_STATIONS[args.station]
        base_lat = base_info["lat"]
        base_lon = base_info["lon"]
    else:
        base_lon = float(args.base[0])
        base_lat = float(args.base[1])

    df_rel = convert_csv_to_relative(args.input_csv, base_lon, base_lat)
    write_csv(df_rel, args.output_csv)
    write_json_poses(df_rel, args.output_json)
    print(
        f"Wrote relative CSV to '{args.output_csv}' and poses JSON to '{args.output_json}'")


if __name__ == "__main__":
    main()
