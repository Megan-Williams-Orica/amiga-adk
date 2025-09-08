#!/usr/bin/env python3
'''
Script is written to take csv blast hole data and plot 
course in coordinates relative to base station/Amiga unit.
'''
import pandas as pd
from pyproj import Transformer
import json

# configure your base here
base_lat, base_lon, base_alt = -32.96025, 151.62366, 0.0

# setup projection
transformer = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
base_easting, base_northing = transformer.transform(base_lon, base_lat)

# load your CSV (already in MGA56) and compute relative offsets
df = pd.read_csv("~/Downloads/physics_lab_pattern.csv")
df["rel_east"]  = df["X"] - base_easting
df["rel_north"] = df["Y"] - base_northing
df["rel_down"]  = df["Z"] - base_alt

# build the Amiga Track JSON
waypoints = []
for _, row in df.iterrows():
    wp = {
      "aFromB": {
        "rotation": { "unitQuaternion": { "imag": {"x":0,"y":0,"z":0}, "real":1 } },
        "translation": { "x": row["rel_north"], "y": row["rel_east"] }
      },
      "frameA": "world",
      "frameB": "robot",
      "tangentOfBInA": {
        "linearVelocity":  {"x": 0},
        "angularVelocity": {"z": 0}
      }
    }
    if row["rel_down"] != 0:
        wp["aFromB"]["translation"]["z"] = row["rel_down"]
    waypoints.append(wp)

track = {"waypoints": waypoints}
with open("track.json","w") as f:
    json.dump(track, f, indent=2)

print(f"Wrote {len(waypoints)} waypoints to track.json")
