#!/usr/bin/env python3
"""
file_converter.py

Convert a GeoJSON of metre‐based waypoints into an Amiga Track JSON
by directly mapping each point to a waypoint (no interpolation).
"""

import json
import numpy as np
from pathlib import Path
from google.protobuf.json_format import MessageToJson
from farm_ng.track.track_pb2 import Track
from farm_ng_core_pybind import Isometry3F64, Pose3F64

# === Configuration ===
GEOJSON_PATH = "Waypoint_test.geojson"  # Input GeoJSON with metre-based points
OUTPUT_JSON  = "Waypoint_test.json"     # Amiga-format output

def main():
    # 1) Load the GeoJSON features
    data = json.loads(Path(GEOJSON_PATH).read_text())
    pts = [feat["geometry"]["coordinates"][:2] for feat in data["features"]]

    # 2) Build a Track proto with one Pose3F64 per point
    track = Track()
    for x, y in pts:
        iso = Isometry3F64()
        iso.translation = np.array([x, y, 0.0], dtype=np.float64)

        # Create Pose3F64 with identity rotation and zero tangent
        p = Pose3F64(iso, "world", "robot")
        track.waypoints.append(p.to_proto())

    # 3) Serialize to camelCase JSON for Amiga
    json_out = MessageToJson(track, preserving_proto_field_name=False)
    Path(OUTPUT_JSON).write_text(json_out)
    print(f"Wrote {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
