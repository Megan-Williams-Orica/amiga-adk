#!/usr/bin/env python3
"""
plot_holes.py

Plot blast holes from a CSV file with optional bearing arrows.
Supports diameter in millimetres and offsets label above hole.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# === Parse arguments ===
parser = argparse.ArgumentParser(description="Plot blast holes from CSV")
parser.add_argument("--csv", help="CSV file containing hole data")
args = parser.parse_args()

# === Load CSV ===
df = pd.read_csv(args.csv)

# Column names
ID_COL      = "ID"
X_COL       = "X"
Y_COL       = "Y"
Z_COL       = "Z"
DIAM_COL    = "Diameter"  # in millimetres

# Convert diameters from mm to metres
df["diam_m"] = df[DIAM_COL] / 1000.0

# === Create plot ===
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal", "box")
ax.set_title("Blast Hole Locations")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# Maximum diameter (metres) for arrow sizing
max_diam_m = df["diam_m"].max()

# Plot circles and labels
label_offset = max_diam_m * 0.2  # offset label by 20% of max diameter
for _, row in df.iterrows():
    x, y = row[X_COL], row[Y_COL]
    diam_m = row["diam_m"]
    hid = row[ID_COL]
    radius = diam_m / 2.0

    # draw hole circle
    circle = Circle(
        (x, y),
        radius=radius,
        edgecolor="black",
        facecolor="cyan",
        alpha=0.5,
        linewidth=1
    )
    ax.add_patch(circle)

    # label above hole
    ax.text(
        x,
        y + radius + label_offset,
        str(hid),
        ha="center",
        va="bottom",
        fontsize=8,
        color="red"
    )

# Autoscale with a 5 m gap beyond the holes
max_radius = df["diam_m"].max() / 2.0
border_gap = 5.0  # metres
margin = max_radius + border_gap

ax.set_xlim(df[X_COL].min() - margin, df[X_COL].max() + margin)
ax.set_ylim(df[Y_COL].min() - margin, df[Y_COL].max() + margin)

ax.grid(True)
plt.show()
