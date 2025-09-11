#!/usr/bin/env python3
"""
Two subplots from a CSV:
  • Subplot 1: X vs Y
  • Subplot 2: dx vs dy  (or compute from X,Y if --compute-dxdy and --base-en are given)

Examples:
  # just plot existing columns
  python plot_offsets.py --csv powerLinesSurveyed.csv --show

  # compute dx,dy on the fly from X,Y using a base (E,N) in EPSG:7856
  python plot_offsets.py --csv powerLinesSurveyed.csv \
      --compute-dxdy --base-en 371365.70474972005 6352279.06411331 --show

  # specify different column names
  python plot_offsets.py --csv data.csv --x-col Easting --y-col Northing --dx-col dx --dy-col dy --show
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

def _num_series(s: pd.Series) -> pd.Series:
    # tolerate commas, stray spaces, and empty strings
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")

def main():
    ap = argparse.ArgumentParser(description="Plot X/Y and dx/dy into two subplots.")
    ap.add_argument("--csv", required=True, type=Path, help="Input CSV file.")
    ap.add_argument("--x-col", default="X", help="Column for Easting (default: X).")
    ap.add_argument("--y-col", default="Y", help="Column for Northing (default: Y).")
    ap.add_argument("--dx-col", default="dx", help="Column for dx (default: dx).")
    ap.add_argument("--dy-col", default="dy", help="Column for dy (default: dy).")
    ap.add_argument("--title", default=None, help="Figure title.")
    ap.add_argument("--save", default=None, type=Path, help="Path to save the figure (PNG).")
    ap.add_argument("--show", action="store_true", help="Show the plot window.")
    ap.add_argument("--alpha", type=float, default=0.8, help="Point alpha.")
    ap.add_argument("--s", type=float, default=8.0, help="Point size.")
    ap.add_argument("--limit", type=int, default=None, help="Plot at most N rows.")

    # Optional: compute dx,dy from X,Y on the fly
    ap.add_argument("--compute-dxdy", action="store_true",
                    help="Compute dx,dy from X,Y for plotting (doesn't modify the CSV).")
    ap.add_argument("--base-en", nargs=2, type=float, metavar=("E", "N"),
                    help="Base Easting/Northing (EPSG:7856) for --compute-dxdy.")
    ap.add_argument("--xy-crs", default=None,
                    help="CRS of X/Y if not EPSG:7856 (e.g., EPSG:28356). Used only with --compute-dxdy.")
    ap.add_argument("--projection", default="EPSG:7856",
                    help="Target CRS for on-the-fly dx,dy (default EPSG:7856).")
    args = ap.parse_args()

    cols = [args.x_col, args.y_col, args.dx_col, args.dy_col]
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, usecols=cols)


    # Validate columns exist
    for col in (args.x_col, args.y_col):
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}' in {args.csv}")

    # Numeric X,Y
    X = _num_series(df[args.x_col])
    Y = _num_series(df[args.y_col])

    # Optionally limit rows
    if args.limit and args.limit > 0:
        X = X.iloc[:args.limit]
        Y = Y.iloc[:args.limit]
        df = df.iloc[:args.limit]

    # Prepare dx,dy (from file or computed)
    compute = args.compute_dxdy
    if compute:
        if args.base_en is None:
            raise SystemExit("--compute-dxdy requires --base-en E N")
        E0, N0 = float(args.base_en[0]), float(args.base_en[1])

        # Reproject X,Y if needed
        if args.xy_crs and args.xy_crs != args.projection:
            tf = Transformer.from_crs(args.xy_crs, args.projection, always_xy=True)
            E, N = tf.transform(X.to_numpy(), Y.to_numpy())
            dX = pd.Series(E - E0, index=X.index)
            dY = pd.Series(N - N0, index=Y.index)
        else:
            dX = X - E0
            dY = Y - N0
    else:
        # Use existing dx/dy columns
        for col in (args.dx_col, args.dy_col):
            if col not in df.columns:
                raise SystemExit(
                    f"Missing column '{col}' in {args.csv}. "
                    f"Either add/fill it, or use --compute-dxdy with --base-en."
                )
        dX = _num_series(df[args.dx_col])
        dY = _num_series(df[args.dy_col])

    # Masks & counts
    m_xy = ~(X.isna() | Y.isna())
    m_dxy = ~(dX.isna() | dY.isna())

    n_xy = int(m_xy.sum())
    n_dxy = int(m_dxy.sum())

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # X vs Y
    ax1.scatter(X[m_xy], Y[m_xy], s=args.s, alpha=args.alpha)
    ax1.set_xlabel(args.x_col)
    ax1.set_ylabel(args.y_col)
    ax1.set_title(f"{args.x_col} vs {args.y_col}  (n={n_xy})")
    ax1.grid(True, linestyle="--", linewidth=0.5)
    ax1.set_aspect("equal", adjustable="box")

    # dx vs dy
    if n_dxy == 0:
        ax2.text(0.5, 0.5, "No numeric dx/dy to plot",
                 ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.scatter(dX[m_dxy], dY[m_dxy], s=args.s, alpha=args.alpha)
    ax2.axhline(0, linewidth=0.8)
    ax2.axvline(0, linewidth=0.8)
    ax2.set_xlabel(args.dx_col if not compute else "dx (computed)")
    ax2.set_ylabel(args.dy_col if not compute else "dy (computed)")
    ax2.set_title(f"dx vs dy  (n={n_dxy})")
    ax2.grid(True, linestyle="--", linewidth=0.5)
    ax2.set_aspect("equal", adjustable="box")

    if args.title:
        fig.suptitle(args.title)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")

    if args.show or not args.save:
        plt.show()

if __name__ == "__main__":
    main()
