"""
Analyze decoded optimization results — plateau score + best parameter report.
Objective = NetProfit^2 / |MaxDrawdown|  (only when NetProfit > 0 AND MaxDD < 0)
"""
import csv
import struct
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter
import os

CSV_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_decoded.csv"
OUT_DIR = r"C:\Users\Tim\MultichartAI\results"
RADIUS = 2  # plateau neighborhood radius


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def build_grid(rows, p1_name, p2_name, obj_name):
    p1_vals = sorted(set(r[p1_name] for r in rows))
    p2_vals = sorted(set(r[p2_name] for r in rows))
    p1_idx = {v: i for i, v in enumerate(p1_vals)}
    p2_idx = {v: i for i, v in enumerate(p2_vals)}
    grid = np.zeros((len(p1_vals), len(p2_vals)), dtype=float)
    for r in rows:
        i = p1_idx[r[p1_name]]
        j = p2_idx[r[p2_name]]
        grid[i, j] = r[obj_name]
    return grid, p1_vals, p2_vals


def main():
    print(f"Loading: {CSV_PATH}")
    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} rows")

    # Build objective column
    net_col = "Net Profit"
    dd_col = "Max Intraday Drawdown"

    for r in rows:
        np_ = r[net_col]
        dd = r[dd_col]
        if np_ > 0 and dd < 0:
            r["objective"] = (np_ * np_) / abs(dd)
        else:
            r["objective"] = 0.0

    nonzero = sum(1 for r in rows if r["objective"] > 0)
    print(f"Rows with objective > 0: {nonzero} / {len(rows)}")

    # Build 2D grid
    grid, le_vals, se_vals = build_grid(rows, "LE", "SE", "objective")
    print(f"Grid shape: {grid.shape}  ({len(le_vals)} LE values x {len(se_vals)} SE values)")
    print(f"Grid max objective: {grid.max():.2f}  at index {np.unravel_index(grid.argmax(), grid.shape)}")

    # Plateau score = sliding minimum over (2r+1) neighborhood
    plateau = minimum_filter(grid, size=(2*RADIUS+1, 2*RADIUS+1), mode="nearest")

    # Find top plateau candidates
    flat_indices = np.argsort(plateau.ravel())[::-1]

    print(f"\n=== TOP 20 PLATEAU CANDIDATES (radius={RADIUS}) ===")
    print(f"{'LE':>4} {'SE':>4}  {'NetProfit':>12}  {'MaxDD':>14}  {'Objective':>14}  {'PlateauScore':>14}")

    seen = set()
    count = 0
    for flat_idx in flat_indices:
        if count >= 20:
            break
        i, j = np.unravel_index(flat_idx, grid.shape)
        le = int(le_vals[i])
        se = int(se_vals[j])
        if (le, se) in seen:
            continue
        seen.add((le, se))
        obj = grid[i, j]
        plat = plateau[i, j]
        # Find original row
        row = next(r for r in rows if int(r["LE"]) == le and int(r["SE"]) == se)
        net_p = row[net_col]
        max_dd = row[dd_col]
        print(f"  {le:>4} {se:>4}  {net_p:>12.2f}  {max_dd:>14.2f}  {obj:>14.2f}  {plat:>14.2f}")
        count += 1

    # Find single best by objective (ignoring plateau)
    best_by_obj = sorted(rows, key=lambda r: r["objective"], reverse=True)[:5]
    print(f"\n=== TOP 5 BY OBJECTIVE ONLY (no plateau filter) ===")
    print(f"{'LE':>4} {'SE':>4}  {'NetProfit':>12}  {'MaxDD':>14}  {'Objective':>14}")
    for r in best_by_obj:
        print(f"  {int(r['LE']):>4} {int(r['SE']):>4}  {r[net_col]:>12.2f}  {r[dd_col]:>14.2f}  {r['objective']:>14.2f}")

    # Find best by plateau score × objective
    all_pts = []
    for i in range(len(le_vals)):
        for j in range(len(se_vals)):
            le = int(le_vals[i])
            se = int(se_vals[j])
            obj = grid[i, j]
            plat = plateau[i, j]
            all_pts.append((plat, obj, le, se))
    all_pts.sort(reverse=True)

    print(f"\n=== OVERALL RECOMMENDATION ===")
    print(f"Best plateau score: LE={all_pts[0][2]}, SE={all_pts[0][3]}")
    print(f"  Plateau score: {all_pts[0][0]:.2f}")
    print(f"  Objective:     {all_pts[0][1]:.2f}")
    row = next(r for r in rows if int(r["LE"]) == all_pts[0][2] and int(r["SE"]) == all_pts[0][3])
    print(f"  Net Profit:    {row[net_col]:.2f}")
    print(f"  Max Drawdown:  {row[dd_col]:.2f}")
    total_t = row.get("Total Trades", 0)
    pct_win = row.get("Pct Profitable", 0)
    print(f"  Total Trades:  {int(total_t)}")
    print(f"  Win Rate:      {pct_win:.1f}%")

    # Save enriched CSV with objective and plateau columns
    out_enr = os.path.join(OUT_DIR, "Breakout_Daily_analyzed.csv")
    fieldnames = list(rows[0].keys()) + ["plateau_score"]
    for i_r, r in enumerate(rows):
        le = int(r["LE"])
        se = int(r["SE"])
        i = le_vals.index(float(le))
        j = se_vals.index(float(se))
        r["plateau_score"] = plateau[i, j]

    with open(out_enr, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAnalyzed CSV saved: {out_enr}")


if __name__ == "__main__":
    main()
