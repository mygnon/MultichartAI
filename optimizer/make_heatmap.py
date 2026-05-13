"""Generate heatmap for Breakout_Daily optimization results."""
import csv
import numpy as np
from scipy.ndimage import minimum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

CSV_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_decoded.csv"
OUT_DIR = r"C:\Users\Tim\MultichartAI\results"
RADIUS = 2


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    rows = load_csv(CSV_PATH)
    for r in rows:
        np_ = r["Net Profit"]
        dd = r["Max Intraday Drawdown"]
        r["objective"] = (np_ * np_) / abs(dd) if np_ > 0 and dd < 0 else 0.0

    le_vals = sorted(set(int(r["LE"]) for r in rows))
    se_vals = sorted(set(int(r["SE"]) for r in rows))
    le_idx = {v: i for i, v in enumerate(le_vals)}
    se_idx = {v: i for i, v in enumerate(se_vals)}

    grid = np.zeros((len(le_vals), len(se_vals)))
    np_grid = np.zeros_like(grid)
    dd_grid = np.zeros_like(grid)
    for r in rows:
        i, j = le_idx[int(r["LE"])], se_idx[int(r["SE"])]
        grid[i, j] = r["objective"]
        np_grid[i, j] = r["Net Profit"]
        dd_grid[i, j] = r["Max Intraday Drawdown"]

    plateau = minimum_filter(grid, size=(2*RADIUS+1, 2*RADIUS+1), mode="nearest")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Breakout Daily — Parameter Optimization  (LE x SE, 50x50)", fontsize=14)

    # Objective heatmap
    ax = axes[0]
    im = ax.imshow(grid.T / 1e6, aspect="auto", origin="lower",
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_title("Objective  (NetProfit^2 / |MaxDD|)  [millions]")
    ax.set_xlabel("LE")
    ax.set_ylabel("SE")
    step = 5
    ax.set_xticks(range(0, len(le_vals), step))
    ax.set_xticklabels(le_vals[::step])
    ax.set_yticks(range(0, len(se_vals), step))
    ax.set_yticklabels(se_vals[::step])
    fig.colorbar(im, ax=ax)
    # Mark top-5 objective
    top5_obj = sorted(rows, key=lambda r: r["objective"], reverse=True)[:5]
    for r in top5_obj:
        xi = le_idx[int(r["LE"])]
        yi = se_idx[int(r["SE"])]
        ax.plot(xi, yi, "b*", markersize=12)

    # Plateau heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(plateau.T / 1e6, aspect="auto", origin="lower",
                     cmap="RdYlGn", interpolation="nearest")
    ax2.set_title(f"Plateau Score (radius={RADIUS}) [millions]")
    ax2.set_xlabel("LE")
    ax2.set_ylabel("SE")
    ax2.set_xticks(range(0, len(le_vals), step))
    ax2.set_xticklabels(le_vals[::step])
    ax2.set_yticks(range(0, len(se_vals), step))
    ax2.set_yticklabels(se_vals[::step])
    fig.colorbar(im2, ax=ax2)
    # Mark top-5 plateau
    flat = np.argsort(plateau.ravel())[::-1]
    seen = set()
    cnt = 0
    for fi in flat:
        if cnt >= 5:
            break
        i, j = np.unravel_index(fi, plateau.shape)
        k = (i, j)
        if k not in seen:
            seen.add(k)
            ax2.plot(i, j, "b*", markersize=12)
            ax2.text(i+0.5, j+0.5, f"LE={le_vals[i]}\nSE={se_vals[j]}", fontsize=7, color="blue")
            cnt += 1

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Breakout_Daily_heatmap.png")
    plt.savefig(out_path, dpi=120)
    print(f"Saved heatmap: {out_path}")

    # Print final recommendation table
    print("\n=== FINAL RECOMMENDATION ===")
    print("Top plateau candidates (LE=7-9, SE=43-50 region):")
    print(f"{'LE':>4} {'SE':>4}  {'NetProfit':>12}  {'MaxDD':>14}  {'Objective':>14}  {'PlateauScore':>14}")

    all_pts = []
    for i in range(len(le_vals)):
        for j in range(len(se_vals)):
            all_pts.append((plateau[i,j], grid[i,j], le_vals[i], se_vals[j]))
    all_pts.sort(reverse=True)

    seen2 = set()
    cnt2 = 0
    for plat, obj, le, se in all_pts:
        if cnt2 >= 10:
            break
        if (le, se) in seen2:
            continue
        seen2.add((le, se))
        r = next(x for x in rows if int(x["LE"]) == le and int(x["SE"]) == se)
        print(f"  {le:>4} {se:>4}  {r['Net Profit']:>12.0f}  {r['Max Intraday Drawdown']:>14.0f}  {obj:>14.0f}  {plat:>14.0f}")
        cnt2 += 1


if __name__ == "__main__":
    main()
