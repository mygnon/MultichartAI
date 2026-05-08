from __future__ import annotations
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import StrategyConfig, PLATEAU_NEIGHBORHOOD_RADIUS
from plateau import PlateauResult


def plot_heatmap(
    p1_vals: np.ndarray,
    p2_vals: np.ndarray,
    objective_grid: np.ndarray,
    plateau_grid: np.ndarray,
    cfg: StrategyConfig,
    candidates: List[PlateauResult],
    output_dir: str,
) -> str:
    p1_name = cfg.params[0].name
    p2_name = cfg.params[1].name

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{cfg.name}  —  Objective & Plateau Score  (radius={PLATEAU_NEIGHBORHOOD_RADIUS})",
        fontsize=13, fontweight="bold",
    )

    def _draw_panel(ax, grid, title):
        clean = np.where(np.isnan(grid), 0.0, grid)
        # use p2 as x-axis (columns) and p1 as y-axis (rows)
        p1_labels = [f"{v:.4g}" for v in p1_vals]
        p2_labels = [f"{v:.4g}" for v in p2_vals]
        # show every Nth tick to avoid crowding
        max_ticks = 15
        p1_step = max(1, len(p1_labels) // max_ticks)
        p2_step = max(1, len(p2_labels) // max_ticks)

        sns.heatmap(
            clean,
            ax=ax,
            cmap="RdYlGn",
            xticklabels=p2_step,
            yticklabels=p1_step,
            linewidths=0,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xticklabels(
            [p2_labels[i] for i in range(0, len(p2_labels), p2_step)],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_yticklabels(
            [p1_labels[i] for i in range(0, len(p1_labels), p1_step)],
            rotation=0, fontsize=7,
        )
        ax.set_xlabel(p2_name, fontsize=9)
        ax.set_ylabel(p1_name, fontsize=9)
        ax.set_title(title, fontsize=10)

        # mark top-5 candidates
        for c in candidates[:5]:
            ci = np.searchsorted(p1_vals, c.param1_value)
            cj = np.searchsorted(p2_vals, c.param2_value)
            ax.plot(cj + 0.5, ci + 0.5, marker="*", color="white",
                    markersize=12, markeredgecolor="black", markeredgewidth=0.5)
            if c.plateau_rank <= 3:
                ax.text(cj + 0.5, ci - 0.2, f"#{c.plateau_rank}",
                        color="white", fontsize=7, ha="center", fontweight="bold")

    _draw_panel(axes[0], objective_grid, f"Objective  (NetProfit² / |MaxDD|)")
    _draw_panel(axes[1], plateau_grid,   f"Plateau Score  (min in {2*PLATEAU_NEIGHBORHOOD_RADIUS+1}×{2*PLATEAU_NEIGHBORHOOD_RADIUS+1} window)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{cfg.name}_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_results_json(
    candidates: List[PlateauResult],
    cfg: StrategyConfig,
    df: pd.DataFrame,
    output_dir: str,
) -> str:
    total_runs = len(df)
    valid_runs = int((df["Objective"] > 0).sum())

    payload = {
        "strategy": cfg.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_runs": total_runs,
        "valid_runs": valid_runs,
        "plateau_radius": PLATEAU_NEIGHBORHOOD_RADIUS,
        "candidates": [
            {
                "rank": c.plateau_rank,
                c.param1_name: c.param1_value,
                c.param2_name: c.param2_value,
                "objective": round(c.objective, 2),
                "plateau_score": round(c.plateau_score, 2),
                "neighborhood_mean": round(c.neighborhood_mean, 2),
                "neighborhood_std": round(c.neighborhood_std, 2),
                "net_profit": round(c.net_profit, 2),
                "max_drawdown": round(c.max_drawdown, 2),
                "total_trades": c.total_trades,
            }
            for c in candidates
        ],
    }
    out_path = os.path.join(output_dir, f"{cfg.name}_plateau.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def print_summary_table(
    all_candidates: Dict[str, List[PlateauResult]],
    oos_results: Optional[Dict[str, list]] = None,
) -> None:
    for strategy_name, candidates in all_candidates.items():
        oos_list = (oos_results or {}).get(strategy_name, [])
        has_oos = bool(oos_list)
        sep = "=" * (90 if has_oos else 72)
        print(f"\n{sep}")
        print(f"  {strategy_name}")
        print(sep)
        if not candidates:
            print("  (no valid candidates)")
            continue
        c0 = candidates[0]
        header = (
            f"{'Rank':>4}  {c0.param1_name:>10}  {c0.param2_name:>8}"
            f"  {'NetProfit(IS)':>13}  {'MaxDD(IS)':>10}  {'PlateauScore':>13}  {'Trades':>6}"
        )
        if has_oos:
            header += f"  {'OOS_NP':>12}  {'OOS_MDD':>10}  {'MDD?':>6}"
        print(header)
        print("-" * len(header))
        for idx, c in enumerate(candidates[:10]):
            oos = oos_list[idx] if idx < len(oos_list) else None
            line = (
                f"  #{c.plateau_rank:<3}  {c.param1_value:>10.4g}  {c.param2_value:>8.4g}"
                f"  {c.net_profit:>13,.0f}  {c.max_drawdown:>10,.0f}"
                f"  {c.plateau_score:>13,.0f}  {c.total_trades:>6}"
            )
            if has_oos:
                if oos:
                    oos_np = oos.get("NetProfit", 0)
                    oos_mdd = oos.get("MaxDrawdown", 0)
                    mdd_break = abs(oos_mdd) > abs(c.max_drawdown)
                    mdd_tag = "FAIL" if mdd_break else "PASS"
                    line += f"  {oos_np:>12,.0f}  {oos_mdd:>10,.0f}  {mdd_tag:>6}"
                else:
                    line += f"  {'N/A':>12}  {'N/A':>10}  {'N/A':>6}"
            print(line)


def generate_html_report(
    all_candidates: Dict[str, List[PlateauResult]],
    cfg_list: List[StrategyConfig],
    heatmap_paths: Dict[str, str],
    output_dir: str,
    oos_results: Optional[Dict[str, list]] = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []

    for cfg in cfg_list:
        candidates = all_candidates.get(cfg.name, [])
        hmap_path = heatmap_paths.get(cfg.name)
        oos_list = (oos_results or {}).get(cfg.name, [])
        has_oos = bool(oos_list)

        img_tag = ""
        if hmap_path and os.path.exists(hmap_path):
            with open(hmap_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0;">'

        rows = ""
        for idx, c in enumerate(candidates[:20]):
            oos = oos_list[idx] if idx < len(oos_list) else None
            oos_cols = ""
            if has_oos:
                if oos:
                    oos_np = oos.get("NetProfit", 0)
                    oos_mdd = oos.get("MaxDrawdown", 0)
                    mdd_break = abs(oos_mdd) > abs(c.max_drawdown)
                    style = ' style="color:#c00;font-weight:bold"' if mdd_break else ' style="color:#080"'
                    tag = "FAIL" if mdd_break else "PASS"
                    oos_cols = (
                        f"<td>{oos_np:,.0f}</td><td>{oos_mdd:,.0f}</td>"
                        f"<td{style}>{tag}</td>"
                    )
                else:
                    oos_cols = "<td>—</td><td>—</td><td>—</td>"
            rows += (
                f"<tr><td>{c.plateau_rank}</td>"
                f"<td>{c.param1_value:.4g}</td><td>{c.param2_value:.4g}</td>"
                f"<td>{c.net_profit:,.0f}</td><td>{c.max_drawdown:,.0f}</td>"
                f"<td>{c.objective:,.0f}</td><td>{c.plateau_score:,.0f}</td>"
                f"<td>{c.neighborhood_mean:,.0f}</td><td>{c.neighborhood_std:,.0f}</td>"
                f"<td>{c.total_trades}</td>{oos_cols}</tr>\n"
            )

        p1 = cfg.params[0].name
        p2 = cfg.params[1].name
        oos_headers = (
            "<th>OOS NetProfit</th><th>OOS MaxDD</th><th>MDD跌破?</th>"
            if has_oos else ""
        )
        oos_period = (
            f" &nbsp;|&nbsp; OOS: {cfg.outsample.from_date} ~ {cfg.outsample.to_date}"
            if has_oos else ""
        )
        sections.append(f"""
        <section>
          <h2>{cfg.name}</h2>
          <p class="meta">IS: {cfg.insample.from_date} ~ {cfg.insample.to_date}{oos_period}</p>
          {img_tag}
          <table>
            <thead>
              <tr>
                <th>Rank</th><th>{p1}</th><th>{p2}</th>
                <th>NetProfit(IS)</th><th>MaxDD(IS)</th>
                <th>Objective</th><th>PlateauScore</th>
                <th>Nbhd Mean</th><th>Nbhd Std</th><th>Trades</th>
                {oos_headers}
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>MultiCharts64 Parameter Plateau Report</title>
<style>
  body {{ font-family: sans-serif; margin: 32px; color: #222; }}
  h1 {{ color: #1a4a8a; }}
  h2 {{ color: #2c6fad; border-bottom: 2px solid #cce; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; font-size: 13px; width: 100%; margin-bottom: 24px; }}
  th {{ background: #2c6fad; color: white; padding: 6px 10px; text-align: right; }}
  th:first-child {{ text-align: center; }}
  td {{ padding: 5px 10px; text-align: right; border-bottom: 1px solid #eee; }}
  td:first-child {{ text-align: center; }}
  tr:nth-child(even) {{ background: #f5f8ff; }}
  .meta {{ color: #666; font-size: 12px; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>MultiCharts64 Parameter Plateau Report</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Objective = NetProfit² / |MaxDrawdown| &nbsp;|&nbsp;
Plateau Score = min objective in {2*PLATEAU_NEIGHBORHOOD_RADIUS+1}×{2*PLATEAU_NEIGHBORHOOD_RADIUS+1} neighborhood</p>
{''.join(sections)}
</body>
</html>"""

    out_path = os.path.join(output_dir, "optimization_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
