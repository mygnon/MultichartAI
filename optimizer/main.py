"""
MultiCharts64 Parameter Plateau Optimizer
------------------------------------------
Usage examples:

  # Run all 4 strategies (requires MC64 running with correct workspaces)
  python main.py --strategies all

  # Run only daily strategies
  python main.py --strategies breakout_daily,supertrend_daily

  # Skip MC automation — re-analyze existing CSV files in results/
  python main.py --strategies all --from-csv C:\\Users\\Tim\\MultichartAI\\results

  # Dry run: just print config and estimated run counts
  python main.py --strategies all --dry-run

  # Tune plateau radius (default 2)
  python main.py --strategies all --from-csv results/ --radius 3
"""
from __future__ import annotations
import argparse
import ctypes
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _relaunch_as_admin() -> None:
    """Re-launch this script elevated (UAC prompt) and exit the current process.

    Uses ShellExecuteW("runas", python_exe, ...) directly — avoids cmd.exe
    quoting issues.  The elevated child window stays visible.
    """
    script = str(Path(sys.argv[0]).resolve())
    workdir = str(Path(sys.argv[0]).resolve().parent)

    # Rebuild argv without --_elevated to avoid passing it twice
    extra_args = [a for a in sys.argv[1:] if a != "--_elevated"]
    # Quote arguments that contain spaces
    quoted = [f'"{a}"' if (" " in a and not a.startswith('"')) else a
              for a in extra_args]
    all_args = f'"{script}" ' + " ".join(quoted) + " --_elevated"

    print("[auto-elevate] Requesting elevation — a UAC prompt will appear.")
    print(f"[auto-elevate] Command: {sys.executable} {all_args}")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1
    )
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). "
              "Please run manually as Administrator.")
    else:
        print(f"[auto-elevate] Elevated process launched (code={ret}).")


import plateau as plateau_mod
import visualize
from config import (
    ALL_STRATEGIES, STRATEGY_MAP, StrategyConfig,
    RESULTS_OUTPUT_DIR, PLATEAU_NEIGHBORHOOD_RADIUS, PLATEAU_TOP_N,
    OPTIMIZATION_TIMEOUT_SECONDS,
)
import mc_automation as mc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MultiCharts64 parameter plateau optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--strategies",
        default="all",
        help=(
            'Comma-separated strategy names or "all". '
            'Choices: breakout_daily, breakout_daily_stp_lmt, breakout_hourly, '
            'supertrend_daily, supertrend_hourly'
        ),
    )
    p.add_argument(
        "--from-csv",
        metavar="DIR",
        help="Skip MC automation; load existing CSV files from this directory",
    )
    p.add_argument(
        "--output-dir",
        default=RESULTS_OUTPUT_DIR,
        help=f"Directory for results (default: {RESULTS_OUTPUT_DIR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and estimated run counts, then exit",
    )
    p.add_argument(
        "--radius",
        type=int,
        default=PLATEAU_NEIGHBORHOOD_RADIUS,
        help=f"Plateau neighborhood radius (default: {PLATEAU_NEIGHBORHOOD_RADIUS})",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=PLATEAU_TOP_N,
        help=f"Number of top candidates to report (default: {PLATEAU_TOP_N})",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=OPTIMIZATION_TIMEOUT_SECONDS,
        help=f"Per-strategy optimization timeout in seconds (default: {OPTIMIZATION_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--skip-oos",
        action="store_true",
        help="Skip OOS validation even when MC connection is available",
    )
    p.add_argument(
        "--_elevated",
        action="store_true",
        help=argparse.SUPPRESS,  # internal flag — set by auto-elevation relaunch
    )
    return p.parse_args()


def select_strategies(args: argparse.Namespace) -> List[StrategyConfig]:
    if args.strategies.lower() == "all":
        return list(ALL_STRATEGIES)
    names = [n.strip() for n in args.strategies.split(",")]
    selected = []
    for name in names:
        key = name.lower().replace("-", "_")
        if key not in STRATEGY_MAP:
            print(f"[ERROR] Unknown strategy '{name}'. "
                  f"Valid: {list(STRATEGY_MAP.keys())}", file=sys.stderr)
            sys.exit(1)
        selected.append(STRATEGY_MAP[key])
    return selected


def _discover_existing_csvs(csv_dir: Path, strategies: List[StrategyConfig]) -> Dict[str, str]:
    paths = {}
    for cfg in strategies:
        p = csv_dir / f"{cfg.name}_raw.csv"
        if p.exists():
            paths[cfg.name] = str(p)
            logging.getLogger(__name__).info("Found existing CSV: %s", p)
        else:
            logging.getLogger(__name__).warning("No CSV found for %s at %s", cfg.name, p)
    return paths


def run_automation_phase(
    conn: mc.MultiChartsConnection,
    strategies: List[StrategyConfig],
    output_dir: Path,
    timeout: int,
) -> Dict[str, str]:
    csv_paths: Dict[str, str] = {}
    for cfg in strategies:
        t0 = time.time()
        try:
            csv_path = mc.run_optimization_for_strategy(conn, cfg, str(output_dir))
            elapsed = time.time() - t0
            logging.getLogger(__name__).info(
                "Completed %s in %.1f min", cfg.name, elapsed / 60
            )
            csv_paths[cfg.name] = csv_path
        except Exception as e:
            logging.getLogger(__name__).error(
                "FAILED %s: %s", cfg.name, e, exc_info=True
            )
    return csv_paths


def run_analysis_phase(
    csv_paths: Dict[str, str],
    strategies: List[StrategyConfig],
    output_dir: Path,
    radius: int,
    top_n: int,
) -> Tuple[Dict[str, list], Dict[str, str]]:
    logger = logging.getLogger(__name__)
    all_candidates: Dict[str, list] = {}
    heatmap_paths: Dict[str, str] = {}

    for cfg in strategies:
        csv_path = csv_paths.get(cfg.name)
        if not csv_path or not Path(csv_path).exists():
            logger.warning("No CSV for %s — skipping analysis", cfg.name)
            continue

        df = mc.load_results_csv(csv_path, cfg)
        logger.info("%s: %d valid rows loaded", cfg.name, len(df))

        p1, p2, grid, scores, candidates = plateau_mod.analyze(df, cfg, radius, top_n)
        all_candidates[cfg.name] = candidates

        hmap_path = visualize.plot_heatmap(p1, p2, grid, scores, cfg, candidates, str(output_dir))
        heatmap_paths[cfg.name] = hmap_path
        logger.info("Heatmap saved: %s", hmap_path)

        json_path = visualize.save_results_json(candidates, cfg, df, str(output_dir))
        logger.info("JSON saved: %s", json_path)

    return all_candidates, heatmap_paths


def run_oos_validation_phase(
    conn: mc.MultiChartsConnection,
    strategies: List[StrategyConfig],
    all_candidates: Dict[str, list],
    output_dir: Path,
) -> Dict[str, List[Optional[Dict[str, float]]]]:
    """
    For each top-N plateau candidate, run a single backtest on the OOS period
    and compare OOS MaxDrawdown vs IS MaxDrawdown.
    Returns dict mapping strategy name → list of OOS result dicts (parallel to candidates list).
    Each element is either {"NetProfit": x, "MaxDrawdown": y} or None if backtest failed.
    """
    logger = logging.getLogger(__name__)
    oos_results: Dict[str, List[Optional[Dict[str, float]]]] = {}

    for cfg in strategies:
        candidates = all_candidates.get(cfg.name, [])
        if not candidates:
            continue

        results: List[Optional[Dict[str, float]]] = []
        logger.info("OOS validation for %s (%d candidates)...", cfg.name, len(candidates))

        for c in candidates:
            params = {c.param1_name: c.param1_value, c.param2_name: c.param2_value}
            oos = mc.run_oos_backtest_for_candidate(conn, cfg, params)
            results.append(oos)
            if oos:
                oos_mdd = oos.get("MaxDrawdown", 0)
                mdd_break = abs(oos_mdd) > abs(c.max_drawdown)
                logger.info(
                    "  #%d %s=%.4g %s=%.4g | IS_MDD=%.0f OOS_MDD=%.0f | %s",
                    c.plateau_rank,
                    c.param1_name, c.param1_value,
                    c.param2_name, c.param2_value,
                    c.max_drawdown, oos_mdd,
                    "FAIL(MDD跌破)" if mdd_break else "PASS",
                )
            else:
                logger.warning("  #%d OOS backtest returned no data", c.plateau_rank)

        oos_results[cfg.name] = results
        _save_oos_json(cfg, candidates, results, str(output_dir))

    return oos_results


def _save_oos_json(
    cfg: StrategyConfig,
    candidates: list,
    oos_results: List[Optional[Dict[str, float]]],
    output_dir: str,
) -> None:
    import json
    from datetime import datetime
    rows = []
    for c, oos in zip(candidates, oos_results):
        oos_mdd = oos.get("MaxDrawdown", 0) if oos else None
        rows.append({
            "rank": c.plateau_rank,
            c.param1_name: c.param1_value,
            c.param2_name: c.param2_value,
            "is_net_profit": round(c.net_profit, 2),
            "is_max_drawdown": round(c.max_drawdown, 2),
            "oos_net_profit": round(oos["NetProfit"], 2) if oos else None,
            "oos_max_drawdown": round(oos_mdd, 2) if oos_mdd is not None else None,
            "mdd_breaks": (abs(oos_mdd) > abs(c.max_drawdown)) if oos_mdd is not None else None,
        })
    payload = {
        "strategy": cfg.name,
        "insample": {"from": cfg.insample.from_date, "to": cfg.insample.to_date},
        "outsample": {"from": cfg.outsample.from_date, "to": cfg.outsample.to_date},
        "candidates": rows,
    }
    out_path = os.path.join(output_dir, f"{cfg.name}_oos.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.getLogger(__name__).info("OOS JSON saved: %s", out_path)


def _preflight_check(
    conn: mc.MultiChartsConnection,
    strategies: List[StrategyConfig],
    logger: logging.Logger,
) -> None:
    """Warn about any strategy workspaces not currently open in MC."""
    open_windows = mc._find_all_mc_windows()
    open_titles = [title for _, title, _ in open_windows]
    missing = []
    for cfg in strategies:
        stem = Path(cfg.chart_workspace).stem
        if not any(stem in t for t in open_titles):
            missing.append((cfg.name, cfg.chart_workspace))
    if missing:
        logger.warning("The following workspaces are NOT open in MultiCharts64:")
        for name, path in missing:
            logger.warning("  %-25s  %s", name, path)
        logger.warning(
            "Please open them manually before running, or the script will fail "
            "when it reaches those strategies."
        )
    else:
        logger.info("Preflight OK — all %d workspaces found in MC.", len(strategies))


def main() -> int:
    args = parse_args()

    # MC64 runs as Administrator; we must match its integrity level for UI automation.
    # Auto-elevate via UAC if not already elevated (unless --from-csv / --dry-run).
    _elevated_flag = getattr(args, "_elevated", False)
    if not args.from_csv and not args.dry_run and not _is_admin() and not _elevated_flag:
        _relaunch_as_admin()
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.output_dir, f"run_{int(time.time())}.log"),
                encoding="utf-8",
            ) if not args.dry_run else logging.NullHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    strategies = select_strategies(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(s.total_runs() for s in strategies)
    logger.info("Strategies: %s", [s.name for s in strategies])
    logger.info("Estimated total runs: %d", total_runs)
    if total_runs > 5000:
        logger.warning("Total runs (%d) exceeds 5000 — consider reducing grid size", total_runs)

    if args.dry_run:
        print("\n=== DRY RUN — no MC interaction ===")
        for s in strategies:
            print(f"  {s.name:<25}  {s.total_runs():>5} runs  "
                  f"params: {[(p.name, p.start, p.stop, p.step) for p in s.params]}")
        print(f"\nTotal: {total_runs} runs")
        return 0

    conn: Optional[mc.MultiChartsConnection] = None

    # Phase 1: Automation or load from CSV
    if args.from_csv:
        csv_paths = _discover_existing_csvs(Path(args.from_csv), strategies)
    else:
        conn = mc.MultiChartsConnection()
        conn.connect()
        _preflight_check(conn, strategies, logger)
        csv_paths = run_automation_phase(conn, strategies, output_dir, args.timeout)

    if not csv_paths:
        logger.error("No CSV files available for analysis. Exiting.")
        return 1

    # Phase 2: IS analysis — plateau detection
    all_candidates, heatmap_paths = run_analysis_phase(
        csv_paths, strategies, output_dir, args.radius, args.top_n
    )

    # Phase 3: OOS validation (requires live MC connection)
    oos_results: Dict[str, List[Optional[Dict[str, float]]]] = {}
    if conn is not None and not args.skip_oos and all_candidates:
        oos_results = run_oos_validation_phase(conn, strategies, all_candidates, output_dir)
    elif args.from_csv:
        logger.info("OOS validation skipped (--from-csv mode; no MC connection).")

    # Phase 4: Final report
    visualize.print_summary_table(all_candidates, oos_results)

    html_path = visualize.generate_html_report(
        all_candidates, strategies, heatmap_paths, str(output_dir), oos_results
    )
    logger.info("HTML report: %s", html_path)
    print(f"\nReport ready: {html_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
