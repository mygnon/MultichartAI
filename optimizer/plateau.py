from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, uniform_filter

from config import StrategyConfig, PLATEAU_NEIGHBORHOOD_RADIUS, PLATEAU_TOP_N


@dataclass
class PlateauResult:
    strategy_name: str
    param1_name: str
    param2_name: str
    param1_value: float
    param2_value: float
    objective: float
    plateau_score: float        # min objective in neighborhood
    plateau_rank: int
    neighborhood_mean: float
    neighborhood_std: float
    net_profit: float
    max_drawdown: float
    total_trades: int


def compute_objective(df: pd.DataFrame) -> pd.Series:
    """
    Objective = NetProfit² / abs(MaxDrawdown)
    条件: NetProfit > 0 且 MaxDrawdown < 0，否则为 0
    """
    obj = pd.Series(0.0, index=df.index)
    mask = (df["NetProfit"] > 0) & (df["MaxDrawdown"] < 0)
    obj[mask] = df.loc[mask, "NetProfit"] ** 2 / df.loc[mask, "MaxDrawdown"].abs()
    return obj


def _detect_variable_params(df: pd.DataFrame, cfg: StrategyConfig) -> Tuple[str, str]:
    """Return the names of the two params that vary in df (>1 unique value).

    Skips fixed params (only 1 unique value) so plateau analysis always uses
    the actually-swept axes even when fixed params are listed in cfg.params.
    """
    variable = [p.name for p in cfg.params if p.name in df.columns and df[p.name].nunique() > 1]
    if len(variable) >= 2:
        return variable[0], variable[1]
    # fallback: just use first two listed params
    names = [p.name for p in cfg.params]
    return names[0], names[1]


def build_objective_grid(
    df: pd.DataFrame,
    cfg: StrategyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape flat DataFrame into a 2D grid for spatial analysis.

    Returns:
        p1_vals: sorted unique values of param1, shape (N,)
        p2_vals: sorted unique values of param2, shape (M,)
        grid:    shape (N, M), grid[i,j] = Objective at (p1[i], p2[j]), NaN if missing
    """
    p1_name, p2_name = _detect_variable_params(df, cfg)

    p1_vals = np.sort(df[p1_name].unique())
    p2_vals = np.sort(df[p2_name].unique())

    grid = np.full((len(p1_vals), len(p2_vals)), np.nan)
    p1_idx = {v: i for i, v in enumerate(p1_vals)}
    p2_idx = {v: j for j, v in enumerate(p2_vals)}

    for _, row in df.iterrows():
        i = p1_idx.get(row[p1_name])
        j = p2_idx.get(row[p2_name])
        if i is not None and j is not None:
            grid[i, j] = row["Objective"]

    return p1_vals, p2_vals, grid


def _clean_grid(grid: np.ndarray) -> np.ndarray:
    clean = np.where(np.isnan(grid), 0.0, grid)
    return np.clip(clean, 0.0, None)


def compute_plateau_scores(
    grid: np.ndarray,
    radius: int = PLATEAU_NEIGHBORHOOD_RADIUS,
) -> np.ndarray:
    """
    Sliding minimum over (2r+1)×(2r+1) neighborhood.
    A point scores high only when ALL neighbors are also high — the true plateau definition.
    """
    size = 2 * radius + 1
    return minimum_filter(_clean_grid(grid), size=size, mode="nearest")


def compute_neighborhood_stats(
    grid: np.ndarray,
    radius: int = PLATEAU_NEIGHBORHOOD_RADIUS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_grid, std_grid) over each point's neighborhood."""
    size = 2 * radius + 1
    clean = _clean_grid(grid)
    mean_grid = uniform_filter(clean, size=size, mode="nearest")
    mean_sq = uniform_filter(clean ** 2, size=size, mode="nearest")
    var_grid = np.maximum(mean_sq - mean_grid ** 2, 0.0)
    return mean_grid, np.sqrt(var_grid)


def rank_plateau_candidates(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    plateau_scores: np.ndarray,
    mean_scores: np.ndarray,
    std_scores: np.ndarray,
    p1_vals: np.ndarray,
    p2_vals: np.ndarray,
    top_n: int = PLATEAU_TOP_N,
) -> List[PlateauResult]:
    p1_name, p2_name = _detect_variable_params(df, cfg)

    flat_scores = plateau_scores.ravel()
    sorted_idx = np.argsort(flat_scores)[::-1]

    seen: set = set()
    results: List[PlateauResult] = []
    rank = 0

    for flat_i in sorted_idx:
        if len(results) >= top_n:
            break
        i, j = np.unravel_index(flat_i, plateau_scores.shape)
        p1v = float(p1_vals[i])
        p2v = float(p2_vals[j])

        # deduplicate adjacent cells with identical plateau scores
        key = round(p1v, 8), round(p2v, 8)
        if key in seen:
            continue
        seen.add(key)

        # look up raw metrics from DataFrame
        mask = (df[p1_name] == p1v) & (df[p2_name] == p2v)
        row = df[mask]
        if row.empty:
            continue

        row = row.iloc[0]
        rank += 1
        results.append(PlateauResult(
            strategy_name=cfg.name,
            param1_name=p1_name,
            param2_name=p2_name,
            param1_value=p1v,
            param2_value=p2v,
            objective=float(row["Objective"]),
            plateau_score=float(flat_scores[flat_i]),
            plateau_rank=rank,
            neighborhood_mean=float(mean_scores[i, j]),
            neighborhood_std=float(std_scores[i, j]),
            net_profit=float(row["NetProfit"]),
            max_drawdown=float(row["MaxDrawdown"]),
            total_trades=int(row["TotalTrades"]),
        ))

    return results


def analyze(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    radius: int = PLATEAU_NEIGHBORHOOD_RADIUS,
    top_n: int = PLATEAU_TOP_N,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[PlateauResult]]:
    """
    Full plateau analysis for one strategy's result DataFrame.

    Returns: p1_vals, p2_vals, objective_grid, plateau_scores, ranked_candidates
    """
    df = df.copy()
    df["Objective"] = compute_objective(df)

    p1_vals, p2_vals, grid = build_objective_grid(df, cfg)
    plateau_scores = compute_plateau_scores(grid, radius)
    mean_scores, std_scores = compute_neighborhood_stats(grid, radius)
    candidates = rank_plateau_candidates(
        df, cfg, plateau_scores, mean_scores, std_scores,
        p1_vals, p2_vals, top_n,
    )
    return p1_vals, p2_vals, grid, plateau_scores, candidates
