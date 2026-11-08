"""Deterministic donor-balanced, cell-type-floored subsampling on obs metadata."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.download import fetch_expression_for_subsample
from src.lupus_p1.logging_utils import log_memory


def resolve_celltype_column(obs_df: pd.DataFrame, configured_col: str) -> str:
    """Resolve the cell-type column in obs_df, falling back to 'cell_type' if needed."""
    if configured_col in obs_df.columns:
        return configured_col
    if "cell_type" in obs_df.columns:
        return "cell_type"
    raise ValueError(
        f"Neither configured celltype column {configured_col!r} nor fallback 'cell_type' "
        f"exists in obs_df. Available columns: {sorted(obs_df.columns)}"
    )


def select_subsample(
    obs_df: pd.DataFrame,
    *,
    donor_col: str = "donor_id",
    celltype_col: str = "cell_type",
    target_cells: int = 80000,
    max_cells_per_donor: int | None = None,
    min_cells_per_type: int = 200,
    seed: int = 0,
) -> tuple[pd.Index, dict[str, Any]]:
    """Pure, deterministic donor-balanced and cell-type-floored subsampling on an obs DataFrame."""
    if obs_df.empty:
        raise ValueError("Cannot subsample an empty obs_df.")
    for col in (donor_col, celltype_col):
        if col not in obs_df.columns:
            raise ValueError(f"Column {col!r} not found in obs_df.")

    rng = np.random.default_rng(seed)

    work_df = obs_df[[donor_col, celltype_col]].copy()
    work_df["_orig_pos"] = np.arange(len(work_df))
    work_df["_index_str"] = work_df.index.astype(str)
    work_df = work_df.sort_values(["_index_str", "_orig_pos"])

    donors = sorted(work_df[donor_col].astype(str).unique())
    n_donors = len(donors)
    if n_donors == 0:
        raise ValueError("No donors found in obs_df.")

    initial_quota = math.ceil(target_cells / n_donors)
    effective_cap = (
        min(initial_quota, max_cells_per_donor)
        if max_cells_per_donor is not None
        else initial_quota
    )

    donor_groups: dict[str, np.ndarray] = {}
    for d_id, grp in work_df.groupby(work_df[donor_col].astype(str), sort=True):
        indices = grp.index.to_numpy()
        perm = rng.permutation(len(indices))
        donor_groups[str(d_id)] = indices[perm]

    donor_taken_count: dict[str, int] = {d: 0 for d in donors}
    for d in donors:
        avail = len(donor_groups[d])
        take = min(avail, effective_cap)
        donor_taken_count[d] = take

    current_total = sum(donor_taken_count.values())
    hard_donor_cap = max_cells_per_donor if max_cells_per_donor is not None else len(work_df)
    while current_total < target_cells:
        eligible_donors = [
            d
            for d in donors
            if donor_taken_count[d] < min(len(donor_groups[d]), hard_donor_cap)
        ]
        if not eligible_donors:
            break
        deficit = target_cells - current_total
        step_quota = max(1, math.ceil(deficit / len(eligible_donors)))
        added = 0
        for d in eligible_donors:
            rem_cap = min(len(donor_groups[d]), hard_donor_cap) - donor_taken_count[d]
            add_d = min(rem_cap, step_quota, target_cells - current_total)
            donor_taken_count[d] += add_d
            current_total += add_d
            added += add_d
            if current_total >= target_cells:
                break
        if added == 0:
            break

    selected_set: set[Any] = set()
    for d in donors:
        k = donor_taken_count[d]
        if k > 0:
            selected_set.update(donor_groups[d][:k].tolist())

    celltypes = sorted(work_df[celltype_col].astype(str).unique())
    floor_topups: dict[str, int] = {}

    for ct in celltypes:
        ct_df = work_df[work_df[celltype_col].astype(str) == ct]
        ct_total_avail = len(ct_df)
        desired_floor = min(min_cells_per_type, ct_total_avail)
        currently_selected_ct = [idx for idx in ct_df.index if idx in selected_set]
        deficit_ct = desired_floor - len(currently_selected_ct)
        if deficit_ct <= 0:
            floor_topups[ct] = 0
            continue

        unselected_ct_df = ct_df.loc[[idx for idx in ct_df.index if idx not in selected_set]]
        ct_donor_pools: dict[str, list[Any]] = {}
        for d_id, grp in unselected_ct_df.groupby(unselected_ct_df[donor_col].astype(str), sort=True):
            arr = grp.index.to_numpy()
            perm = rng.permutation(len(arr))
            ct_donor_pools[str(d_id)] = arr[perm].tolist()

        added_ct = 0
        pool_donors = sorted(ct_donor_pools.keys())
        while added_ct < deficit_ct and pool_donors:
            next_pool_donors: list[str] = []
            for d_id in pool_donors:
                if added_ct >= deficit_ct:
                    break
                if ct_donor_pools[d_id]:
                    chosen_idx = ct_donor_pools[d_id].pop(0)
                    selected_set.add(chosen_idx)
                    added_ct += 1
                if ct_donor_pools[d_id]:
                    next_pool_donors.append(d_id)
            pool_donors = next_pool_donors

        floor_topups[ct] = added_ct

    n_trimmed = 0
    excess = len(selected_set) - target_cells
    if excess > 0:
        selected_df = work_df.loc[work_df.index.isin(selected_set)]
        ct_selected_lists: dict[str, list[Any]] = {}
        for ct, grp in selected_df.groupby(selected_df[celltype_col].astype(str), sort=True):
            arr = grp.index.to_numpy()
            perm = rng.permutation(len(arr))
            ct_selected_lists[str(ct)] = arr[perm].tolist()

        while excess > 0:
            trim_candidates = [
                ct for ct, lst in ct_selected_lists.items() if len(lst) > min_cells_per_type
            ]
            if not trim_candidates:
                break
            largest_ct = max(trim_candidates, key=lambda c: (len(ct_selected_lists[c]), c))
            ct_selected_lists[largest_ct].pop()
            excess -= 1
            n_trimmed += 1

        selected_set = {idx for lst in ct_selected_lists.values() for idx in lst}

    final_mask = obs_df.index.isin(selected_set)
    selected_index = obs_df.index[final_mask]

    n_selected = len(selected_index)
    pct_diff = ((n_selected - target_cells) / max(target_cells, 1)) * 100.0
    if abs(pct_diff) <= 2.0:
        status_reason = (
            f"Final subsample size ({n_selected}) is within ±2% of target ({target_cells}; diff={pct_diff:+.2f}%)."
        )
    elif n_selected < target_cells:
        status_reason = (
            f"Final subsample size ({n_selected}) is below target ({target_cells}; diff={pct_diff:+.2f}%) "
            "due to total cell availability or per-donor cap constraints."
        )
    else:
        status_reason = (
            f"Final subsample size ({n_selected}) exceeds target ({target_cells}; diff={pct_diff:+.2f}%) "
            "to satisfy strict per-cell-type floor constraints."
        )

    info = {
        "n_total_before": len(obs_df),
        "n_selected": n_selected,
        "target_cells": target_cells,
        "n_donors_before": n_donors,
        "n_donors_after": int(obs_df.loc[selected_index, donor_col].nunique()),
        "initial_per_donor_quota": initial_quota,
        "effective_per_donor_cap": effective_cap,
        "min_cells_per_type": min_cells_per_type,
        "floor_topups_by_celltype": floor_topups,
        "n_trimmed": n_trimmed,
        "status_reason": status_reason,
    }
    return selected_index, info


def build_subsample_comparison_table(
    obs_before: pd.DataFrame,
    selected_index: pd.Index,
    *,
    donor_col: str,
    disease_col: str,
    celltype_col: str,
    ancestry_col: str,
) -> pd.DataFrame:
    """Build a unified table of cell and donor counts before vs. after subsampling across key strata."""
    obs_after = obs_before.loc[selected_index]
    rows: list[dict[str, Any]] = []

    strata_specs = [
        ("cell_type", celltype_col),
        ("disease", disease_col),
        ("ancestry", ancestry_col),
        ("donor", donor_col),
    ]

    for stratum_label, col_name in strata_specs:
        if col_name not in obs_before.columns:
            continue
        before_counts = obs_before[col_name].astype(str).value_counts()
        after_counts = obs_after[col_name].astype(str).value_counts()
        before_donors = obs_before.groupby(obs_before[col_name].astype(str))[donor_col].nunique()
        after_donors = obs_after.groupby(obs_after[col_name].astype(str))[donor_col].nunique()

        all_levels = sorted(set(before_counts.index).union(set(after_counts.index)))
        for lvl in all_levels:
            b_c = int(before_counts.get(lvl, 0))
            a_c = int(after_counts.get(lvl, 0))
            rows.append(
                {
                    "stratum_variable": stratum_label,
                    "stratum_value": lvl,
                    "cells_before": b_c,
                    "cells_after": a_c,
                    "pct_cells_before": (b_c / max(len(obs_before), 1)) * 100.0,
                    "pct_cells_after": (a_c / max(len(obs_after), 1)) * 100.0,
                    "donors_before": int(before_donors.get(lvl, 0)),
                    "donors_after": int(after_donors.get(lvl, 0)),
                }
            )

    return pd.DataFrame(rows)


def run_subsample_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute the `subsample` step: select subsample from full obs, save count tables, and fetch expression."""
    obs_parquet = cfg.outputs_path / "data" / "obs_full_metadata.parquet"
    if not obs_parquet.is_file():
        raise FileNotFoundError(
            f"Full metadata file {obs_parquet} not found. Run `--step fetch` first."
        )

    obs_df = pd.read_parquet(obs_parquet)
    celltype_col = resolve_celltype_column(obs_df, cfg.columns.celltype)

    selected_idx, info = select_subsample(
        obs_df,
        donor_col=cfg.columns.donor,
        celltype_col=celltype_col,
        target_cells=cfg.subsample.target_cells,
        max_cells_per_donor=cfg.subsample.max_cells_per_donor,
        min_cells_per_type=cfg.subsample.min_cells_per_type,
        seed=cfg.seed,
    )
    logger.info("Subsampling result: %s", info["status_reason"])
    logger.info(
        "Subsampled %d -> %d cells across %d -> %d donors (initial quota=%d, trimmed=%d).",
        info["n_total_before"],
        info["n_selected"],
        info["n_donors_before"],
        info["n_donors_after"],
        info["initial_per_donor_quota"],
        info["n_trimmed"],
    )

    comp_df = build_subsample_comparison_table(
        obs_df,
        selected_idx,
        donor_col=cfg.columns.donor,
        disease_col=cfg.columns.disease,
        celltype_col=celltype_col,
        ancestry_col=cfg.columns.ancestry,
    )
    counts_csv_path = cfg.outputs_path / "tables" / "subsample_counts.csv"
    comp_df.to_csv(counts_csv_path, index=False)
    logger.info("Saved before/after subsample count table to %s", counts_csv_path)

    if cfg.subsample.seeds_for_stability:
        stab_rows: list[dict[str, Any]] = []
        base_set = set(selected_idx)
        for alt_seed in cfg.subsample.seeds_for_stability:
            alt_idx, alt_info = select_subsample(
                obs_df,
                donor_col=cfg.columns.donor,
                celltype_col=celltype_col,
                target_cells=cfg.subsample.target_cells,
                max_cells_per_donor=cfg.subsample.max_cells_per_donor,
                min_cells_per_type=cfg.subsample.min_cells_per_type,
                seed=alt_seed,
            )
            alt_set = set(alt_idx)
            jaccard = len(base_set & alt_set) / max(len(base_set | alt_set), 1)
            stab_rows.append(
                {
                    "seed": alt_seed,
                    "n_selected": alt_info["n_selected"],
                    "jaccard_overlap_with_primary_seed": jaccard,
                }
            )
        stab_df = pd.DataFrame(stab_rows)
        stab_path = cfg.outputs_path / "tables" / "subsample_seed_stability.csv"
        stab_df.to_csv(stab_path, index=False)
        logger.info("Saved subsample stability check across %d seeds to %s", len(stab_rows), stab_path)

    selected_obs = obs_df.loc[selected_idx].copy()
    selected_obs_path = cfg.outputs_path / "data" / "subsample_selected_obs.parquet"
    selected_obs.to_parquet(selected_obs_path, index=True)

    adata_sub = fetch_expression_for_subsample(selected_obs, cfg, logger)
    raw_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_raw.h5ad"
    adata_sub.write_h5ad(raw_h5ad_path, compression="gzip")
    logger.info(
        "Saved raw subsampled AnnData (%d cells x %d genes) to %s",
        adata_sub.n_obs,
        adata_sub.n_vars,
        raw_h5ad_path,
    )
    log_memory(logger, "end of subsample step")
