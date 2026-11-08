"""Unit tests for deterministic donor-balanced, cell-type-floored subsampling."""

from __future__ import annotations

import pandas as pd

from src.lupus_p1.subsample import build_subsample_comparison_table, select_subsample


def test_subsample_target_size_and_determinism(synthetic_obs_df: pd.DataFrame) -> None:
    target = 1000
    idx1, info1 = select_subsample(
        synthetic_obs_df,
        donor_col="donor_id",
        celltype_col="cell_type",
        target_cells=target,
        min_cells_per_type=40,
        seed=0,
    )
    idx2, _ = select_subsample(
        synthetic_obs_df,
        donor_col="donor_id",
        celltype_col="cell_type",
        target_cells=target,
        min_cells_per_type=40,
        seed=0,
    )
    idx_diff_seed, _ = select_subsample(
        synthetic_obs_df,
        donor_col="donor_id",
        celltype_col="cell_type",
        target_cells=target,
        min_cells_per_type=40,
        seed=99,
    )

    assert idx1.equals(idx2)
    assert not idx1.equals(idx_diff_seed)
    assert abs(len(idx1) - target) <= int(0.02 * target)
    assert info1["n_donors_after"] == info1["n_donors_before"]


def test_subsample_respects_per_cell_type_floor(synthetic_obs_df: pd.DataFrame) -> None:
    total_plasmablasts = int((synthetic_obs_df["cell_type"] == "plasmablast").sum())
    floor = min(50, total_plasmablasts)

    selected_idx, _ = select_subsample(
        synthetic_obs_df,
        donor_col="donor_id",
        celltype_col="cell_type",
        target_cells=800,
        min_cells_per_type=50,
        seed=7,
    )
    sub_obs = synthetic_obs_df.loc[selected_idx]
    ct_counts = sub_obs["cell_type"].value_counts()

    assert int(ct_counts.get("plasmablast", 0)) >= floor
    for ct in synthetic_obs_df["cell_type"].unique():
        avail = int((synthetic_obs_df["cell_type"] == ct).sum())
        assert int(ct_counts.get(ct, 0)) >= min(50, avail)


def test_subsample_comparison_table(synthetic_obs_df: pd.DataFrame) -> None:
    selected_idx, _ = select_subsample(
        synthetic_obs_df,
        donor_col="donor_id",
        celltype_col="cell_type",
        target_cells=800,
        min_cells_per_type=30,
        seed=0,
    )
    comp_df = build_subsample_comparison_table(
        synthetic_obs_df,
        selected_idx,
        donor_col="donor_id",
        disease_col="disease",
        celltype_col="cell_type",
        ancestry_col="self_reported_ethnicity",
    )
    assert not comp_df.empty
    assert set(comp_df["stratum_variable"].unique()) == {
        "cell_type",
        "disease",
        "ancestry",
        "donor",
    }
