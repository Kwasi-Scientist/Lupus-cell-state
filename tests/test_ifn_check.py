"""Unit tests for donor-level ISG aggregation, Mann-Whitney U test, BH adjustment, and rank-biserial sign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lupus_p1.ifn_check import (
    compute_donor_celltype_means,
    resolve_disease_labels,
    run_donor_level_ifn_tests,
)


def test_ifn_donor_level_aggregation_and_effect_sign(synthetic_obs_df: pd.DataFrame) -> None:
    obs = synthetic_obs_df.copy()
    rng = np.random.default_rng(77)

    is_sle = (obs["disease"] == "systemic lupus erythematosus").to_numpy()
    obs["isg_score"] = rng.normal(loc=0.0, scale=0.3, size=len(obs)) + np.where(is_sle, 1.5, 0.0)

    sle_label, normal_label = resolve_disease_labels(obs["disease"])
    assert sle_label == "systemic lupus erythematosus"
    assert normal_label == "normal"

    donor_means = compute_donor_celltype_means(
        obs,
        score_col="isg_score",
        donor_col="donor_id",
        disease_col="disease",
        celltype_col="cell_type",
        ancestry_col="self_reported_ethnicity",
        min_cells_per_donor_celltype=10,
    )

    assert (donor_means["n_cells"] >= 10).all()
    assert not donor_means.duplicated(subset=["donor_id", "cell_type"]).any()

    stats_df = run_donor_level_ifn_tests(
        donor_means,
        disease_col="disease",
        celltype_col="cell_type",
        ancestry_col="self_reported_ethnicity",
        sle_label=sle_label,
        normal_label=normal_label,
    )
    overall = stats_df[stats_df["stratum"] == "all_ancestries"]
    assert not overall.empty
    assert "p_adj_bh" in overall.columns
    assert overall["p_adj_bh"].notna().all()

    assert (overall["diff_median_sle_minus_normal"] > 0).all()
    assert (overall["rank_biserial"] > 0).all()
    assert (overall["n_donors_sle"] <= 10).all()
    assert (overall["n_donors_normal"] <= 10).all()
