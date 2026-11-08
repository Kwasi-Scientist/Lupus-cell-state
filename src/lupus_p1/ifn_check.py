"""Step D: IFN-stimulated gene (ISG) module score and donor-level Mann-Whitney U comparisons."""

from __future__ import annotations

import logging
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.logging_utils import log_memory
from src.lupus_p1.subsample import resolve_celltype_column


def resolve_disease_labels(
    disease_series: pd.Series,
    normal_override: str | None = None,
    sle_regex: str = r"(?i)lupus|\bSLE\b",
) -> tuple[str, str]:
    """Auto-detect the exact 'normal' and 'SLE' labels in `adata.obs[disease_col]`."""
    unique_labels = sorted(disease_series.dropna().astype(str).unique().tolist())
    if not unique_labels:
        raise ValueError("Disease column contains no non-null values.")

    if normal_override is not None:
        if normal_override not in unique_labels:
            raise ValueError(
                f"Configured normal_disease_label={normal_override!r} not found in disease column. "
                f"Available labels: {unique_labels}"
            )
        normal_label = normal_override
    else:
        normal_matches = [
            lbl for lbl in unique_labels if re.search(r"(?i)^normal$|healthy|control", lbl)
        ]
        if len(normal_matches) != 1:
            raise ValueError(
                f"Could not uniquely auto-detect normal/healthy label from {unique_labels} "
                f"(matches: {normal_matches}). Set columns.normal_disease_label in config.yaml."
            )
        normal_label = normal_matches[0]

    sle_matches = [
        lbl for lbl in unique_labels if lbl != normal_label and re.search(sle_regex, lbl)
    ]
    if len(sle_matches) != 1:
        raise ValueError(
            f"Could not uniquely detect SLE disease label matching {sle_regex!r} from {unique_labels} "
            f"(matches: {sle_matches})."
        )
    sle_label = sle_matches[0]
    return sle_label, normal_label


def compute_donor_celltype_means(
    obs_df: pd.DataFrame,
    *,
    score_col: str = "isg_score",
    donor_col: str = "donor_id",
    disease_col: str = "disease",
    celltype_col: str = "cell_type",
    ancestry_col: str | None = "self_reported_ethnicity",
    min_cells_per_donor_celltype: int = 10,
) -> pd.DataFrame:
    """Aggregate cell-level `isg_score` to donor-level means within each cell type."""
    required = [score_col, donor_col, disease_col, celltype_col]
    for c in required:
        if c not in obs_df.columns:
            raise ValueError(f"Required column {c!r} missing from obs_df.")

    group_cols = [donor_col, celltype_col, disease_col]
    has_ancestry = ancestry_col is not None and ancestry_col in obs_df.columns
    if has_ancestry and ancestry_col not in group_cols:
        group_cols.append(ancestry_col)

    work = obs_df[group_cols + [score_col]].copy()
    for c in group_cols:
        work[c] = work[c].astype(str)
    work[score_col] = work[score_col].astype(float)

    agg_df = (
        work.groupby(group_cols, observed=True)
        .agg(
            n_cells=(score_col, "size"),
            mean_isg_score=(score_col, "mean"),
            median_cell_isg_score=(score_col, "median"),
        )
        .reset_index()
    )

    filtered_df = agg_df[agg_df["n_cells"] >= min_cells_per_donor_celltype].copy()
    filtered_df = filtered_df.sort_values([celltype_col, disease_col, donor_col]).reset_index(
        drop=True
    )
    return filtered_df


def _mann_whitney_and_rank_biserial(
    sle_vals: np.ndarray,
    normal_vals: np.ndarray,
) -> dict[str, float]:
    """Compute two-sided Mann-Whitney U test and rank-biserial effect size (SLE vs normal)."""
    n_sle = len(sle_vals)
    n_norm = len(normal_vals)
    res = mannwhitneyu(sle_vals, normal_vals, alternative="two-sided", method="auto")
    u_stat = float(res.statistic)
    pval = float(res.pvalue)
    r_rb = (2.0 * u_stat / float(n_sle * n_norm)) - 1.0
    med_sle = float(np.median(sle_vals))
    med_norm = float(np.median(normal_vals))
    return {
        "n_donors_sle": n_sle,
        "n_donors_normal": n_norm,
        "median_sle": med_sle,
        "median_normal": med_norm,
        "diff_median_sle_minus_normal": med_sle - med_norm,
        "mean_sle": float(np.mean(sle_vals)),
        "mean_normal": float(np.mean(normal_vals)),
        "u_statistic": u_stat,
        "rank_biserial": r_rb,
        "p_value": pval,
    }


def run_donor_level_ifn_tests(
    donor_means_df: pd.DataFrame,
    *,
    disease_col: str = "disease",
    celltype_col: str = "cell_type",
    ancestry_col: str | None = "self_reported_ethnicity",
    sle_label: str,
    normal_label: str,
    min_donors_per_group_overall: int = 2,
    min_donors_per_group_stratum: int = 5,
) -> pd.DataFrame:
    """Run donor-level Mann-Whitney U tests per cell type (overall and stratified by ancestry when n >= 5)."""
    rows: list[dict[str, Any]] = []
    celltypes = sorted(donor_means_df[celltype_col].astype(str).unique())

    for ct in celltypes:
        sub = donor_means_df[donor_means_df[celltype_col] == ct]
        sle_vals = sub.loc[sub[disease_col] == sle_label, "mean_isg_score"].to_numpy(dtype=float)
        norm_vals = sub.loc[sub[disease_col] == normal_label, "mean_isg_score"].to_numpy(
            dtype=float
        )
        if len(sle_vals) < min_donors_per_group_overall or len(norm_vals) < min_donors_per_group_overall:
            continue
        stats = _mann_whitney_and_rank_biserial(sle_vals, norm_vals)
        rows.append(
            {
                "stratum": "all_ancestries",
                "cell_type": ct,
                **stats,
            }
        )

    if ancestry_col and ancestry_col in donor_means_df.columns:
        ancestries = sorted(donor_means_df[ancestry_col].astype(str).unique())
        for anc in ancestries:
            anc_df = donor_means_df[donor_means_df[ancestry_col] == anc]
            for ct in celltypes:
                sub = anc_df[anc_df[celltype_col] == ct]
                sle_vals = sub.loc[sub[disease_col] == sle_label, "mean_isg_score"].to_numpy(
                    dtype=float
                )
                norm_vals = sub.loc[sub[disease_col] == normal_label, "mean_isg_score"].to_numpy(
                    dtype=float
                )
                if (
                    len(sle_vals) < min_donors_per_group_stratum
                    or len(norm_vals) < min_donors_per_group_stratum
                ):
                    continue
                stats = _mann_whitney_and_rank_biserial(sle_vals, norm_vals)
                rows.append(
                    {
                        "stratum": f"ancestry:{anc}",
                        "cell_type": ct,
                        **stats,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "stratum",
                "cell_type",
                "n_donors_sle",
                "n_donors_normal",
                "median_sle",
                "median_normal",
                "diff_median_sle_minus_normal",
                "mean_sle",
                "mean_normal",
                "u_statistic",
                "rank_biserial",
                "p_value",
                "p_adj_bh",
            ]
        )

    stats_df = pd.DataFrame(rows)
    stats_df["p_adj_bh"] = np.nan

    for stratum_val, grp in stats_df.groupby("stratum"):
        _, qvals, _, _ = multipletests(grp["p_value"].to_numpy(dtype=float), method="fdr_bh")
        stats_df.loc[grp.index, "p_adj_bh"] = qvals

    stats_df = stats_df.sort_values(["stratum", "p_value", "cell_type"]).reset_index(drop=True)
    return stats_df


def run_ifn_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute Step D (`ifn`): score ISG module genes and run donor-level Mann-Whitney U tests."""
    proc_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_processed.h5ad"
    if not proc_h5ad_path.is_file():
        raise FileNotFoundError(
            f"Processed AnnData not found at {proc_h5ad_path}. Run `--step qc` first."
        )

    logger.info("Loading processed AnnData from %s for IFN known-answer check...", proc_h5ad_path)
    adata = ad.read_h5ad(proc_h5ad_path)
    celltype_col = resolve_celltype_column(adata.obs, cfg.columns.celltype)

    requested_genes = cfg.ifn.active_genes
    present_genes = [g for g in requested_genes if g in adata.var_names]
    missing_genes = [g for g in requested_genes if g not in adata.var_names]
    logger.info(
        "ISG module genes: %d/%d present in adata.var_names (present=%s, missing=%s).",
        len(present_genes),
        len(requested_genes),
        present_genes,
        missing_genes,
    )
    if not present_genes:
        raise RuntimeError(
            f"None of the requested ISG genes {requested_genes} were found in adata.var_names. "
            "Check gene symbol mapping in Step B."
        )

    sc.tl.score_genes(
        adata,
        gene_list=present_genes,
        score_name="isg_score",
        random_state=cfg.seed,
    )
    adata.uns["ifn_genes_requested"] = requested_genes
    adata.uns["ifn_genes_present"] = present_genes
    adata.uns["ifn_genes_missing"] = missing_genes

    sle_label, normal_label = resolve_disease_labels(
        adata.obs[cfg.columns.disease],
        normal_override=cfg.columns.normal_disease_label,
        sle_regex=cfg.columns.sle_disease_regex,
    )
    logger.info("Resolved disease labels for IFN test: SLE=%r vs normal=%r", sle_label, normal_label)

    donor_means_df = compute_donor_celltype_means(
        adata.obs,
        score_col="isg_score",
        donor_col=cfg.columns.donor,
        disease_col=cfg.columns.disease,
        celltype_col=celltype_col,
        ancestry_col=cfg.columns.ancestry,
        min_cells_per_donor_celltype=cfg.ifn.min_cells_per_donor_celltype,
    )

    stats_df = run_donor_level_ifn_tests(
        donor_means_df,
        disease_col=cfg.columns.disease,
        celltype_col=celltype_col,
        ancestry_col=cfg.columns.ancestry,
        sle_label=sle_label,
        normal_label=normal_label,
    )

    tables_dir = cfg.outputs_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    means_csv = tables_dir / "ifn_donor_means.csv"
    stats_csv = tables_dir / "ifn_donor_level_stats.csv"

    donor_means_df.to_csv(means_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Saved donor-level ISG means (%d rows) to %s", len(donor_means_df), means_csv)
    logger.info("Saved donor-level ISG statistics (%d rows) to %s", len(stats_df), stats_csv)

    adata.write_h5ad(proc_h5ad_path, compression="gzip")
    log_memory(logger, "end of ifn step")
