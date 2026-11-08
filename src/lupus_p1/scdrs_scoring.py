"""Step C: scDRS GWAS trait selection, covariate construction, per-trait cell scoring, and group analysis."""

from __future__ import annotations

import gc
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from statsmodels.stats.multitest import multipletests

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.logging_utils import log_memory
from src.lupus_p1.provenance import read_provenance
from src.lupus_p1.subsample import resolve_celltype_column


def match_trait_by_regex(
    available_traits: list[str] | tuple[str, ...],
    pattern: str,
    label: str = "trait",
) -> str:
    """Select exactly one trait name from `available_traits` matching `pattern`."""
    regex = re.compile(pattern)
    matches = [t for t in available_traits if regex.search(str(t))]
    if len(matches) == 0:
        raise ValueError(
            f"Zero traits matched {label} regex {pattern!r}. "
            f"Available traits ({len(available_traits)}): {sorted(available_traits)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple traits ({len(matches)}: {matches}) matched {label} regex {pattern!r}. "
            f"Refine the regex in config.yaml so it matches exactly one trait. "
            f"Available traits ({len(available_traits)}): {sorted(available_traits)}"
        )
    return matches[0]


def select_scdrs_traits(
    available_traits: list[str] | tuple[str, ...],
    sle_regex: str,
    height_regex: str,
    extra_trait_regexes: list[str] | None = None,
) -> dict[str, str]:
    """Resolve SLE, height (negative control), and any optional extra traits by regex."""
    resolved: dict[str, str] = {}
    resolved["sle"] = match_trait_by_regex(available_traits, sle_regex, label="SLE")
    resolved["height"] = match_trait_by_regex(
        available_traits, height_regex, label="height (negative control)"
    )

    for idx, extra_re in enumerate(extra_trait_regexes or []):
        key = f"extra_{idx + 1}"
        resolved[key] = match_trait_by_regex(available_traits, extra_re, label=key)

    return resolved


def resolve_gs_file_path(cfg: PipelineConfig) -> Path:
    """Locate the downloaded scDRS .gs file from provenance.json or cache_dir."""
    prov_path = cfg.outputs_path / "provenance" / "provenance.json"
    if prov_path.is_file():
        prov = read_provenance(prov_path)
        for res in prov.get("resources", []):
            if res.get("resource_name") == "scdrs_figshare_gs" and res.get("local_file"):
                p = Path(res["local_file"])
                if p.is_file():
                    return p

    scdrs_cache = cfg.cache_path / "scdrs"
    if scdrs_cache.is_dir():
        gs_files = sorted(scdrs_cache.glob("*.gs"))
        if gs_files:
            return gs_files[0]

    raise FileNotFoundError(
        "Could not locate downloaded scDRS .gs file in provenance.json or data_cache/scdrs/. "
        "Run `--step fetch` first."
    )


def build_scdrs_covariate_df(
    obs_df: pd.DataFrame,
    batch_col: str | None,
    use_batch_covariates: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build covariate matrix for scdrs.preprocess: intercept (const), n_genes, and optional batch indicators."""
    cov_df = pd.DataFrame(index=obs_df.index)
    cov_df["const"] = 1.0

    if "n_genes_by_counts" in obs_df.columns:
        cov_df["n_genes"] = obs_df["n_genes_by_counts"].astype(float)
    elif "n_genes" in obs_df.columns:
        cov_df["n_genes"] = obs_df["n_genes"].astype(float)
    else:
        raise RuntimeError(
            "Neither 'n_genes_by_counts' nor 'n_genes' found in adata.obs for scDRS covariates."
        )

    if use_batch_covariates and batch_col and batch_col in obs_df.columns:
        batch_series = obs_df[batch_col].astype(str)
        if batch_series.nunique() > 1:
            dummies = pd.get_dummies(batch_series, prefix="batch", drop_first=True, dtype=float)
            for col in dummies.columns:
                if dummies[col].std() > 0:
                    cov_df[col] = dummies[col]
            logger.info(
                "Added %d one-hot batch covariate column(s) from %r to scDRS covariates.",
                dummies.shape[1],
                batch_col,
            )

    return cov_df


def prepare_adata_for_scdrs(adata_proc: ad.AnnData, logger: logging.Logger) -> ad.AnnData:
    """Construct an AnnData holding raw counts normalized across ALL genes (normalize_total + log1p) for scDRS."""
    if "counts" not in adata_proc.layers:
        raise RuntimeError(
            "adata.layers['counts'] is missing from processed AnnData; required for scDRS normalization."
        )

    adata_scdrs = ad.AnnData(
        X=adata_proc.layers["counts"].copy(),
        obs=adata_proc.obs.copy(),
        var=adata_proc.var[[]].copy(),
        obsp={"connectivities": adata_proc.obsp["connectivities"].copy()}
        if "connectivities" in adata_proc.obsp
        else {},
        uns=adata_proc.uns.copy(),
    )
    logger.info(
        "Normalizing full gene set (%d cells x %d genes) with normalize_total(1e4) + log1p for scDRS...",
        adata_scdrs.n_obs,
        adata_scdrs.n_vars,
    )
    sc.pp.normalize_total(adata_scdrs, target_sum=1e4)
    sc.pp.log1p(adata_scdrs)
    return adata_scdrs


def _run_scdrs_python_mode(
    adata_proc: ad.AnnData,
    gs_path: Path,
    cov_df: pd.DataFrame,
    celltype_col: str,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Run scDRS via its documented Python API one trait at a time.

    # TODO(verify): Confirm exact signatures and return column names of:
    # - scdrs.util.load_gs(gs_path, src_species="human", dst_species="human", to_intersect=adata.var_names)
    # - scdrs.preprocess(adata, cov=df_cov, n_mean_bin=20, n_var_bin=20, copy=False)
    # - scdrs.score_cell(adata, gene_list, gene_weight=gene_weight, ctrl_match_key="mean_var",
    #                    n_ctrl=1000, weight_opt="vs", return_ctrl_raw_score=False,
    #                    return_ctrl_norm_score=True, verbose=False)
    # - scdrs.method.downstream_group_analysis(adata, df_full_score, group_cols=[celltype_column])
    # in scdrs==1.0.2.
    """
    import scdrs

    adata_scdrs = prepare_adata_for_scdrs(adata_proc, logger)
    log_memory(logger, "after preparing normalized full-gene AnnData for scDRS")

    logger.info("Loading scDRS gene sets from %s ...", gs_path)
    dict_gs = scdrs.util.load_gs(
        str(gs_path),
        src_species="human",
        dst_species="human",
        to_intersect=adata_scdrs.var_names,
    )
    available_traits = sorted(dict_gs.keys())
    logger.info("Loaded %d traits from .gs file: %s", len(available_traits), available_traits)

    selected_traits = select_scdrs_traits(
        available_traits=available_traits,
        sle_regex=cfg.scdrs.traits.sle_regex,
        height_regex=cfg.scdrs.traits.height_regex,
        extra_trait_regexes=cfg.scdrs.extra_traits,
    )
    logger.info("Resolved scDRS target traits: %s", selected_traits)

    logger.info("Running scdrs.preprocess(adata, cov=cov_df, n_mean_bin=20, n_var_bin=20, copy=False)...")
    scdrs.preprocess(
        adata_scdrs,
        cov=cov_df,
        n_mean_bin=20,
        n_var_bin=20,
        copy=False,
    )
    log_memory(logger, "after scdrs.preprocess")

    scdrs_data_dir = cfg.outputs_path / "data" / "scdrs"
    scdrs_data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.outputs_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)

    combined_group_dfs: list[pd.DataFrame] = []

    for role_key, trait_name in selected_traits.items():
        gene_list, gene_weights = dict_gs[trait_name]
        logger.info(
            "Scoring trait [%s] %r (%d intersecting genes, n_ctrl=%d)...",
            role_key,
            trait_name,
            len(gene_list),
            cfg.scdrs.n_ctrl,
        )
        np.random.seed(cfg.seed)
        df_full_score = scdrs.score_cell(
            adata_scdrs,
            gene_list,
            gene_weight=gene_weights,
            ctrl_match_key="mean_var",
            n_ctrl=cfg.scdrs.n_ctrl,
            weight_opt="vs",
            return_ctrl_raw_score=False,
            return_ctrl_norm_score=True,
            verbose=False,
        )

        pval_col = "mc_pval" if "mc_pval" in df_full_score.columns else "pval"
        _, cell_qvals, _, _ = multipletests(
            df_full_score[pval_col].to_numpy(dtype=float), method="fdr_bh"
        )
        df_full_score["fdr"] = cell_qvals

        safe_trait = re.sub(r"[^A-Za-z0-9_.-]+", "_", trait_name)
        full_score_gz = scdrs_data_dir / f"{safe_trait}.full_score.gz"
        df_full_score.to_csv(full_score_gz, sep="\t", index=True, compression="gzip")

        summary_cols = [
            c
            for c in ("raw_score", "norm_score", "mc_pval", "pval", "nlog10_pval", "zscore", "fdr")
            if c in df_full_score.columns
        ]
        score_parquet = scdrs_data_dir / f"{safe_trait}.score.parquet"
        df_full_score[summary_cols].to_parquet(score_parquet, index=True)

        adata_proc.obs[f"scdrs_norm_score_{role_key}"] = df_full_score["norm_score"].astype(
            np.float32
        )
        adata_proc.obs[f"scdrs_zscore_{role_key}"] = df_full_score["zscore"].astype(np.float32)
        adata_proc.obs[f"scdrs_fdr_{role_key}"] = df_full_score["fdr"].astype(np.float32)

        logger.info(
            "Running scdrs.method.downstream_group_analysis for trait %r grouped by %r...",
            trait_name,
            celltype_col,
        )
        dict_group = scdrs.method.downstream_group_analysis(
            adata=adata_scdrs,
            df_full_score=df_full_score,
            group_cols=[celltype_col],
        )
        df_group = dict_group[celltype_col].copy()
        df_group.index.name = "group"
        df_group = df_group.reset_index()

        if "n_fdr_0.1" in df_group.columns and "n_cell" in df_group.columns:
            df_group["frac_fdr_0.1"] = df_group["n_fdr_0.1"] / df_group["n_cell"].clip(lower=1)
        else:
            ct_series = adata_scdrs.obs[celltype_col].astype(str)
            fdr_pass = (df_full_score["fdr"] < cfg.scdrs.fdr_cell).groupby(ct_series).agg(["sum", "count"])
            df_group["n_fdr_0.1"] = df_group["group"].map(fdr_pass["sum"]).fillna(0.0)
            df_group["frac_fdr_0.1"] = df_group["group"].map(
                fdr_pass["sum"] / fdr_pass["count"].clip(lower=1)
            ).fillna(0.0)

        if "assoc_mcp" in df_group.columns:
            _, assoc_bh, _, _ = multipletests(
                df_group["assoc_mcp"].to_numpy(dtype=float), method="fdr_bh"
            )
            df_group["assoc_mcp_bh"] = assoc_bh

        df_group["trait_role"] = role_key
        df_group["trait"] = trait_name
        df_group = df_group.sort_values(["assoc_mcp", "group"], ascending=[True, True]).reset_index(
            drop=True
        )

        role_csv = tables_dir / f"scdrs_celltype_{role_key}.csv"
        trait_csv = tables_dir / f"scdrs_celltype_{safe_trait}.csv"
        df_group.to_csv(role_csv, index=False)
        if trait_csv.resolve() != role_csv.resolve():
            df_group.to_csv(trait_csv, index=False)

        combined_group_dfs.append(df_group)

        del df_full_score
        gc.collect()
        log_memory(logger, f"after scoring and freeing trait {trait_name!r}")

    del adata_scdrs
    gc.collect()

    combined_df = pd.concat(combined_group_dfs, ignore_index=True)
    return selected_traits, combined_df


def _run_scdrs_cli_mode(
    adata_proc: ad.AnnData,
    gs_path: Path,
    cov_df: pd.DataFrame,
    celltype_col: str,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Fallback CLI wrapper calling `scdrs compute-score` and `scdrs perform-downstream` via subprocess.

    # TODO(verify): Confirm exact CLI flag names (`--h5ad-file`, `--gs-file`, `--cov-file`,
    # `--group-analysis`, `--out-folder`) with `scdrs compute-score --help` in scdrs==1.0.2.
    """
    gs_df = pd.read_csv(gs_path, sep="\t")
    if "TRAIT" not in gs_df.columns or "GENESET" not in gs_df.columns:
        raise RuntimeError(f"Expected TRAIT and GENESET columns in {gs_path}, got {list(gs_df.columns)}")

    available_traits = sorted(gs_df["TRAIT"].astype(str).unique().tolist())
    selected_traits = select_scdrs_traits(
        available_traits=available_traits,
        sle_regex=cfg.scdrs.traits.sle_regex,
        height_regex=cfg.scdrs.traits.height_regex,
        extra_trait_regexes=cfg.scdrs.extra_traits,
    )

    scdrs_dir = cfg.outputs_path / "data" / "scdrs"
    scdrs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.outputs_path / "tables"

    sub_gs_df = gs_df[gs_df["TRAIT"].isin(selected_traits.values())].copy()
    sub_gs_path = scdrs_dir / "selected_traits.gs"
    sub_gs_df.to_csv(sub_gs_path, sep="\t", index=False)

    cov_path = scdrs_dir / "covariates.tsv"
    cov_df.to_csv(cov_path, sep="\t", index=True)

    adata_scdrs = prepare_adata_for_scdrs(adata_proc, logger)
    temp_h5ad = scdrs_dir / "temp_scdrs_input.h5ad"
    adata_scdrs.write_h5ad(temp_h5ad)
    del adata_scdrs
    gc.collect()

    cmd_score = [
        "scdrs",
        "compute-score",
        "--h5ad-file",
        str(temp_h5ad),
        "--h5ad-species",
        "human",
        "--gs-file",
        str(sub_gs_path),
        "--gs-species",
        "human",
        "--cov-file",
        str(cov_path),
        "--ctrl-match-opt",
        "mean_var",
        "--n-ctrl",
        str(cfg.scdrs.n_ctrl),
        "--weight-opt",
        "vs",
        "--return-ctrl-raw-score",
        "False",
        "--return-ctrl-norm-score",
        "True",
        "--out-folder",
        str(scdrs_dir),
    ]
    logger.info("Executing scDRS CLI compute-score: %s", " ".join(cmd_score))
    subprocess.run(cmd_score, check=True)

    combined_dfs: list[pd.DataFrame] = []
    for role_key, trait_name in selected_traits.items():
        full_score_gz = scdrs_dir / f"{trait_name}.full_score.gz"
        cmd_down = [
            "scdrs",
            "perform-downstream",
            "--h5ad-file",
            str(temp_h5ad),
            "--score-file",
            str(full_score_gz),
            "--group-analysis",
            celltype_col,
            "--out-folder",
            str(scdrs_dir),
        ]
        logger.info("Executing scDRS CLI perform-downstream: %s", " ".join(cmd_down))
        subprocess.run(cmd_down, check=True)

        group_file = scdrs_dir / f"{trait_name}.scdrs_group.{celltype_col}"
        df_group = pd.read_csv(group_file, sep="\t")
        if "n_fdr_0.1" in df_group.columns and "n_cell" in df_group.columns:
            df_group["frac_fdr_0.1"] = df_group["n_fdr_0.1"] / df_group["n_cell"].clip(lower=1)
        _, assoc_bh, _, _ = multipletests(df_group["assoc_mcp"].to_numpy(dtype=float), method="fdr_bh")
        df_group["assoc_mcp_bh"] = assoc_bh
        df_group["trait_role"] = role_key
        df_group["trait"] = trait_name
        df_group.to_csv(tables_dir / f"scdrs_celltype_{role_key}.csv", index=False)
        combined_dfs.append(df_group)

    if temp_h5ad.is_file():
        temp_h5ad.unlink()

    return selected_traits, pd.concat(combined_dfs, ignore_index=True)


def check_height_calibration(combined_df: pd.DataFrame, logger: logging.Logger) -> dict[str, Any]:
    """Check whether any cell type is significant (BH-adjusted assoc_mcp < 0.05) for height negative control."""
    height_df = combined_df[combined_df["trait_role"] == "height"].copy()
    if height_df.empty:
        return {"height_evaluated": False, "calibration_warning": False, "significant_cell_types": []}

    sig_df = height_df[height_df["assoc_mcp_bh"] < 0.05]
    sig_types = sig_df["group"].astype(str).tolist()
    if sig_types:
        logger.warning(
            "CALIBRATION WARNING: Negative-control trait 'height' showed BH-significant association "
            "(assoc_mcp_bh < 0.05) in %d cell type(s): %s. Calibration needs investigation!",
            len(sig_types),
            sig_types,
        )
        return {
            "height_evaluated": True,
            "calibration_warning": True,
            "significant_cell_types": sig_types,
        }

    logger.info(
        "Calibration check passed: 0 cell types had assoc_mcp_bh < 0.05 for negative-control trait 'height'."
    )
    return {
        "height_evaluated": True,
        "calibration_warning": False,
        "significant_cell_types": [],
    }


def run_scdrs_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute Step C (`scdrs`): score SLE and height traits and run downstream cell-type group tests."""
    proc_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_processed.h5ad"
    if not proc_h5ad_path.is_file():
        raise FileNotFoundError(
            f"Processed AnnData not found at {proc_h5ad_path}. Run `--step qc` first."
        )

    gs_path = resolve_gs_file_path(cfg)
    logger.info("Loading processed subsample AnnData from %s ...", proc_h5ad_path)
    adata_proc = ad.read_h5ad(proc_h5ad_path)
    celltype_col = resolve_celltype_column(adata_proc.obs, cfg.columns.celltype)

    cov_df = build_scdrs_covariate_df(
        obs_df=adata_proc.obs,
        batch_col=cfg.columns.batch,
        use_batch_covariates=cfg.scdrs.batch_covariates,
        logger=logger,
    )

    if cfg.scdrs.mode == "python":
        selected_traits, combined_df = _run_scdrs_python_mode(
            adata_proc=adata_proc,
            gs_path=gs_path,
            cov_df=cov_df,
            celltype_col=celltype_col,
            cfg=cfg,
            logger=logger,
        )
    else:
        selected_traits, combined_df = _run_scdrs_cli_mode(
            adata_proc=adata_proc,
            gs_path=gs_path,
            cov_df=cov_df,
            celltype_col=celltype_col,
            cfg=cfg,
            logger=logger,
        )

    cal_info = check_height_calibration(combined_df, logger)
    adata_proc.uns["scdrs_selected_traits"] = selected_traits
    adata_proc.uns["scdrs_height_calibration"] = cal_info

    combined_df = combined_df.sort_values(
        ["trait_role", "assoc_mcp", "group"], ascending=[True, True, True]
    ).reset_index(drop=True)
    combined_csv_path = cfg.outputs_path / "tables" / "scdrs_celltype_combined.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info("Saved combined scDRS cell-type table to %s", combined_csv_path)

    adata_proc.write_h5ad(proc_h5ad_path, compression="gzip")
    logger.info("Updated %s with per-cell scDRS columns.", proc_h5ad_path)
    log_memory(logger, "end of scdrs step")
