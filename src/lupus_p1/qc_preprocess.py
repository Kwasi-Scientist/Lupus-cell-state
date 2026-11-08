"""Step B: QC metrics, light filtering, normalization, HVG, PCA, Harmony, kNN, and UMAP."""

from __future__ import annotations

import gc
import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.logging_utils import log_memory
from src.lupus_p1.subsample import resolve_celltype_column


def map_var_names_to_gene_symbols(adata: ad.AnnData, logger: logging.Logger) -> ad.AnnData:
    """Map Ensembl gene IDs in adata.var_names to HGNC gene symbols using `feature_name` if needed.

    # TODO(verify): Confirm exact var column name holding HGNC gene symbols in CELLxGENE Census
    # / source H5AD (`feature_name` vs `gene_symbols` vs `gene_short_name`).
    """
    sample_names = [str(x) for x in adata.var_names[: min(50, adata.n_vars)]]
    looks_like_ensembl = sum(name.startswith("ENSG") for name in sample_names) > (
        0.5 * len(sample_names)
    )

    symbol_candidates = ("feature_name", "gene_symbols", "gene_symbol", "gene_short_name", "symbol")
    chosen_col: str | None = None
    for col in symbol_candidates:
        if col in adata.var.columns:
            chosen_col = col
            break

    if looks_like_ensembl and chosen_col is not None:
        logger.info(
            "Mapping Ensembl var_names to HGNC gene symbols using adata.var[%r]...",
            chosen_col,
        )
        adata.var["ensembl_id"] = adata.var_names.astype(str)
        symbols = adata.var[chosen_col].astype(str).str.strip()
        valid_mask = (symbols != "") & (symbols.str.lower() != "nan")
        new_names = np.where(valid_mask, symbols, adata.var["ensembl_id"])
        adata.var_names = pd.Index(new_names, name="gene_symbol")
        adata.var_names_make_unique()
    elif looks_like_ensembl and chosen_col is None:
        raise RuntimeError(
            "adata.var_names appear to be Ensembl IDs (ENSG...), but none of the expected symbol "
            f"columns {symbol_candidates} were found in adata.var. Available var columns: "
            f"{sorted(adata.var.columns)}"
        )
    else:
        adata.var_names_make_unique()
        logger.info("adata.var_names already appear to be gene symbols (sample: %s).", sample_names[:5])

    return adata


def run_harmony_integration(
    adata: ad.AnnData,
    batch_col: str,
    seed: int,
    logger: logging.Logger,
) -> bool:
    """Run Harmony integration on `adata.obsm['X_pca']` and store `adata.obsm['X_pca_harmony']`.

    # TODO(verify): Confirm harmonypy.run_harmony return object attribute (`.Z_corr`) and
    # matrix orientation (`n_pcs x n_cells` vs `n_cells x n_pcs`) in harmonypy==0.0.10.
    """
    if batch_col not in adata.obs.columns:
        logger.info("Batch column %r not present in adata.obs; skipping Harmony.", batch_col)
        return False

    n_batches = int(adata.obs[batch_col].nunique(dropna=True))
    if n_batches < 2:
        logger.info(
            "Batch column %r has only %d unique value(s); skipping Harmony.",
            batch_col,
            n_batches,
        )
        return False

    import harmonypy as hm

    logger.info(
        "Running Harmony integration on X_pca with batch_key=%r (%d batches, random_state=%d)...",
        batch_col,
        n_batches,
        seed,
    )
    pca_mat = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    meta_df = adata.obs[[batch_col]].copy()
    meta_df[batch_col] = meta_df[batch_col].astype(str)

    ho = hm.run_harmony(
        pca_mat,
        meta_df,
        vars_use=[batch_col],
        random_state=seed,
        verbose=False,
    )
    z_corr = np.asarray(ho.Z_corr)
    if z_corr.shape == (pca_mat.shape[1], pca_mat.shape[0]) and pca_mat.shape[0] != pca_mat.shape[1]:
        z_corr = z_corr.T
    elif z_corr.shape[0] != adata.n_obs and z_corr.shape[1] == adata.n_obs:
        z_corr = z_corr.T

    if z_corr.shape != pca_mat.shape:
        raise RuntimeError(
            f"Unexpected shape from harmonypy Z_corr: {z_corr.shape} (expected {pca_mat.shape})."
        )

    adata.obsm["X_pca_harmony"] = z_corr.astype(np.float32)
    return True


def preprocess_and_qc_anndata(
    adata: ad.AnnData,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Perform QC metrics calculation, light filtering, normalization, HVG, PCA, Harmony, kNN, and UMAP."""
    sc.settings.NJOBS = cfg.n_jobs
    adata = map_var_names_to_gene_symbols(adata, logger)

    resolved_ct_col = resolve_celltype_column(adata.obs, cfg.columns.celltype)
    adata.uns["resolved_celltype_column"] = resolved_ct_col

    adata.var["mt"] = adata.var_names.astype(str).str.upper().str.startswith("MT-")
    n_mt_genes = int(adata.var["mt"].sum())
    logger.info("Identified %d mitochondrial (MT-) genes.", n_mt_genes)

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    n_cells_initial = int(adata.n_obs)
    n_genes_initial = int(adata.n_vars)
    obs_min_genes_before = float(adata.obs["n_genes_by_counts"].min()) if n_cells_initial else 0.0
    obs_max_mt_before = float(adata.obs["pct_counts_mt"].max()) if n_cells_initial else 0.0

    looks_prefiltered = (
        obs_min_genes_before >= cfg.qc.min_genes and obs_max_mt_before <= cfg.qc.max_pct_mt
    )
    logger.info(
        "Pre-filter inspection: min(n_genes_by_counts)=%.1f, max(pct_counts_mt)=%.2f%% -> looks_prefiltered=%s",
        obs_min_genes_before,
        obs_max_mt_before,
        looks_prefiltered,
    )

    keep_min_genes = adata.obs["n_genes_by_counts"] >= cfg.qc.min_genes
    n_removed_min_genes = int((~keep_min_genes).sum())
    adata = adata[keep_min_genes].copy()

    keep_mt = adata.obs["pct_counts_mt"] <= cfg.qc.max_pct_mt
    n_removed_max_mt = int((~keep_mt).sum())
    adata = adata[keep_mt].copy()

    n_genes_before_gene_filter = int(adata.n_vars)
    sc.pp.filter_genes(adata, min_cells=cfg.qc.min_cells_per_gene)
    n_removed_genes = n_genes_before_gene_filter - int(adata.n_vars)

    logger.info(
        "QC filtering summary: cells %d -> %d (removed %d by min_genes<%d, %d by pct_counts_mt>%.1f%%); "
        "genes %d -> %d (removed %d by min_cells<%d).",
        n_cells_initial,
        adata.n_obs,
        n_removed_min_genes,
        cfg.qc.min_genes,
        n_removed_max_mt,
        cfg.qc.max_pct_mt,
        n_genes_initial,
        adata.n_vars,
        n_removed_genes,
        cfg.qc.min_cells_per_gene,
    )

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    hvg_batch_key: str | None = None
    if cfg.columns.batch and cfg.columns.batch in adata.obs.columns:
        hvg_batch_key = cfg.columns.batch
    logger.info(
        "Selecting %d highly variable genes (flavor='seurat', batch_key=%r)...",
        cfg.preprocess.n_top_genes,
        hvg_batch_key,
    )
    n_top = min(cfg.preprocess.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        batch_key=hvg_batch_key,
        flavor="seurat",
    )

    n_pcs = min(cfg.preprocess.n_pcs, adata.n_obs - 1, int(adata.var["highly_variable"].sum()) - 1)
    n_pcs = max(2, n_pcs)
    logger.info("Running PCA with n_comps=%d, random_state=%d...", n_pcs, cfg.seed)
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True, random_state=cfg.seed)

    used_harmony = False
    if cfg.preprocess.use_harmony and cfg.columns.batch:
        used_harmony = run_harmony_integration(
            adata=adata,
            batch_col=cfg.columns.batch,
            seed=cfg.seed,
            logger=logger,
        )
    else:
        logger.info(
            "Skipping Harmony integration (preprocess.use_harmony=%s, columns.batch=%r).",
            cfg.preprocess.use_harmony,
            cfg.columns.batch,
        )

    rep_used = "X_pca_harmony" if used_harmony else "X_pca"
    adata.uns["neighbors_use_rep"] = rep_used
    logger.info(
        "Computing kNN neighborhood graph (use_rep=%r, n_neighbors=%d, random_state=%d)...",
        rep_used,
        cfg.preprocess.n_neighbors,
        cfg.seed,
    )
    sc.pp.neighbors(
        adata,
        use_rep=rep_used,
        n_neighbors=cfg.preprocess.n_neighbors,
        random_state=cfg.seed,
    )

    logger.info("Computing UMAP embedding (random_state=%d)...", cfg.seed)
    sc.tl.umap(adata, random_state=cfg.seed)

    qc_summary_df = pd.DataFrame(
        [
            {
                "n_cells_initial": n_cells_initial,
                "n_genes_initial": n_genes_initial,
                "min_genes_observed_before_filter": obs_min_genes_before,
                "max_pct_mt_observed_before_filter": obs_max_mt_before,
                "data_looked_prefiltered": looks_prefiltered,
                "min_genes_threshold": cfg.qc.min_genes,
                "max_pct_mt_threshold": cfg.qc.max_pct_mt,
                "min_cells_per_gene_threshold": cfg.qc.min_cells_per_gene,
                "n_cells_removed_min_genes": n_removed_min_genes,
                "n_cells_removed_max_pct_mt": n_removed_max_mt,
                "n_genes_removed_min_cells": n_removed_genes,
                "n_cells_retained": int(adata.n_obs),
                "n_genes_retained": int(adata.n_vars),
                "n_mt_genes_detected": n_mt_genes,
                "n_hvg_selected": int(adata.var["highly_variable"].sum()),
                "n_pcs": n_pcs,
                "harmony_used": used_harmony,
                "neighbors_representation": rep_used,
                "resolved_celltype_column": resolved_ct_col,
            }
        ]
    )

    gc.collect()
    return adata, qc_summary_df


def run_qc_preprocess_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute Step B (`qc`): load subsampled raw H5AD, run QC & preprocessing, and save processed H5AD."""
    raw_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_raw.h5ad"
    if not raw_h5ad_path.is_file():
        raise FileNotFoundError(
            f"Raw subsample H5AD not found at {raw_h5ad_path}. Run `--step subsample` first."
        )

    logger.info("Loading raw subsampled AnnData from %s...", raw_h5ad_path)
    adata = ad.read_h5ad(raw_h5ad_path)
    log_memory(logger, "after loading raw subsampled AnnData")

    adata, qc_summary_df = preprocess_and_qc_anndata(adata, cfg, logger)

    qc_csv_path = cfg.outputs_path / "tables" / "qc_filter_summary.csv"
    qc_summary_df.to_csv(qc_csv_path, index=False)
    logger.info("Saved QC filter summary table to %s", qc_csv_path)

    proc_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_processed.h5ad"
    adata.write_h5ad(proc_h5ad_path, compression="gzip")
    logger.info("Saved processed subsample AnnData to %s", proc_h5ad_path)
    log_memory(logger, "end of qc_preprocess step")
