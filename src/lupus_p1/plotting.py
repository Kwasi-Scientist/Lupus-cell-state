"""Step E: Publication-quality figures (PNG at 300 DPI + PDF, colorblind-safe palettes)."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.logging_utils import log_memory
from src.lupus_p1.subsample import resolve_celltype_column

OKABE_ITO_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]


def _save_fig_png_and_pdf(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Save figure as both 300 DPI PNG and vector PDF, then close figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / f"{stem}.png"
    pdf_path = fig_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_qc_violins(adata: ad.AnnData, fig_dir: Path) -> None:
    """Generate `qc_violin.png`/`.pdf` showing genes per cell, total UMIs, and % mitochondrial."""
    metrics = [
        ("n_genes_by_counts", "Detected Genes per Cell"),
        ("total_counts", "Total UMI Counts per Cell"),
        ("pct_counts_mt", "Mitochondrial Reads (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    n_cells = adata.n_obs
    for ax, (col, title) in zip(axes, metrics):
        vals = adata.obs[col].to_numpy(dtype=float)
        sns.violinplot(y=vals, ax=ax, color="#56B4E9", inner="quartile", cut=0)
        ax.set_title(f"{title}\n(n = {n_cells:,} cells)", fontsize=10.5)
        ax.set_ylabel(col)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Single-Cell Quality Control Metrics (Subsample)", fontsize=12, y=1.03)
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, "qc_violin")


def plot_umap_categorical(
    adata: ad.AnnData,
    col: str,
    title: str,
    stem: str,
    fig_dir: Path,
) -> None:
    """Plot a 2D UMAP colored by a categorical obs column using a colorblind-safe palette."""
    coords = np.asarray(adata.obsm["X_umap"])
    cats = adata.obs[col].astype(str)
    unique_cats = sorted(cats.unique())

    palette = (
        OKABE_ITO_PALETTE
        if len(unique_cats) <= len(OKABE_ITO_PALETTE)
        else sns.color_palette("husl", len(unique_cats))
    )
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    rng = np.random.default_rng(0)
    order = rng.permutation(adata.n_obs)
    coords_shuf = coords[order]
    cats_shuf = cats.iloc[order]

    for c in unique_cats:
        mask = (cats_shuf == c).to_numpy()
        ax.scatter(
            coords_shuf[mask, 0],
            coords_shuf[mask, 1],
            s=1.8,
            alpha=0.7,
            color=color_map[c],
            label=f"{c} (n={int(mask.sum()):,})",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{title} (n = {adata.n_obs:,} cells)", fontsize=11.5)
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        markerscale=4.0,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, stem)


def plot_umap_continuous(
    adata: ad.AnnData,
    col: str,
    title: str,
    cbar_label: str,
    stem: str,
    fig_dir: Path,
    cmap: str = "viridis",
) -> None:
    """Plot a 2D UMAP colored by a continuous score in adata.obs."""
    coords = np.asarray(adata.obsm["X_umap"])
    vals = adata.obs[col].to_numpy(dtype=float)
    vmin, vmax = np.nanpercentile(vals, [1.0, 99.0])

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    order = np.argsort(vals)
    sc_plot = ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=vals[order],
        s=2.0,
        alpha=0.75,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    cbar = fig.colorbar(sc_plot, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{title} (n = {adata.n_obs:,} cells)", fontsize=11.5)
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, stem)


def plot_subsample_composition(subsample_counts_csv: Path, fig_dir: Path) -> None:
    """Plot bar charts comparing cell-type and disease composition before vs. after subsampling."""
    df = pd.read_csv(subsample_counts_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [2.2, 1.0]})

    ct_df = df[df["stratum_variable"] == "cell_type"].sort_values("pct_cells_before", ascending=True)
    y_pos = np.arange(len(ct_df))
    height = 0.38
    axes[0].barh(
        y_pos - height / 2,
        ct_df["pct_cells_before"],
        height=height,
        color="#0072B2",
        label="Full Dataset (Before)",
    )
    axes[0].barh(
        y_pos + height / 2,
        ct_df["pct_cells_after"],
        height=height,
        color="#E69F00",
        label="Subsample (After)",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ct_df["stratum_value"], fontsize=8.5)
    axes[0].set_xlabel("Percentage of Cells (%)")
    axes[0].set_title("Cell-Type Composition: Full vs. Subsample")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    dis_df = df[df["stratum_variable"] == "disease"].sort_values("pct_cells_before", ascending=False)
    x_pos = np.arange(len(dis_df))
    width = 0.35
    axes[1].bar(
        x_pos - width / 2,
        dis_df["pct_cells_before"],
        width=width,
        color="#0072B2",
        label="Full Dataset (Before)",
    )
    axes[1].bar(
        x_pos + width / 2,
        dis_df["pct_cells_after"],
        width=width,
        color="#E69F00",
        label="Subsample (After)",
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(dis_df["stratum_value"], rotation=20, ha="right")
    axes[1].set_ylabel("Percentage of Cells (%)")
    axes[1].set_title("Disease Status: Full vs. Subsample")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, "subsample_composition")


def plot_scdrs_celltype_bars(
    combined_df: pd.DataFrame,
    role_key: str,
    fig_dir: Path,
) -> None:
    """Plot -log10(association MC p) per cell type for a single trait."""
    sub = combined_df[combined_df["trait_role"] == role_key].copy()
    if sub.empty:
        return
    trait_name = str(sub["trait"].iloc[0])
    sub["neg_log10_p"] = -np.log10(sub["assoc_mcp"].clip(lower=1e-10))
    sub = sub.sort_values("neg_log10_p", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.38 * len(sub))))
    colors = [
        "#D55E00" if q < 0.05 else "#0072B2" for q in sub["assoc_mcp_bh"].to_numpy(dtype=float)
    ]
    labels = [
        f"{row['group']} (n={int(row['n_cell']):,})" for _, row in sub.iterrows()
    ]
    ax.barh(labels, sub["neg_log10_p"], color=colors)
    ax.axvline(-np.log10(0.05), color="#333333", linestyle="--", linewidth=1.1, label="Nominal p = 0.05")
    ax.set_xlabel(r"$-\log_{10}(\mathrm{scDRS\ Association\ MC\ } p)$")
    ax.set_title(f"scDRS Cell-Type Association: {trait_name} ({role_key})\n(Orange = BH FDR < 0.05)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, f"scdrs_celltype_{role_key}")


def plot_scdrs_sle_vs_height(combined_df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot side-by-side comparison of scDRS cell-type enrichment for SLE vs. Height."""
    sle_df = combined_df[combined_df["trait_role"] == "sle"].copy()
    hgt_df = combined_df[combined_df["trait_role"] == "height"].copy()
    if sle_df.empty or hgt_df.empty:
        return

    merged = pd.merge(
        sle_df[["group", "n_cell", "assoc_mcp", "assoc_mcz", "assoc_mcp_bh"]],
        hgt_df[["group", "assoc_mcp", "assoc_mcz", "assoc_mcp_bh"]],
        on="group",
        suffixes=("_sle", "_height"),
    )
    merged["neg_log10_p_sle"] = -np.log10(merged["assoc_mcp_sle"].clip(lower=1e-10))
    merged["neg_log10_p_height"] = -np.log10(merged["assoc_mcp_height"].clip(lower=1e-10))
    merged = merged.sort_values("neg_log10_p_sle", ascending=True)

    y_pos = np.arange(len(merged))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.42 * len(merged))))
    ax.barh(
        y_pos + height / 2,
        merged["neg_log10_p_sle"],
        height=height,
        color="#D55E00",
        label="SLE (GWAS)",
    )
    ax.barh(
        y_pos - height / 2,
        merged["neg_log10_p_height"],
        height=height,
        color="#0072B2",
        label="Height (Negative Control)",
    )
    ax.axvline(-np.log10(0.05), color="#333333", linestyle="--", linewidth=1.1, label="Nominal p = 0.05")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['group']} (n={int(r['n_cell']):,})" for _, r in merged.iterrows()],
        fontsize=8.5,
    )
    ax.set_xlabel(r"$-\log_{10}(\mathrm{scDRS\ Association\ MC\ } p)$")
    ax.set_title("scDRS Cell-Type Enrichment: SLE vs. Height Negative Control")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, "scdrs_sle_vs_height")


def plot_ifn_donor_boxplots(
    donor_means_csv: Path,
    disease_col: str,
    celltype_col: str,
    fig_dir: Path,
) -> None:
    """Plot donor-level mean ISG score boxplots + individual donor points per cell type."""
    df = pd.read_csv(donor_means_csv)
    if df.empty:
        return

    order = (
        df.groupby(celltype_col)["mean_isg_score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    diseases = sorted(df[disease_col].astype(str).unique())
    palette = {d: OKABE_ITO_PALETTE[i % len(OKABE_ITO_PALETTE)] for i, d in enumerate(diseases)}

    fig, ax = plt.subplots(figsize=(max(9.0, 0.75 * len(order)), 5.8))
    sns.boxplot(
        data=df,
        x=celltype_col,
        y="mean_isg_score",
        hue=disease_col,
        order=order,
        palette=palette,
        fliersize=0,
        boxprops={"alpha": 0.65},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=celltype_col,
        y="mean_isg_score",
        hue=disease_col,
        order=order,
        palette=palette,
        dodge=True,
        size=4.2,
        alpha=0.85,
        legend=False,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Author Cell Type")
    ax.set_ylabel("Donor-Level Mean ISG Module Score")
    ax.set_title("Donor-Level Type-I IFN (ISG) Module Score by Cell Type and Disease")
    ax.legend(title=disease_col, frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_fig_png_and_pdf(fig, fig_dir, "ifn_isg_by_celltype_disease")


def run_plotting_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute Step E (`figures`): generate all required PNG (300 DPI) and PDF figures."""
    fig_dir = cfg.outputs_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    proc_h5ad_path = cfg.outputs_path / "data" / "sle_pbmc_subsample_processed.h5ad"
    if not proc_h5ad_path.is_file():
        raise FileNotFoundError(
            f"Processed AnnData not found at {proc_h5ad_path}. Run earlier steps first."
        )

    logger.info("Loading processed AnnData from %s for figure generation...", proc_h5ad_path)
    adata = ad.read_h5ad(proc_h5ad_path)
    celltype_col = resolve_celltype_column(adata.obs, cfg.columns.celltype)

    plot_qc_violins(adata, fig_dir)

    plot_umap_categorical(adata, celltype_col, "UMAP by Author Cell Type", "umap_celltype", fig_dir)
    plot_umap_categorical(
        adata, cfg.columns.disease, "UMAP by Disease Status", "umap_disease", fig_dir
    )
    if cfg.columns.ancestry in adata.obs.columns:
        plot_umap_categorical(
            adata, cfg.columns.ancestry, "UMAP by Donor Ancestry", "umap_ancestry", fig_dir
        )
    if cfg.columns.batch and cfg.columns.batch in adata.obs.columns:
        plot_umap_categorical(
            adata, cfg.columns.batch, "UMAP by Batch / Cohort", "umap_batch", fig_dir
        )

    subsample_counts_csv = cfg.outputs_path / "tables" / "subsample_counts.csv"
    if subsample_counts_csv.is_file():
        plot_subsample_composition(subsample_counts_csv, fig_dir)

    combined_csv = cfg.outputs_path / "tables" / "scdrs_celltype_combined.csv"
    if combined_csv.is_file():
        combined_df = pd.read_csv(combined_csv)
        for role_key in combined_df["trait_role"].unique():
            score_col = f"scdrs_norm_score_{role_key}"
            if score_col in adata.obs.columns:
                plot_umap_continuous(
                    adata,
                    col=score_col,
                    title=f"scDRS Normalized Score: {role_key.upper()}",
                    cbar_label="scDRS Normalized Score",
                    stem=f"scdrs_umap_{role_key}",
                    fig_dir=fig_dir,
                    cmap="viridis",
                )
            plot_scdrs_celltype_bars(combined_df, role_key=str(role_key), fig_dir=fig_dir)

        plot_scdrs_sle_vs_height(combined_df, fig_dir)

    donor_means_csv = cfg.outputs_path / "tables" / "ifn_donor_means.csv"
    if donor_means_csv.is_file():
        plot_ifn_donor_boxplots(
            donor_means_csv=donor_means_csv,
            disease_col=cfg.columns.disease,
            celltype_col=celltype_col,
            fig_dir=fig_dir,
        )

    if "isg_score" in adata.obs.columns:
        plot_umap_continuous(
            adata,
            col="isg_score",
            title="Type-I Interferon (ISG) Module Score",
            cbar_label="ISG Module Score",
            stem="umap_isg_score",
            fig_dir=fig_dir,
            cmap="magma",
        )

    logger.info("All figures generated in %s (PNG 300 DPI + PDF).", fig_dir)
    log_memory(logger, "end of figures step")
