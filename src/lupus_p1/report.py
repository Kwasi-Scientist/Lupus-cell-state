"""Step F: Automated templated Markdown run report populated strictly from computed tables."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.provenance import read_provenance


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-flavored Markdown table without extra dependencies."""
    if df.empty:
        return "_No rows available._"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        formatted_cells: list[str] = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, float):
                if abs(val) > 0 and (abs(val) < 1e-3 or abs(val) >= 1e5):
                    formatted_cells.append(f"{val:.3e}")
                else:
                    formatted_cells.append(f"{val:.4f}")
            else:
                formatted_cells.append(str(val))
        body_lines.append("| " + " | ".join(formatted_cells) + " |")
    return "\n".join([header, sep, *body_lines])


def generate_phase1_report(cfg: PipelineConfig) -> Path:
    """Assemble `outputs/reports/phase1_run_report.md` from provenance.json and CSV tables."""
    out = cfg.outputs_path
    tables_dir = out / "tables"
    prov_path = out / "provenance" / "provenance.json"
    report_path = out / "reports" / "phase1_run_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    prov = read_provenance(prov_path)
    disease_df = pd.read_csv(tables_dir / "metadata_summary_disease.csv")
    ancestry_df = pd.read_csv(tables_dir / "metadata_summary_ancestry.csv")
    subsample_df = pd.read_csv(tables_dir / "subsample_counts.csv")
    qc_df = pd.read_csv(tables_dir / "qc_filter_summary.csv")
    scdrs_df = pd.read_csv(tables_dir / "scdrs_celltype_combined.csv")
    ifn_stats_df = pd.read_csv(tables_dir / "ifn_donor_level_stats.csv")

    med_csv = tables_dir / "metadata_medication_columns.csv"
    med_df = pd.read_csv(med_csv) if med_csv.is_file() else pd.DataFrame()

    prov_rows = []
    for r in prov.get("resources", []):
        prov_rows.append(
            {
                "Resource": r.get("resource_name"),
                "Dataset/Collection": r.get("dataset_id") or r.get("collection_id") or r.get("geo_accession") or "-",
                "Census Build": r.get("census_build_resolved") or r.get("census_build_requested") or "-",
                "Bytes": r.get("file_size_bytes") if r.get("file_size_bytes") is not None else "-",
                "MD5 Match": r.get("provider_md5_match") if r.get("provider_md5_match") is not None else "-",
            }
        )
    prov_table_df = pd.DataFrame(prov_rows)

    sub_ct_df = subsample_df[subsample_df["stratum_variable"] == "cell_type"][
        [
            "stratum_value",
            "cells_before",
            "cells_after",
            "pct_cells_before",
            "pct_cells_after",
            "donors_before",
            "donors_after",
        ]
    ].sort_values("cells_after", ascending=False)

    height_sub = scdrs_df[scdrs_df["trait_role"] == "height"]
    sig_height = height_sub[height_sub["assoc_mcp_bh"] < 0.05]
    if sig_height.empty:
        calibration_banner = (
            "**PASS**: Negative-control trait `height` showed zero cell types with "
            "BH-adjusted association $p < 0.05$."
        )
    else:
        sig_list = ", ".join(sig_height["group"].astype(str).tolist())
        calibration_banner = (
            f"**WARNING — CALIBRATION INVESTIGATION NEEDED**: Negative-control trait `height` "
            f"showed BH-adjusted association $p < 0.05$ in: {sig_list}."
        )

    scdrs_cols = [
        c
        for c in (
            "trait_role",
            "trait",
            "group",
            "n_cell",
            "assoc_mcp",
            "assoc_mcp_bh",
            "assoc_mcz",
            "hetero_mcp",
            "hetero_mcz",
            "n_fdr_0.1",
            "frac_fdr_0.1",
        )
        if c in scdrs_df.columns
    ]
    scdrs_display_df = scdrs_df[scdrs_cols].copy()

    lines: list[str] = [
        "# Phase 1 Run Report: Lupus Cell States to Drugs (`lupus_p1`)",
        "",
        f"- **Generated Timestamp (UTC)**: `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Global Seed**: `{cfg.seed}`",
        f"- **Acquisition Backend**: `{cfg.cellxgene.backend}`",
        f"- **cellxgene-census version**: `{prov.get('cellxgene_census_version')}`",
        f"- **tiledbsoma version**: `{prov.get('tiledbsoma_version')}`",
        "",
        "## 1. Data & Version Provenance",
        "",
        _df_to_markdown_table(prov_table_df),
        "",
        "## 2. Full-Dataset Metadata Summary (Step A)",
        "",
        "### Disease Distribution (Full Dataset)",
        _df_to_markdown_table(disease_df),
        "",
        "### Self-Reported Ethnicity / Ancestry Distribution (Full Dataset)",
        _df_to_markdown_table(ancestry_df),
        "",
        "### Medication / Treatment Columns Detected in `.obs`",
        _df_to_markdown_table(med_df),
        "",
        "## 3. Subsampling Summary (Donor-Balanced, Cell-Type-Floored)",
        "",
        _df_to_markdown_table(sub_ct_df),
        "",
        "## 4. Quality Control & Preprocessing Summary (Step B)",
        "",
        _df_to_markdown_table(qc_df),
        "",
        "## 5. Polygenic GWAS Enrichment (`scDRS`, Step C)",
        "",
        f"### Negative-Control Calibration Check (`height`)\n{calibration_banner}",
        "",
        "### Cell-Type Association, Heterogeneity, and FDR<0.1 Cell Fractions",
        _df_to_markdown_table(scdrs_display_df),
        "",
        "## 6. Type-I Interferon (ISG) Known-Answer Check (Donor-Level, Step D)",
        "",
        _df_to_markdown_table(ifn_stats_df),
        "",
        "## 7. Scientific & Methodological Caveats",
        "",
        "1. **Subsampling Variance**: Phase 1 analyzes a donor-balanced, cell-type-floored subsample "
        f"(target `{cfg.subsample.target_cells:,}` cells) to remain within ~5 GB RAM. Rerunning with "
        "additional seeds via `subsample.seeds_for_stability` is recommended to verify stability.",
        "2. **Batch and Ancestry Confounding**: Disease status and donor ancestry can correlate with "
        "processing cohorts/pools. Cross-tabulations are logged in `outputs/tables/metadata_summary_disease_x_batch.csv`.",
        "3. **European-Biased GWAS Gene Sets**: The scDRS 74-trait MAGMA gene sets (`030122`) derive "
        "predominantly from European-ancestry GWAS summary statistics, whereas Perez et al. 2022 includes "
        "both European and East Asian donors.",
        "4. **Medication & Clinical Heterogeneity**: Observational clinical cohorts reflect heterogeneous "
        "treatment regimens (e.g., hydroxychloroquine, corticosteroids, immunosuppressants) and flare states.",
        "5. **CELLxGENE Census Standardized Labels vs. Source H5AD**: The `census` backend provides standardized "
        "Cell Ontology (`cell_type`) labels and core schema covariates; author-specific processing batch or "
        "subcluster annotations require running with `cellxgene.backend: 'h5ad'`.",
        "",
        "## 8. Proposed Phase 2 Roadmap",
        "",
        "1. **Donor-Aware Differential Abundance & State Shifts (`pertpy` — CNA / Milo)**: Quantify fine-grained "
        "neighborhood abundance shifts within scDRS-enriched compartments while controlling for donor, batch, and ancestry.",
        "2. **Single-Cell eQTL Anchoring (`OneK1K`)**: Intersect top scDRS-weighted lupus genes in enriched cell states "
        "with cell-type-specific cis-eQTLs from OneK1K to establish directional genetic effects.",
        "3. **Transcriptional Signature Reversal & Drug-Target Prioritization (`LINCS L1000` & `PrimeKG`)**: Derive "
        "donor-level differential expression signatures from genetics-anchored lupus cell states and query LINCS L1000 "
        "perturbation signatures and PrimeKG drug-target knowledge graphs for candidate therapeutics.",
        "",
    ]

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return report_path


def run_report_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute Step F (`report`): build `outputs/reports/phase1_run_report.md` from computed tables."""
    report_path = generate_phase1_report(cfg)
    logger.info("Generated automated Phase 1 run report at %s", report_path)
