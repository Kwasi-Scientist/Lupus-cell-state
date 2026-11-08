# Lupus Cell States to Drugs — Phase 1 (`lupus_p1`)

## Purpose and Scientific Context

Systemic lupus erythematosus (SLE) is a chronic autoimmune disease characterized by a prominent type-I interferon (IFN) transcriptional signature in peripheral blood mononuclear cells (PBMCs) and dysregulation across multiple lymphoid and myeloid compartments. Phase 1 of this project builds a reproducible, memory-bounded computational foundation to link genetics-anchored disease cell states to candidate therapeutics in downstream phases.

Phase 1 pursues three specific objectives without making causal or drug-efficacy claims:
1. **Dataset Acquisition, Metadata Summary, Subsampling, and QC**: Resolve and retrieve the Perez et al. 2022 (*Science*, `doi:10.1126/science.abf1970`) single-cell RNA-seq PBMC dataset from CZ CELLxGENE Discover (`436154da-bcf1-4130-9c8b-120ff9a888f2`), summarize full-dataset donor and cell metadata without loading the full expression matrix into RAM, construct a deterministic donor-balanced and cell-type-floored subsample (60,000–100,000 cells), and perform quality control, normalization, batch integration (Harmony), and neighborhood graph construction while preserving author cell-type annotations.
2. **Polygenic Disease Enrichment Scoring (`scDRS`)**: Score every subsampled cell for SLE polygenic GWAS enrichment using single-cell Disease Relevance Score (scDRS; Zhang et al. 2022, *Nature Genetics*, `doi:10.1038/s41588-022-01167-z`) alongside **height** as a negative-control trait. Aggregate single-cell scores across author cell types using scDRS downstream Monte Carlo tests for cell-type-level association, within-cell-type heterogeneity, and false discovery rate (FDR)-thresholded cell fractions.
3. **Interferon Known-Answer Check**: Compute an IFN-stimulated gene (ISG) module score per cell and evaluate SLE-versus-healthy differences strictly **at the donor level** within each cell type (and stratified by donor ancestry where sample sizes permit) via two-sided Mann-Whitney U tests with Benjamini-Hochberg (BH) multiple-testing correction and rank-biserial effect sizes.

Prior biological expectations (used strictly to evaluate calibration and biological plausibility after execution; never hard-coded or assumed by the pipeline):
- SLE scDRS enrichment is expected to appear in immune compartments implicated in lupus pathogenesis (such as B cells/plasmablasts, monocytes, dendritic cells, or T/NK subsets).
- Height (negative control) should show no statistically significant cell-type association in PBMCs; any significant association triggers an automated calibration warning.
- ISG module scores are expected to be elevated in SLE donors relative to healthy controls, particularly in myeloid/monocyte populations.

---

## Data Sources, Accessions, and Terms of Use

| Resource | Identifier / Endpoint | Role in Phase 1 | License / Terms |
| :--- | :--- | :--- | :--- |
| **Perez et al. 2022 SLE PBMC scRNA-seq** | CZ CELLxGENE Discover Collection: `436154da-bcf1-4130-9c8b-120ff9a888f2` (`https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2`) | Primary single-cell expression and donor/cell metadata source. Dataset ID(s) inside the collection are resolved dynamically at runtime via the CELLxGENE Curation REST API (`https://api.cellxgene.cziscience.com/curation/v1/collections/436154da-bcf1-4130-9c8b-120ff9a888f2`). | Distributed under CZ CELLxGENE Discover terms (typically CC-BY 4.0; see collection page). Individual-level metadata and expression matrices are used strictly as curated and distributed by CELLxGENE. |
| **Perez et al. 2022 Archival Accession** | NCBI GEO: `GSE174188` | Recorded in `outputs/provenance/provenance.json` for archival cross-reference only. | NCBI GEO terms of use. |
| **CZ CELLxGENE Census** | Python package `cellxgene-census`, pinned Census build `"2025-11-08"` (LTS) | Default metadata-first query and slice retrieval backend (`is_primary_data == True` and `dataset_id == '<resolved_id>'`). | CZI Census terms of service. |
| **scDRS MAGMA GWAS Gene Sets (74 traits)** | Figshare Article `19312583` (`https://figshare.com/articles/dataset/scDRS_data_release_030122/19312583`; API: `https://api.figshare.com/v2/articles/19312583`) | Polygenic GWAS gene sets and weights (`.gs` format) for SLE and height negative control. Verified by MD5/SHA256 upon download. | Academic open-access release accompanying Zhang et al. 2022 (*Nat Genet*). |
| **Kang et al. 2018 IFN-$\beta$ PBMC (Optional)** | NCBI GEO: `GSE96583` | Optional external positive control for the ISG module score (`ifn_external_control.enabled: false` by default). | NCBI GEO terms of use. |

---

## Compute Requirements and Memory Strategy

### Target Hardware Envelope
- **CPU**: 8 cores (`n_jobs: 8` configurable in `config.yaml`)
- **Memory**: ~5 GB free RAM (no swap assumed)
- **GPU**: None required
- **Disk**: ~100 GB free disk space (for H5AD cache, SOMA tile cache, intermediate scores, and outputs)

### Memory-Bounded Strategy
The full Perez et al. 2022 dataset contains ~1.2 million cells and **must never be loaded in full into memory**:
1. **Metadata-First Discovery (`fetch` & `subsample`)**:
   - **`census` backend (default)**: Queries only the `obs` table via `cellxgene_census.get_obs()` filtered to `dataset_id == '<resolved_id>'` and `is_primary_data == True`. The subsampling algorithm runs purely on this `pandas.DataFrame` (~1.2M rows of metadata uses <300 MB RAM). Only the selected ~80,000 cell coordinates (`soma_joinid`) are passed to `cellxgene_census.get_anndata()` to fetch raw counts.
   - **`h5ad` backend**: Downloads the source `.h5ad` file to `data_cache/` via streaming HTTP/Census download, opens it on disk with `anndata.read_h5ad(..., backed="r")`, extracts `.obs` into memory to select the subsample indices, and materializes only the selected slice (`adata[selected_idx].to_memory()`), immediately closing the backed file handle.
2. **Sequential Per-Trait Scoring (`scdrs`)**:
   - `scDRS` generates `n_ctrl` (default 1,000) Monte Carlo control scores per cell. Scoring traits sequentially (one trait at a time), writing `outputs/data/scdrs/<trait>.full_score.gz` and `.parquet` immediately to disk, and explicitly deleting full-score matrices (`del` + `gc.collect()`) keeps peak RSS within the ~5 GB budget.
3. **Stepwise Checkpointing & Peak RSS Logging**:
   - Every pipeline step reads its inputs from disk and writes outputs to `outputs/`, allowing single-step execution (`python run_phase1.py --step <name>`) and automatic skipping of completed steps unless `--force` is passed. Peak resident set size (RSS) is logged via `psutil` and `resource.getrusage` at the start and end of every step.

---

## Environment Setup (`uv`) and Execution

### Requirements and Version Pinning
Python **3.12** is required (`scanpy >= 1.12` requires Python $\ge 3.12$). All package dependencies in `requirements.txt` are pinned with exact `==` specifiers for reproducibility.

> **Note on dependency resolution**: `scdrs` and `cellxgene-census` (`tiledbsoma`) can impose strict upper/lower bounds on `numpy`, `pandas`, `scipy`, and `anndata`. Before installing, you can verify or re-lock the exact environment on your target Linux host using `uv pip compile requirements.txt -o requirements.lock.txt`.

### Installation
```bash
# 1. Create a Python 3.12 virtual environment using uv
uv venv --python 3.12 .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Verify/compile dependency compatibility (optional but recommended)
uv pip compile requirements.txt

# 4. Install pinned dependencies
uv pip install -r requirements.txt
```
Or simply run:
```bash
make setup
```

### Running the Pipeline
Run the entire Phase 1 workflow end-to-end:
```bash
make all
# Equivalent to:
python run_phase1.py --step all --config config.yaml
```

Run individual steps (each step checks for existing outputs and skips unless `--force` is supplied):
```bash
python run_phase1.py --step fetch --config config.yaml
python run_phase1.py --step subsample --config config.yaml
python run_phase1.py --step qc --config config.yaml
python run_phase1.py --step scdrs --config config.yaml
python run_phase1.py --step ifn --config config.yaml
python run_phase1.py --step figures --config config.yaml
python run_phase1.py --step report --config config.yaml
```

Dry-run mode (validates configuration, resolves paths, and logs planned actions without network or compute execution):
```bash
python run_phase1.py --step all --config config.yaml --dry-run
```

Run the offline unit test suite (executes in <1 minute on synthetic fixtures):
```bash
make test
# Equivalent to:
pytest -q
```

### Estimated Runtime Ranges (8 CPU Cores, ~5 GB RAM, Linux)
*(Note: The ranges below are engineering estimates dependent on network bandwidth and disk I/O speed.)*
- **`fetch`**: ~3–15 minutes (Census `obs` fetch + Figshare `.gs` download; ~15–40 minutes if downloading the full source `.h5ad` in `h5ad` backend mode).
- **`subsample`**: ~1–3 minutes (in-memory obs stratification + slicing ~80,000 cells from Census or backed H5AD).
- **`qc`**: ~5–12 minutes (QC metrics, normalization, HVG selection, 50-component PCA, Harmony integration, kNN graph, and UMAP on ~80,000 cells).
- **`scdrs`**: ~15–35 minutes (~7–15 minutes per trait for `n_ctrl=1000` on ~80,000 cells, plus kNN-based downstream group analysis).
- **`ifn`**: ~1–2 minutes (`score_genes` + donor-level aggregation and Mann-Whitney U tests).
- **`figures` & `report`**: ~1–3 minutes.

---

## Output Inventory

All outputs are written relative to `outputs_dir` (default `outputs/`):

- **Provenance (`outputs/provenance/`)**:
  - `provenance.json`: Structured JSON records for every remote resource accessed (URLs, resolved dataset IDs, requested and resolved Census versions, `cellxgene_census` and `tiledbsoma` package versions, ISO-8601 timestamps, byte sizes, computed MD5/SHA256 digests, provider MD5s, and verification flags).
  - `checksums_outputs.txt`: SHA256 digests of every file produced in `outputs/` at the end of the run.
- **Logs (`outputs/logs/`)**:
  - `run_<timestamp>.log`: Detailed timestamped execution log including step durations, package versions, git commit hash, and peak RAM usage per step.
  - `config_used_<timestamp>.yaml`: Exact snapshot of the configuration used for the run.
- **Processed Data (`outputs/data/`)**:
  - `obs_full_metadata.parquet`: Full-dataset cell metadata (`obs`) fetched during `fetch`.
  - `subsample_selected_obs.parquet`: Metadata table for the selected subsample cells.
  - `sle_pbmc_subsample_raw.h5ad`: Raw-count AnnData for the subsampled cells prior to QC filtering.
  - `sle_pbmc_subsample_processed.h5ad`: QC-filtered, normalized, HVG-annotated, PCA/Harmony/kNN/UMAP-processed AnnData with raw counts preserved in `adata.layers["counts"]`, per-cell scDRS scores in `adata.obs`, and `isg_score` in `adata.obs`.
  - `scdrs/<trait>.full_score.gz` and `scdrs/<trait>.score.parquet`: Per-cell scDRS normalized scores, raw scores, z-scores, Monte Carlo p-values, and cell-level FDRs.
- **Tables (`outputs/tables/`)**:
  - `metadata_summary_disease.csv`, `metadata_summary_ancestry.csv`, `metadata_summary_sex.csv`, `metadata_summary_batch.csv`, `metadata_summary_disease_x_batch.csv`, `metadata_medication_columns.csv`: Full-dataset donor and cell count summaries and confounding checks.
  - `subsample_counts.csv`: Cell and donor counts before vs. after subsampling stratified by donor, disease, cell type, and ancestry.
  - `qc_filter_summary.csv`: Numbers of cells and genes retained and removed at each QC step.
  - `scdrs_celltype_<trait>.csv`: Per-cell-type scDRS downstream summary (`n_cell`, `assoc_mcp`, `assoc_mcz`, `hetero_mcp`, `hetero_mcz`, `n_fdr_0.1`, `frac_fdr_0.1`).
  - `scdrs_celltype_combined.csv`: Combined table across all scored traits including within-trait BH-adjusted association q-values (`assoc_mcp_bh`).
  - `ifn_donor_means.csv`: Per-donor, per-cell-type mean `isg_score` and cell counts (restricted to donor-cell-type pairs with $\ge$ `min_cells_per_donor_celltype` cells).
  - `ifn_donor_level_stats.csv`: Donor-level Mann-Whitney U test statistics (overall and ancestry-stratified) comparing SLE vs. healthy donor means per cell type, medians, median differences, rank-biserial correlations, and BH-adjusted p-values.
- **Figures (`outputs/figures/`, each saved as both 300 DPI `.png` and vector `.pdf`)**:
  - `qc_violin.png` / `.pdf`: Per-cell distributions of `n_genes_by_counts`, `total_counts`, and `pct_counts_mt`.
  - `subsample_composition.png` / `.pdf`: Bar charts comparing cell-type and disease proportions before vs. after subsampling.
  - `umap_celltype.png`, `umap_disease.png`, `umap_ancestry.png`, `umap_batch.png` (if batch column present): UMAP embeddings colored by metadata covariates.
  - `scdrs_umap_<trait>.png` / `.pdf`: UMAP colored by per-cell scDRS normalized score for each trait.
  - `scdrs_celltype_<trait>.png` / `.pdf`: Horizontal bar/dot plot of $-\log_{10}(\text{association MC } p)$ per cell type with nominal and BH significance markers.
  - `scdrs_sle_vs_height.png` / `.pdf`: Side-by-side comparison of cell-type enrichment ($-\log_{10} p$ and MC z-score) between SLE and height negative control.
  - `ifn_isg_by_celltype_disease.png` / `.pdf`: Donor-level box/strip plot of mean ISG scores comparing SLE vs. healthy donors across cell types.
  - `umap_isg_score.png` / `.pdf`: UMAP colored by single-cell `isg_score`.
- **Report (`outputs/reports/`)**:
  - `phase1_run_report.md`: Automatically generated markdown report populated strictly from the computed tables, provenance metadata, calibration checks, caveats, and Phase 2 roadmap.

---

## Interpreting `scDRS` Outputs

`scDRS` (Zhang et al. 2022) evaluates polygenic disease enrichment at single-cell and cell-type resolution:
1. **Cell-Level Scores**: For each cell, `scDRS` computes a raw disease score from the MAGMA top-1,000 GWAS genes weighted by their GWAS z-scores and inverse expression technical noise (`weight_opt="vs"`), together with `n_ctrl` (default 1,000) Monte Carlo control gene sets matched on gene expression mean and variance bins (`n_mean_bin=20, n_var_bin=20`). Scores are standardized across cells and control sets to yield `norm_score`, `mc_pval`, `pval`, and `zscore`.
2. **Group-Level Association (`assoc_mcp`, `assoc_mcz`)**: Tests whether the mean normalized score in a given cell type is significantly higher than expected under the Monte Carlo control distribution across cells in that group.
3. **Within-Group Heterogeneity (`hetero_mcp`, `hetero_mcz`)**: Uses Geary's $C$ along the cell-cell kNN graph (`obsp["connectivities"]`) within a given cell type to test whether cells with high disease scores cluster spatially in state space (indicating a disease-associated sub-state within the broader cell type). Note: The kNN graph used by `scDRS` downstream analysis is constructed in Step B (`qc`) from `X_pca_harmony` when a batch covariate is integrated via Harmony (or `X_pca` if no batch column is used).
4. **Monte Carlo P-Value Resolution**: Because `assoc_mcp` and `hetero_mcp` are empirical Monte Carlo p-values computed from `n_ctrl` control gene sets, their minimum possible value is bounded by $\frac{1}{1 + n_{\text{ctrl}}}$ (i.e., $p \ge \frac{1}{1001} \approx 9.99 \times 10^{-4}$ when `n_ctrl: 1000`). Use `assoc_mcz` and `hetero_mcz` (Monte Carlo z-scores) to rank cell types that hit the empirical p-value floor.
5. **FDR Cell Fraction (`n_fdr_0.1`, `frac_fdr_0.1`)**: Proportion of individual cells within a cell type whose cell-level Monte Carlo p-value passes Benjamini-Hochberg $\text{FDR} < 0.1$ across all cells.

---

## Known Limitations

1. **Subsampling Variance**: To operate within ~5 GB RAM, the pipeline analyzes a donor-balanced, cell-type-floored subsample of ~80,000 cells out of ~1.2M total cells. Results depend on the random seed (`seed: 0`). You can evaluate stability across multiple seeds by populating `subsample.seeds_for_stability` in `config.yaml`.
2. **Census vs. Source H5AD Metadata Columns**: The `census` backend provides standardized CZ CELLxGENE schema columns (`donor_id`, `disease`, `cell_type`, `self_reported_ethnicity`, `sex`, `assay`), which may omit study-specific author columns such as processing batch, pool, or finest-grain author subcluster labels. Switching `cellxgene.backend: "h5ad"` preserves all author `.obs` columns from the original H5AD.
3. **GWAS Ancestry Mismatch**: The scDRS 74-trait MAGMA gene sets (release `030122`) were derived predominantly from European-ancestry GWAS cohorts, whereas the Perez et al. 2022 cohort includes donors of diverse ancestry (notably European and East Asian). Cross-ancestry LD differences may attenuate polygenic signal for ancestry-specific risk variants.
4. **Medication and Clinical Confounding**: SLE donors in clinical cohorts frequently receive immunomodulatory therapies (e.g., hydroxychloroquine, corticosteroids, mycophenolate) and vary in disease activity (SLEDAI). Any treatment metadata present in `.obs` are inventoried in Step A, but unrecorded medication differences can influence transcriptional states such as the IFN signature.

---

## List of `TODO(verify)` Items in the Codebase

Every assumption regarding external API response fields, third-party function signatures, or column names is annotated in the source code with `# TODO(verify):` and validated at runtime:
1. **`requirements.txt`**: `# TODO(verify): versions chosen by code generator; confirm with uv pip compile and adjust if resolution fails.`
2. **`src/lupus_p1/download.py` (CELLxGENE Curation API)**: `# TODO(verify): Confirm exact JSON field names returned by GET https://api.cellxgene.cziscience.com/curation/v1/collections/{collection_id}` (specifically `datasets`, `dataset_id` vs `id`, `title`, and `cell_count`). Runtime fallback inspects available keys and logs them.
3. **`src/lupus_p1/download.py` (Census `get_obs` / `get_anndata`)**: `# TODO(verify): Confirm exact parameter names for cellxgene_census.get_obs and cellxgene_census.get_anndata against cellxgene-census==1.16.2` (`obs_value_filter`, `obs_coords`, `column_names`, `X_name="raw"`).
4. **`src/lupus_p1/download.py` (Census `download_source_h5ad`)**: `# TODO(verify): Confirm exact signature of cellxgene_census.download_source_h5ad(dataset_id, to_path=..., census_version=...)`.
5. **`src/lupus_p1/download.py` (Figshare API `19312583`)**: `# TODO(verify): Confirm exact .gs filename in figshare article 19312583` (e.g. `magma_10kb_top1000_zscore.74_traits.rv1.gs` or similar) and JSON keys (`files`, `name`, `download_url`, `supplied_md5`, `computed_md5`).
6. **`src/lupus_p1/download.py` (Optional `GSE96583` Positive Control)**: `# TODO(verify): Confirm exact supplementary file names and barcode/matrix layout on GEO for GSE96583 if enabling ifn_external_control.enabled: true.`
7. **`src/lupus_p1/qc_preprocess.py` (Gene Symbol Mapping)**: `# TODO(verify): Confirm exact var column name holding HGNC gene symbols in CELLxGENE Census / source H5AD` (`feature_name` vs `gene_symbols` vs `var_names`).
8. **`src/lupus_p1/qc_preprocess.py` (Harmony Integration)**: `# TODO(verify): Confirm harmonypy.run_harmony return object attribute (.Z_corr) and orientation (n_pcs x n_cells vs n_cells x n_pcs) in harmonypy==0.0.10.`
9. **`src/lupus_p1/scdrs_scoring.py` (`scDRS` Python API & CLI)**: `# TODO(verify): Confirm exact signatures and return columns of scdrs.util.load_gs, scdrs.preprocess, scdrs.score_cell, and scdrs.method.downstream_group_analysis in scdrs==1.0.2.`

---

## Saving to Google Drive

All pipeline outputs (`outputs/`) and cached datasets (`data_cache/`) are written strictly to the local filesystem (or to `LUPUS_P1_HOME` if pointed at a mounted Google Drive / `rclone` directory). **No automated Google Drive upload code is included.** Once the run completes and `outputs/provenance/checksums_outputs.txt` and `outputs/reports/phase1_run_report.md` are generated, upload the `outputs/` directory and the repository code to your Google Drive project folder (`Lupus cell states to drugs (P1)`) manually or via `rclone copy`.
