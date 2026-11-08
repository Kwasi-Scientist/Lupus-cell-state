"""Data resolution, metadata-first loading, Figshare gene-set download, and Step A summaries."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

from src.lupus_p1.config import PipelineConfig
from src.lupus_p1.logging_utils import log_memory
from src.lupus_p1.provenance import (
    append_provenance_record,
    compute_file_checksums,
    read_provenance,
)

CELLXGENE_CURATION_API_TEMPLATE = (
    "https://api.cellxgene.cziscience.com/curation/v1/collections/{collection_id}"
)
FIGSHARE_ARTICLE_API_TEMPLATE = "https://api.figshare.com/v2/articles/{article_id}"

BATCH_CANDIDATE_REGEX = re.compile(r"(?i)batch|cohort|processing|pool")
MEDICATION_CANDIDATE_REGEX = re.compile(r"(?i)medic|treat|drug|steroid|hydroxy")


def _request_with_retry(
    url: str,
    *,
    method: str = "GET",
    stream: bool = False,
    timeout: int = 60,
    max_retries: int = 4,
    backoff_base: float = 2.0,
    logger: logging.Logger | None = None,
) -> requests.Response:
    """Perform an HTTP request with exponential backoff and clear failure reporting."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method=method, url=url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            wait_s = backoff_base ** (attempt - 1)
            if logger:
                logger.warning(
                    "HTTP %s %s failed on attempt %d/%d (%s). Retrying in %.1fs...",
                    method,
                    url,
                    attempt,
                    max_retries,
                    exc,
                    wait_s,
                )
            time.sleep(wait_s)

    raise RuntimeError(
        f"Network request failed after {max_retries} attempts for URL: {url}. "
        f"Last error: {last_exc}"
    ) from last_exc


def resolve_cellxgene_dataset_id(
    collection_id: str,
    configured_dataset_id: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve the dataset ID(s) inside a CELLxGENE Discover collection via the Curation API.

    # TODO(verify): Confirm exact JSON field names returned by
    # GET https://api.cellxgene.cziscience.com/curation/v1/collections/{collection_id}
    # (specifically `datasets`, `dataset_id` vs `id`, `title` vs `name`, and `cell_count` vs `total_cell_count`).
    """
    url = CELLXGENE_CURATION_API_TEMPLATE.format(collection_id=collection_id)
    logger.info("Querying CELLxGENE Curation API for collection %s: %s", collection_id, url)
    resp = _request_with_retry(url, timeout=60, logger=logger)
    payload = resp.json()

    if not isinstance(payload, dict) or "datasets" not in payload:
        raise RuntimeError(
            f"Unexpected CELLxGENE Curation API response structure from {url}. "
            f"Top-level keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    datasets_raw = payload.get("datasets") or []
    if not datasets_raw:
        raise RuntimeError(
            f"CELLxGENE collection {collection_id} returned an empty 'datasets' list from {url}."
        )

    parsed_datasets: list[dict[str, Any]] = []
    for ds in datasets_raw:
        ds_id = ds.get("dataset_id") or ds.get("id")
        ds_title = ds.get("title") or ds.get("name") or ""
        cell_count = ds.get("cell_count")
        if cell_count is None:
            cell_count = ds.get("total_cell_count")
        assets = ds.get("assets") or []
        parsed_datasets.append(
            {
                "dataset_id": ds_id,
                "title": ds_title,
                "cell_count": int(cell_count) if cell_count is not None else -1,
                "assets": assets,
                "raw_keys": sorted(ds.keys()),
            }
        )
        logger.info(
            "Discovered dataset in collection %s: dataset_id=%s | title=%r | cell_count=%s",
            collection_id,
            ds_id,
            ds_title,
            cell_count,
        )

    if configured_dataset_id is not None:
        matching = [d for d in parsed_datasets if d["dataset_id"] == configured_dataset_id]
        if not matching:
            available_ids = [d["dataset_id"] for d in parsed_datasets]
            raise RuntimeError(
                f"Configured cellxgene.dataset_id={configured_dataset_id!r} was not found in "
                f"collection {collection_id}. Available dataset_ids: {available_ids}"
            )
        chosen = matching[0]
        selection_reason = "Explicitly set in config.yaml (cellxgene.dataset_id)"
    elif len(parsed_datasets) == 1:
        chosen = parsed_datasets[0]
        selection_reason = "Collection contains exactly one dataset"
    else:
        pbmc_candidates = [
            d
            for d in parsed_datasets
            if re.search(r"(?i)pbmc|lupus|sle|multiplex", d["title"] or "")
        ]
        pool = pbmc_candidates if pbmc_candidates else parsed_datasets
        chosen = sorted(pool, key=lambda d: d["cell_count"], reverse=True)[0]
        selection_reason = (
            "Selected main PBMC dataset among multiple collection datasets by title match and largest cell_count"
        )

    if not chosen.get("dataset_id"):
        raise RuntimeError(
            f"Could not extract a valid dataset_id from collection {collection_id}. "
            f"Dataset entry keys: {chosen.get('raw_keys')}"
        )

    logger.info(
        "Selected dataset_id=%s (title=%r, cell_count=%s) | reason: %s",
        chosen["dataset_id"],
        chosen["title"],
        chosen["cell_count"],
        selection_reason,
    )

    return {
        "collection_id": collection_id,
        "curation_api_url": url,
        "selected_dataset_id": chosen["dataset_id"],
        "selected_title": chosen["title"],
        "selected_cell_count": chosen["cell_count"],
        "selection_reason": selection_reason,
        "all_datasets": [
            {
                "dataset_id": d["dataset_id"],
                "title": d["title"],
                "cell_count": d["cell_count"],
            }
            for d in parsed_datasets
        ],
    }


def fetch_obs_from_census(
    dataset_id: str,
    census_version: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch only cell metadata (`obs`) from CELLxGENE Census without loading expression matrices.

    # TODO(verify): Confirm exact parameter names for cellxgene_census.open_soma and
    # cellxgene_census.get_obs against cellxgene-census==1.16.2 (`census_version`, `value_filter`).
    """
    import cellxgene_census

    value_filter = f"dataset_id == '{dataset_id}' and is_primary_data == True"
    logger.info(
        "Opening CELLxGENE Census (census_version=%r) to fetch obs with filter: %s",
        census_version,
        value_filter,
    )

    resolved_version_info: dict[str, Any] = {"requested": census_version, "resolved": census_version}
    try:
        desc = cellxgene_census.get_census_version_description(census_version)
        if isinstance(desc, dict):
            resolved_version_info["description"] = desc
            resolved_version_info["resolved"] = str(
                desc.get("release_build") or desc.get("version") or census_version
            )
    except Exception as exc:
        logger.warning("Could not fetch census_version_description for %r: %s", census_version, exc)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs_df = cellxgene_census.get_obs(
            census,
            organism="Homo sapiens",
            value_filter=value_filter,
        )

    if obs_df.empty:
        raise RuntimeError(
            f"CELLxGENE Census returned 0 cells for filter: {value_filter}. "
            "Verify dataset_id and census_version."
        )

    if "soma_joinid" not in obs_df.columns and obs_df.index.name == "soma_joinid":
        obs_df = obs_df.reset_index()

    obs_df.index = obs_df.index.astype(str)
    logger.info(
        "Fetched %d primary cells and %d obs columns from Census (resolved build: %s).",
        len(obs_df),
        obs_df.shape[1],
        resolved_version_info["resolved"],
    )
    return obs_df, resolved_version_info


def ensure_source_h5ad_downloaded(
    dataset_id: str,
    census_version: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Download the source dataset H5AD using cellxgene_census.download_source_h5ad if not cached.

    # TODO(verify): Confirm exact signature of cellxgene_census.download_source_h5ad
    # (specifically `dataset_id`, `to_path`, `census_version`) in cellxgene-census==1.16.2.
    """
    import cellxgene_census

    cache_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = cache_dir / f"{dataset_id}.h5ad"
    if h5ad_path.is_file() and h5ad_path.stat().st_size > 0:
        logger.info("Using cached source H5AD at %s", h5ad_path)
        return h5ad_path

    logger.info(
        "Downloading source H5AD for dataset_id=%s (census_version=%s) to %s ...",
        dataset_id,
        census_version,
        h5ad_path,
    )
    cellxgene_census.download_source_h5ad(
        dataset_id,
        to_path=str(h5ad_path),
        census_version=census_version,
    )
    if not h5ad_path.is_file() or h5ad_path.stat().st_size == 0:
        raise RuntimeError(f"Source H5AD download failed or produced empty file at {h5ad_path}")
    return h5ad_path


def fetch_obs_from_backed_h5ad(h5ad_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Open a local H5AD in backed='r' mode and extract only .obs into memory."""
    logger.info("Opening %s in backed='r' mode to read .obs metadata only...", h5ad_path)
    adata_backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_df = adata_backed.obs.copy()
    finally:
        if getattr(adata_backed, "file", None) is not None:
            adata_backed.file.close()

    if "is_primary_data" in obs_df.columns:
        n_before = len(obs_df)
        primary_mask = obs_df["is_primary_data"].astype(bool)
        if primary_mask.any():
            obs_df = obs_df.loc[primary_mask].copy()
            logger.info(
                "Filtered backed H5AD obs to is_primary_data == True: %d / %d cells retained.",
                len(obs_df),
                n_before,
            )

    obs_df["obs_barcode_id"] = obs_df.index.astype(str)
    logger.info(
        "Loaded %d cells and %d obs columns from backed H5AD.",
        len(obs_df),
        obs_df.shape[1],
    )
    return obs_df


def summarize_full_metadata(
    obs_df: pd.DataFrame,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run Step A: Summarize full-dataset obs metadata, detect batch/medication columns, and write CSVs."""
    tables_dir = cfg.outputs_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    donor_col = cfg.columns.donor
    disease_col = cfg.columns.disease
    ancestry_col = cfg.columns.ancestry
    sex_col = cfg.columns.sex

    for req_col in (donor_col, disease_col):
        if req_col not in obs_df.columns:
            raise RuntimeError(
                f"Required metadata column {req_col!r} not found in obs. "
                f"Available columns: {sorted(obs_df.columns)}"
            )

    def _summarize_by_column(col_name: str, out_name: str) -> pd.DataFrame:
        if col_name not in obs_df.columns:
            logger.warning("Column %r not present in obs; writing empty summary %s", col_name, out_name)
            empty_df = pd.DataFrame(columns=[col_name, "n_cells", "pct_cells", "n_donors"])
            empty_df.to_csv(tables_dir / out_name, index=False)
            return empty_df

        grouped = (
            obs_df.groupby(col_name, observed=False)
            .agg(
                n_cells=(donor_col, "size"),
                n_donors=(donor_col, "nunique"),
            )
            .reset_index()
        )
        grouped["pct_cells"] = (grouped["n_cells"] / max(len(obs_df), 1)) * 100.0
        grouped = grouped.sort_values(["n_cells", col_name], ascending=[False, True]).reset_index(
            drop=True
        )
        grouped.to_csv(tables_dir / out_name, index=False)
        return grouped

    disease_summary = _summarize_by_column(disease_col, "metadata_summary_disease.csv")
    ancestry_summary = _summarize_by_column(ancestry_col, "metadata_summary_ancestry.csv")
    sex_summary = _summarize_by_column(sex_col, "metadata_summary_sex.csv")

    batch_candidates = sorted(
        col for col in obs_df.columns if BATCH_CANDIDATE_REGEX.search(str(col))
    )
    logger.info(
        "Auto-detected candidate batch/cohort columns matching %r: %s (configured batch column: %r)",
        BATCH_CANDIDATE_REGEX.pattern,
        batch_candidates,
        cfg.columns.batch,
    )

    active_batch_col = cfg.columns.batch
    if active_batch_col and active_batch_col in obs_df.columns:
        batch_summary = _summarize_by_column(active_batch_col, "metadata_summary_batch.csv")
        ctab = (
            pd.crosstab(
                obs_df[disease_col].astype(str),
                obs_df[active_batch_col].astype(str),
                margins=True,
            )
            .reset_index()
        )
        ctab.to_csv(tables_dir / "metadata_summary_disease_x_batch.csv", index=False)
    elif batch_candidates:
        inspected_col = batch_candidates[0]
        logger.info(
            "Summarizing detected candidate batch column %r to metadata_summary_batch.csv for confounding inspection.",
            inspected_col,
        )
        batch_summary = _summarize_by_column(inspected_col, "metadata_summary_batch.csv")
        ctab = (
            pd.crosstab(
                obs_df[disease_col].astype(str),
                obs_df[inspected_col].astype(str),
                margins=True,
            )
            .reset_index()
        )
        ctab.to_csv(tables_dir / "metadata_summary_disease_x_batch.csv", index=False)
    else:
        batch_summary = pd.DataFrame(columns=["batch", "n_cells", "pct_cells", "n_donors"])
        batch_summary.to_csv(tables_dir / "metadata_summary_batch.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "metadata_summary_disease_x_batch.csv", index=False)

    med_candidates = sorted(
        col for col in obs_df.columns if MEDICATION_CANDIDATE_REGEX.search(str(col))
    )
    logger.info(
        "Medication/treatment columns matching %r in obs: %s",
        MEDICATION_CANDIDATE_REGEX.pattern,
        med_candidates,
    )
    med_rows: list[dict[str, Any]] = []
    for mcol in med_candidates:
        n_unique = int(obs_df[mcol].nunique(dropna=True))
        top_vals = (
            obs_df[mcol].astype(str).value_counts(dropna=False).head(5).to_dict()
        )
        med_rows.append(
            {
                "column_name": mcol,
                "n_unique_non_null": n_unique,
                "top_values_preview": str(top_vals),
            }
        )
    med_df = pd.DataFrame(
        med_rows, columns=["column_name", "n_unique_non_null", "top_values_preview"]
    )
    med_df.to_csv(tables_dir / "metadata_medication_columns.csv", index=False)

    return {
        "n_total_cells": len(obs_df),
        "n_total_donors": int(obs_df[donor_col].nunique()),
        "batch_candidates": batch_candidates,
        "medication_candidates": med_candidates,
        "disease_counts": disease_summary.to_dict(orient="records"),
        "ancestry_counts": ancestry_summary.to_dict(orient="records"),
        "sex_counts": sex_summary.to_dict(orient="records"),
    }


def download_scdrs_gs_from_figshare(
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Query Figshare API for article 19312583, select the 74-trait human .gs file, download, and verify MD5.

    # TODO(verify): Confirm exact .gs filename in figshare article 19312583
    # (e.g. `magma_10kb_top1000_zscore.74_traits.rv1.gs` or inside a tarball/folder)
    # and confirm Figshare API file object keys (`name`, `download_url`, `supplied_md5`, `computed_md5`).
    """
    article_id = cfg.scdrs.figshare_article_id
    api_url = FIGSHARE_ARTICLE_API_TEMPLATE.format(article_id=article_id)
    logger.info("Querying Figshare API for scDRS data release article %d: %s", article_id, api_url)

    resp = _request_with_retry(api_url, timeout=60, logger=logger)
    payload = resp.json()
    files_list = payload.get("files") if isinstance(payload, dict) else None
    if not files_list:
        raise RuntimeError(
            f"Figshare API {api_url} returned no 'files' list. Keys: "
            f"{list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    available_names = [str(f.get("name", "")) for f in files_list]
    logger.info("Figshare article %d contains %d files: %s", article_id, len(files_list), available_names)

    selected_file: dict[str, Any] | None = None
    if cfg.scdrs.gs_file_name:
        matches = [f for f in files_list if f.get("name") == cfg.scdrs.gs_file_name]
        if not matches:
            raise RuntimeError(
                f"Configured scdrs.gs_file_name={cfg.scdrs.gs_file_name!r} not found in Figshare "
                f"article {article_id}. Available files: {available_names}"
            )
        selected_file = matches[0]
    else:
        gs_regex = re.compile(cfg.scdrs.gs_file_regex)
        gs_candidates = [f for f in files_list if gs_regex.search(str(f.get("name", "")))]
        if not gs_candidates:
            raise RuntimeError(
                f"No file in Figshare article {article_id} matched scdrs.gs_file_regex="
                f"{cfg.scdrs.gs_file_regex!r}. Available files: {available_names}. "
                "Set scdrs.gs_file_name in config.yaml to override."
            )
        preferred = [
            f
            for f in gs_candidates
            if re.search(r"(?i)74|all_traits|human|hsapiens|magma", str(f.get("name", "")))
            and not re.search(r"(?i)mouse|mmusculus", str(f.get("name", "")))
        ]
        pool = preferred if preferred else gs_candidates
        selected_file = sorted(pool, key=lambda f: int(f.get("size", 0)), reverse=True)[0]

    fname = str(selected_file["name"])
    download_url = str(selected_file["download_url"])
    provider_md5 = selected_file.get("supplied_md5") or selected_file.get("computed_md5")

    dest_dir = cfg.cache_path / "scdrs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / fname

    need_download = True
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        chk = compute_file_checksums(dest_path)
        if provider_md5 and chk["md5"].lower() == str(provider_md5).strip().lower():
            logger.info(
                "Using cached Figshare .gs file %s (MD5 verified: %s)", dest_path, chk["md5"]
            )
            need_download = False
        elif not provider_md5:
            logger.info("Using cached Figshare .gs file %s (no provider MD5)", dest_path)
            need_download = False

    if need_download:
        logger.info("Downloading scDRS .gs file %r from %s ...", fname, download_url)
        with _request_with_retry(download_url, stream=True, timeout=120, logger=logger) as r:
            with dest_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    chk = compute_file_checksums(dest_path)
    if provider_md5:
        if chk["md5"].lower() != str(provider_md5).strip().lower():
            raise RuntimeError(
                f"MD5 mismatch for downloaded Figshare file {dest_path}: "
                f"computed {chk['md5']} vs provider {provider_md5}."
            )
        logger.info("Verified MD5 checksum for %s: %s", fname, chk["md5"])

    return {
        "article_id": article_id,
        "api_url": api_url,
        "file_name": fname,
        "download_url": download_url,
        "local_path": dest_path,
        "provider_md5": provider_md5,
        "available_files": available_names,
    }


def _verify_and_extract_raw_counts_h5ad(adata: ad.AnnData, logger: logging.Logger) -> ad.AnnData:
    """Ensure raw integer UMI counts are in adata.X when loading from a source H5AD."""
    def _looks_like_integer_counts(mat: Any) -> bool:
        if sp.issparse(mat):
            data = mat.data[: min(len(mat.data), 10000)]
        else:
            arr = np.asarray(mat)
            data = arr.ravel()[: min(arr.size, 10000)]
        if data.size == 0:
            return False
        if np.any(data < 0):
            return False
        return bool(np.allclose(data, np.round(data), atol=1e-4))

    if _looks_like_integer_counts(adata.X):
        logger.info("Verified that source H5AD .X contains raw non-negative integer counts.")
        return adata

    if adata.raw is not None and _looks_like_integer_counts(adata.raw.X):
        logger.info("Source H5AD .X is normalized; restoring raw integer counts from adata.raw.")
        raw_adata = ad.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
        )
        return raw_adata

    for layer_name in ("counts", "raw_counts", "raw"):
        if layer_name in adata.layers and _looks_like_integer_counts(adata.layers[layer_name]):
            logger.info("Restoring raw integer counts from adata.layers[%r].", layer_name)
            adata.X = adata.layers[layer_name].copy()
            return adata

    raise RuntimeError(
        "Failed to locate raw integer UMI counts in source H5AD (.X, .raw.X, or layers['counts']). "
        "Raw counts are strictly required for QC and scDRS."
    )


def fetch_expression_for_subsample(
    selected_obs: pd.DataFrame,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> ad.AnnData:
    """Fetch the raw-count expression matrix ONLY for the selected subsample cells.

    # TODO(verify): Confirm exact parameter names for cellxgene_census.get_anndata
    # (`obs_coords`, `X_name="raw"`, `organism="Homo sapiens"`) in cellxgene-census==1.16.2.
    """
    prov_path = cfg.outputs_path / "provenance" / "provenance.json"
    prov = read_provenance(prov_path)
    dataset_id = cfg.cellxgene.dataset_id
    if not dataset_id:
        for res in prov.get("resources", []):
            if res.get("resource_name") == "cellxgene_collection_resolution":
                dataset_id = res.get("dataset_id")
                break
    if not dataset_id:
        raise RuntimeError(
            "Dataset ID could not be determined from config or provenance.json. Run `--step fetch` first."
        )

    if cfg.cellxgene.backend == "census":
        import cellxgene_census

        if "soma_joinid" not in selected_obs.columns:
            raise RuntimeError(
                "selected_obs is missing 'soma_joinid' column required for Census coordinate slicing."
            )
        obs_coords = selected_obs["soma_joinid"].astype(int).to_numpy()
        logger.info(
            "Fetching raw counts for %d subsampled cells from CELLxGENE Census (dataset_id=%s, build=%s)...",
            len(obs_coords),
            dataset_id,
            cfg.cellxgene.census_version,
        )
        with cellxgene_census.open_soma(census_version=cfg.cellxgene.census_version) as census:
            adata = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                X_name="raw",
                obs_coords=obs_coords,
            )
        log_memory(logger, "after cellxgene_census.get_anndata for subsample")
        return adata

    elif cfg.cellxgene.backend == "h5ad":
        h5ad_path = ensure_source_h5ad_downloaded(
            dataset_id=dataset_id,
            census_version=cfg.cellxgene.census_version,
            cache_dir=cfg.cache_path / "cellxgene",
            logger=logger,
        )
        logger.info(
            "Opening source H5AD %s in backed='r' mode and slicing %d selected cells...",
            h5ad_path,
            len(selected_obs),
        )
        adata_backed = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if "obs_barcode_id" in selected_obs.columns:
                idx = selected_obs["obs_barcode_id"].astype(str).tolist()
            else:
                idx = selected_obs.index.astype(str).tolist()
            adata_sub = adata_backed[idx].to_memory()
        finally:
            if getattr(adata_backed, "file", None) is not None:
                adata_backed.file.close()

        adata_sub = _verify_and_extract_raw_counts_h5ad(adata_sub, logger)
        log_memory(logger, "after slicing backed H5AD to memory")
        return adata_sub

    else:
        raise ValueError(f"Unsupported backend: {cfg.cellxgene.backend}")


def fetch_optional_gse96583_control(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Optional external positive control (Kang et al. 2018 IFN-beta PBMC, GSE96583).

    # TODO(verify): Confirm exact supplementary file names and barcode/matrix layout on GEO
    # for GSE96583 if enabling `ifn_external_control.enabled: true`.
    """
    if not cfg.ifn_external_control.enabled:
        logger.info(
            "Optional external IFN control (%s) is disabled in config (ifn_external_control.enabled=false).",
            cfg.ifn_external_control.geo_accession,
        )
        return

    raise NotImplementedError(
        f"Optional external positive control {cfg.ifn_external_control.geo_accession} is enabled, "
        "but automatic GEO supplementary tar layout parsing requires manual verification of the "
        "exact file names (`GSE96583_batch1.total.tsne.df.tsv.gz` / matrix files). "
        "See # TODO(verify) in src/lupus_p1/download.py."
    )


def run_fetch_step(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """Execute the `fetch` step: resolve dataset ID, fetch obs metadata, run Step A summaries, and download scDRS .gs file."""
    prov_path = cfg.outputs_path / "provenance" / "provenance.json"

    resolution = resolve_cellxgene_dataset_id(
        collection_id=cfg.cellxgene.collection_id,
        configured_dataset_id=cfg.cellxgene.dataset_id,
        logger=logger,
    )
    dataset_id = resolution["selected_dataset_id"]

    append_provenance_record(
        prov_path,
        resource_name="cellxgene_collection_resolution",
        url_or_endpoint=resolution["curation_api_url"],
        collection_id=cfg.cellxgene.collection_id,
        dataset_id=dataset_id,
        geo_accession=cfg.geo_accession,
        census_build_requested=cfg.cellxgene.census_version,
        extra_metadata={
            "selected_title": resolution["selected_title"],
            "selected_cell_count": resolution["selected_cell_count"],
            "selection_reason": resolution["selection_reason"],
            "all_datasets_in_collection": resolution["all_datasets"],
        },
    )

    append_provenance_record(
        prov_path,
        resource_name="geo_archival_reference",
        url_or_endpoint=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={cfg.geo_accession}",
        collection_id=cfg.cellxgene.collection_id,
        dataset_id=dataset_id,
        geo_accession=cfg.geo_accession,
        extra_metadata={"note": "Recorded for archival provenance only; primary data loaded via CELLxGENE."},
    )

    if cfg.cellxgene.backend == "census":
        obs_df, census_ver_info = fetch_obs_from_census(
            dataset_id=dataset_id,
            census_version=cfg.cellxgene.census_version,
            logger=logger,
        )
        obs_parquet_path = cfg.outputs_path / "data" / "obs_full_metadata.parquet"
        obs_df.to_parquet(obs_parquet_path, index=True)
        append_provenance_record(
            prov_path,
            resource_name="cellxgene_census_obs_metadata",
            url_or_endpoint=f"cellxgene_census://Homo_sapiens/obs?dataset_id={dataset_id}",
            collection_id=cfg.cellxgene.collection_id,
            dataset_id=dataset_id,
            census_build_requested=cfg.cellxgene.census_version,
            census_build_resolved=census_ver_info.get("resolved"),
            local_file_path=obs_parquet_path,
            extra_metadata={
                "n_cells_fetched": len(obs_df),
                "obs_columns": list(obs_df.columns),
            },
        )
    else:
        h5ad_path = ensure_source_h5ad_downloaded(
            dataset_id=dataset_id,
            census_version=cfg.cellxgene.census_version,
            cache_dir=cfg.cache_path / "cellxgene",
            logger=logger,
        )
        append_provenance_record(
            prov_path,
            resource_name="cellxgene_source_h5ad",
            url_or_endpoint=f"cellxgene_census.download_source_h5ad({dataset_id})",
            collection_id=cfg.cellxgene.collection_id,
            dataset_id=dataset_id,
            census_build_requested=cfg.cellxgene.census_version,
            local_file_path=h5ad_path,
        )
        obs_df = fetch_obs_from_backed_h5ad(h5ad_path, logger)
        obs_parquet_path = cfg.outputs_path / "data" / "obs_full_metadata.parquet"
        obs_df.to_parquet(obs_parquet_path, index=True)

    log_memory(logger, "after fetching full dataset obs metadata")

    meta_summary = summarize_full_metadata(obs_df, cfg, logger)
    logger.info(
        "Completed Step A metadata summary: %d total cells across %d donors.",
        meta_summary["n_total_cells"],
        meta_summary["n_total_donors"],
    )

    gs_info = download_scdrs_gs_from_figshare(cfg, logger)
    append_provenance_record(
        prov_path,
        resource_name="scdrs_figshare_gs",
        url_or_endpoint=gs_info["download_url"],
        local_file_path=gs_info["local_path"],
        provider_md5=gs_info["provider_md5"],
        extra_metadata={
            "figshare_article_id": gs_info["article_id"],
            "figshare_api_url": gs_info["api_url"],
            "file_name": gs_info["file_name"],
            "available_files_in_article": gs_info["available_files"],
        },
    )

    fetch_optional_gse96583_control(cfg, logger)
