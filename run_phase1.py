#!/usr/bin/env python3
"""CLI entry point for Phase 1 of 'Lupus cell states to drugs (P1)'."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.lupus_p1.config import PipelineConfig, load_config
from src.lupus_p1.logging_utils import get_peak_memory_mb, log_memory, setup_logging
from src.lupus_p1.provenance import (
    capture_environment_metadata,
    write_output_checksums,
)

VALID_STEPS = (
    "fetch",
    "subsample",
    "qc",
    "scdrs",
    "ifn",
    "figures",
    "report",
    "all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 pipeline: Lupus cell states to drugs (P1)."
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=VALID_STEPS,
        help="Pipeline step to execute (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run step(s) even if expected output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print planned execution steps without running.",
    )
    return parser.parse_args(argv)


def _step_outputs_exist(cfg: PipelineConfig, step: str) -> bool:
    """Check whether the primary artifacts of a step already exist on disk."""
    out = cfg.outputs_path
    if step == "fetch":
        obs_ok = (out / "data" / "obs_full_metadata.parquet").is_file()
        prov_ok = (out / "provenance" / "provenance.json").is_file()
        summary_ok = (out / "tables" / "metadata_summary_disease.csv").is_file()
        return obs_ok and prov_ok and summary_ok
    if step == "subsample":
        raw_h5ad = (out / "data" / "sle_pbmc_subsample_raw.h5ad").is_file()
        counts_csv = (out / "tables" / "subsample_counts.csv").is_file()
        return raw_h5ad and counts_csv
    if step == "qc":
        proc_h5ad = (out / "data" / "sle_pbmc_subsample_processed.h5ad").is_file()
        qc_csv = (out / "tables" / "qc_filter_summary.csv").is_file()
        return proc_h5ad and qc_csv
    if step == "scdrs":
        comb_csv = (out / "tables" / "scdrs_celltype_combined.csv").is_file()
        return comb_csv
    if step == "ifn":
        stats_csv = (out / "tables" / "ifn_donor_level_stats.csv").is_file()
        means_csv = (out / "tables" / "ifn_donor_means.csv").is_file()
        return stats_csv and means_csv
    if step == "figures":
        fig_dir = out / "figures"
        required = [
            "qc_violin.png",
            "umap_celltype.png",
            "umap_disease.png",
            "umap_ancestry.png",
            "subsample_composition.png",
            "scdrs_sle_vs_height.png",
            "ifn_isg_by_celltype_disease.png",
            "umap_isg_score.png",
        ]
        return all((fig_dir / f).is_file() for f in required)
    if step == "report":
        return (out / "reports" / "phase1_run_report.md").is_file()
    return False


def execute_step(step: str, cfg: PipelineConfig, force: bool, dry_run: bool, logger) -> None:
    """Execute a single pipeline step with timing, memory logging, and skip checks."""
    if not force and _step_outputs_exist(cfg, step):
        logger.info(
            "Step '%s' outputs already exist in %s. Skipping (use --force to recompute).",
            step,
            cfg.outputs_path,
        )
        return

    if dry_run:
        logger.info("[DRY-RUN] Would execute step '%s' with seed=%d", step, cfg.seed)
        return

    t0 = time.monotonic()
    start_iso = datetime.now(timezone.utc).isoformat()
    logger.info("=== STARTING STEP: %s (at %s) ===", step, start_iso)
    log_memory(logger, f"before step '{step}'")

    if step == "fetch":
        from src.lupus_p1.download import run_fetch_step

        run_fetch_step(cfg, logger)

    elif step == "subsample":
        from src.lupus_p1.subsample import run_subsample_step

        run_subsample_step(cfg, logger)

    elif step == "qc":
        from src.lupus_p1.qc_preprocess import run_qc_preprocess_step

        run_qc_preprocess_step(cfg, logger)

    elif step == "scdrs":
        from src.lupus_p1.scdrs_scoring import run_scdrs_step

        run_scdrs_step(cfg, logger)

    elif step == "ifn":
        from src.lupus_p1.ifn_check import run_ifn_step

        run_ifn_step(cfg, logger)

    elif step == "figures":
        from src.lupus_p1.plotting import run_plotting_step

        run_plotting_step(cfg, logger)

    elif step == "report":
        from src.lupus_p1.report import run_report_step

        run_report_step(cfg, logger)

    else:
        raise ValueError(f"Unknown pipeline step: {step}")

    gc.collect()
    elapsed = time.monotonic() - t0
    end_iso = datetime.now(timezone.utc).isoformat()
    peak_mb = get_peak_memory_mb()
    logger.info(
        "=== COMPLETED STEP: %s (at %s | elapsed: %.2f s | peak RSS: %.1f MB) ===",
        step,
        end_iso,
        elapsed,
        peak_mb,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Ensure standard output directory structure exists
    for subdir in ("logs", "provenance", "data", "data/scdrs", "tables", "figures", "reports"):
        (cfg.outputs_path / subdir).mkdir(parents=True, exist_ok=True)
    cfg.cache_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = setup_logging(cfg.outputs_path / "logs", timestamp=timestamp)

    # Copy exact config file used for this invocation
    config_copy_dest = cfg.outputs_path / "logs" / f"config_used_{timestamp}.yaml"
    shutil.copy2(config_path, config_copy_dest)
    logger.info("Copied active configuration to %s", config_copy_dest)

    # Log environment metadata (package versions, git commit, Python version)
    env_meta = capture_environment_metadata(cfg.root_path)
    logger.info("Environment metadata: python=%s, git_commit=%s", env_meta["python_version"], env_meta["git_commit"])
    logger.info("Pinned package versions: %s", env_meta["packages"])

    steps_to_run = (
        ["fetch", "subsample", "qc", "scdrs", "ifn", "figures", "report"]
        if args.step == "all"
        else [args.step]
    )

    for step in steps_to_run:
        execute_step(
            step=step,
            cfg=cfg,
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )

    if not args.dry_run:
        checksums_file = write_output_checksums(cfg.outputs_path)
        logger.info("Wrote SHA256 checksums of all output files to %s", checksums_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
