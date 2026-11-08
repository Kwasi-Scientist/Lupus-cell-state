"""Configuration loading, validation, and path resolution for Phase 1."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CellxGeneConfig:
    collection_id: str = "436154da-bcf1-4130-9c8b-120ff9a888f2"
    dataset_id: str | None = None
    census_version: str = "2025-11-08"
    backend: str = "census"


@dataclass(frozen=True)
class SubsampleConfig:
    target_cells: int = 80000
    min_target: int = 60000
    max_target: int = 100000
    max_cells_per_donor: int | None = None
    min_cells_per_type: int = 200
    seeds_for_stability: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ColumnsConfig:
    donor: str = "donor_id"
    disease: str = "disease"
    celltype: str = "cell_type"
    ancestry: str = "self_reported_ethnicity"
    sex: str = "sex"
    batch: str | None = None
    normal_disease_label: str | None = None
    sle_disease_regex: str = r"(?i)lupus|\bSLE\b"


@dataclass(frozen=True)
class QCConfig:
    min_genes: int = 200
    max_pct_mt: float = 20.0
    min_cells_per_gene: int = 3


@dataclass(frozen=True)
class PreprocessConfig:
    n_top_genes: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    use_harmony: bool = True


@dataclass(frozen=True)
class ScDRSTraitsConfig:
    sle_regex: str = r"(?i)lupus|\bSLE\b"
    height_regex: str = r"(?i)height"


@dataclass(frozen=True)
class ScDRSConfig:
    figshare_article_id: int = 19312583
    gs_file_regex: str = r"(?i).*74.*\.gs$|.*magma.*\.gs$|.*\.gs$"
    gs_file_name: str | None = None
    traits: ScDRSTraitsConfig = field(default_factory=ScDRSTraitsConfig)
    extra_traits: list[str] = field(default_factory=list)
    n_ctrl: int = 1000
    batch_covariates: bool = True
    fdr_cell: float = 0.1
    mode: str = "python"


@dataclass(frozen=True)
class IFNConfig:
    genes: list[str] = field(
        default_factory=lambda: [
            "IFI27",
            "IFI44L",
            "ISG15",
            "MX1",
            "RSAD2",
            "SIGLEC1",
        ]
    )
    extra_genes: list[str] = field(default_factory=lambda: ["IFI44", "OAS1"])
    use_extra_genes: bool = False
    min_cells_per_donor_celltype: int = 10

    @property
    def active_genes(self) -> list[str]:
        if self.use_extra_genes:
            seen: dict[str, None] = {}
            for g in [*self.genes, *self.extra_genes]:
                seen[g] = None
            return list(seen.keys())
        return list(self.genes)


@dataclass(frozen=True)
class IFNExternalControlConfig:
    enabled: bool = False
    geo_accession: str = "GSE96583"


@dataclass(frozen=True)
class PipelineConfig:
    project_root: str
    outputs_dir: str
    cache_dir: str
    seed: int
    n_jobs: int
    cellxgene: CellxGeneConfig
    geo_accession: str
    subsample: SubsampleConfig
    columns: ColumnsConfig
    qc: QCConfig
    preprocess: PreprocessConfig
    scdrs: ScDRSConfig
    ifn: IFNConfig
    ifn_external_control: IFNExternalControlConfig
    root_path: Path
    outputs_path: Path
    cache_path: Path


def _resolve_root_path(configured_root: str, config_file_path: Path | None = None) -> Path:
    """Resolve project root using LUPUS_P1_HOME env var if set, else config value."""
    env_home = os.environ.get("LUPUS_P1_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()

    candidate = Path(configured_root).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if config_file_path is not None:
        return (config_file_path.parent / candidate).resolve()
    return candidate.resolve()


def _resolve_subpath(root_path: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root_path / p).resolve()


def _validate_regex(pattern: str, field_name: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression in '{field_name}': {pattern!r} ({exc})") from exc


def validate_config(cfg: PipelineConfig) -> None:
    """Validate configuration fields and fail loudly if constraints are violated."""
    if cfg.n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {cfg.n_jobs}")

    if cfg.cellxgene.backend not in ("census", "h5ad"):
        raise ValueError(
            f"cellxgene.backend must be 'census' or 'h5ad', got {cfg.cellxgene.backend!r}"
        )

    if not cfg.cellxgene.collection_id or not isinstance(cfg.cellxgene.collection_id, str):
        raise ValueError("cellxgene.collection_id must be a non-empty string.")

    sub = cfg.subsample
    if sub.min_target <= 0 or sub.max_target < sub.min_target:
        raise ValueError(
            f"Invalid subsample target bounds: min_target={sub.min_target}, max_target={sub.max_target}"
        )
    if not (sub.min_target <= sub.target_cells <= sub.max_target):
        raise ValueError(
            f"subsample.target_cells ({sub.target_cells}) must lie within "
            f"[{sub.min_target}, {sub.max_target}]."
        )
    if sub.min_cells_per_type < 0:
        raise ValueError(f"subsample.min_cells_per_type must be >= 0, got {sub.min_cells_per_type}")
    if sub.max_cells_per_donor is not None and sub.max_cells_per_donor < 1:
        raise ValueError(
            f"subsample.max_cells_per_donor must be >= 1 when set, got {sub.max_cells_per_donor}"
        )

    if cfg.qc.min_genes < 0:
        raise ValueError(f"qc.min_genes must be >= 0, got {cfg.qc.min_genes}")
    if not (0.0 < cfg.qc.max_pct_mt <= 100.0):
        raise ValueError(f"qc.max_pct_mt must be in (0, 100], got {cfg.qc.max_pct_mt}")
    if cfg.qc.min_cells_per_gene < 0:
        raise ValueError(f"qc.min_cells_per_gene must be >= 0, got {cfg.qc.min_cells_per_gene}")

    if cfg.preprocess.n_top_genes < 50:
        raise ValueError(f"preprocess.n_top_genes must be >= 50, got {cfg.preprocess.n_top_genes}")
    if cfg.preprocess.n_pcs < 2:
        raise ValueError(f"preprocess.n_pcs must be >= 2, got {cfg.preprocess.n_pcs}")
    if cfg.preprocess.n_neighbors < 2:
        raise ValueError(f"preprocess.n_neighbors must be >= 2, got {cfg.preprocess.n_neighbors}")

    if cfg.scdrs.mode not in ("python", "cli"):
        raise ValueError(f"scdrs.mode must be 'python' or 'cli', got {cfg.scdrs.mode!r}")
    if cfg.scdrs.n_ctrl < 10:
        raise ValueError(f"scdrs.n_ctrl must be >= 10, got {cfg.scdrs.n_ctrl}")
    if not (0.0 < cfg.scdrs.fdr_cell < 1.0):
        raise ValueError(f"scdrs.fdr_cell must be in (0, 1), got {cfg.scdrs.fdr_cell}")

    _validate_regex(cfg.scdrs.gs_file_regex, "scdrs.gs_file_regex")
    _validate_regex(cfg.scdrs.traits.sle_regex, "scdrs.traits.sle_regex")
    _validate_regex(cfg.scdrs.traits.height_regex, "scdrs.traits.height_regex")
    _validate_regex(cfg.columns.sle_disease_regex, "columns.sle_disease_regex")
    for idx, extra_re in enumerate(cfg.scdrs.extra_traits):
        _validate_regex(extra_re, f"scdrs.extra_traits[{idx}]")

    if not cfg.ifn.active_genes:
        raise ValueError("ifn.genes must contain at least one gene symbol.")
    if cfg.ifn.min_cells_per_donor_celltype < 1:
        raise ValueError(
            "ifn.min_cells_per_donor_celltype must be >= 1, "
            f"got {cfg.ifn.min_cells_per_donor_celltype}"
        )


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML configuration from disk, merge defaults, resolve paths, and validate."""
    config_file = Path(path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return build_config_from_dict(raw, config_file_path=config_file)


def build_config_from_dict(
    raw: dict[str, Any], config_file_path: Path | None = None
) -> PipelineConfig:
    """Construct and validate a PipelineConfig from a dictionary."""
    project_root_str = str(raw.get("project_root", "."))
    outputs_dir_str = str(raw.get("outputs_dir", "outputs"))
    cache_dir_str = str(raw.get("cache_dir", "data_cache"))

    root_path = _resolve_root_path(project_root_str, config_file_path=config_file_path)
    outputs_path = _resolve_subpath(root_path, outputs_dir_str)
    cache_path = _resolve_subpath(root_path, cache_dir_str)

    cxg_raw = raw.get("cellxgene", {}) or {}
    cxg = CellxGeneConfig(
        collection_id=str(cxg_raw.get("collection_id", "436154da-bcf1-4130-9c8b-120ff9a888f2")),
        dataset_id=cxg_raw.get("dataset_id", None),
        census_version=str(cxg_raw.get("census_version", "2025-11-08")),
        backend=str(cxg_raw.get("backend", "census")),
    )

    sub_raw = raw.get("subsample", {}) or {}
    sub = SubsampleConfig(
        target_cells=int(sub_raw.get("target_cells", 80000)),
        min_target=int(sub_raw.get("min_target", 60000)),
        max_target=int(sub_raw.get("max_target", 100000)),
        max_cells_per_donor=(
            int(sub_raw["max_cells_per_donor"])
            if sub_raw.get("max_cells_per_donor") is not None
            else None
        ),
        min_cells_per_type=int(sub_raw.get("min_cells_per_type", 200)),
        seeds_for_stability=[int(s) for s in (sub_raw.get("seeds_for_stability") or [])],
    )

    col_raw = raw.get("columns", {}) or {}
    cols = ColumnsConfig(
        donor=str(col_raw.get("donor", "donor_id")),
        disease=str(col_raw.get("disease", "disease")),
        celltype=str(col_raw.get("celltype", "cell_type")),
        ancestry=str(col_raw.get("ancestry", "self_reported_ethnicity")),
        sex=str(col_raw.get("sex", "sex")),
        batch=col_raw.get("batch", None),
        normal_disease_label=col_raw.get("normal_disease_label", None),
        sle_disease_regex=str(col_raw.get("sle_disease_regex", r"(?i)lupus|\bSLE\b")),
    )

    qc_raw = raw.get("qc", {}) or {}
    qc = QCConfig(
        min_genes=int(qc_raw.get("min_genes", 200)),
        max_pct_mt=float(qc_raw.get("max_pct_mt", 20.0)),
        min_cells_per_gene=int(qc_raw.get("min_cells_per_gene", 3)),
    )

    prep_raw = raw.get("preprocess", {}) or {}
    prep = PreprocessConfig(
        n_top_genes=int(prep_raw.get("n_top_genes", 2000)),
        n_pcs=int(prep_raw.get("n_pcs", 50)),
        n_neighbors=int(prep_raw.get("n_neighbors", 15)),
        use_harmony=bool(prep_raw.get("use_harmony", True)),
    )

    scdrs_raw = raw.get("scdrs", {}) or {}
    traits_raw = scdrs_raw.get("traits", {}) or {}
    traits_cfg = ScDRSTraitsConfig(
        sle_regex=str(traits_raw.get("sle_regex", r"(?i)lupus|\bSLE\b")),
        height_regex=str(traits_raw.get("height_regex", r"(?i)height")),
    )
    scdrs_cfg = ScDRSConfig(
        figshare_article_id=int(scdrs_raw.get("figshare_article_id", 19312583)),
        gs_file_regex=str(
            scdrs_raw.get("gs_file_regex", r"(?i).*74.*\.gs$|.*magma.*\.gs$|.*\.gs$")
        ),
        gs_file_name=scdrs_raw.get("gs_file_name", None),
        traits=traits_cfg,
        extra_traits=[str(x) for x in (scdrs_raw.get("extra_traits") or [])],
        n_ctrl=int(scdrs_raw.get("n_ctrl", 1000)),
        batch_covariates=bool(scdrs_raw.get("batch_covariates", True)),
        fdr_cell=float(scdrs_raw.get("fdr_cell", 0.1)),
        mode=str(scdrs_raw.get("mode", "python")),
    )

    ifn_raw = raw.get("ifn", {}) or {}
    default_ifn_genes = ["IFI27", "IFI44L", "ISG15", "MX1", "RSAD2", "SIGLEC1"]
    ifn_cfg = IFNConfig(
        genes=[str(g) for g in (ifn_raw.get("genes") or default_ifn_genes)],
        extra_genes=[str(g) for g in (ifn_raw.get("extra_genes") or ["IFI44", "OAS1"])],
        use_extra_genes=bool(ifn_raw.get("use_extra_genes", False)),
        min_cells_per_donor_celltype=int(ifn_raw.get("min_cells_per_donor_celltype", 10)),
    )

    ext_raw = raw.get("ifn_external_control", {}) or {}
    ext_cfg = IFNExternalControlConfig(
        enabled=bool(ext_raw.get("enabled", False)),
        geo_accession=str(ext_raw.get("geo_accession", "GSE96583")),
    )

    cfg = PipelineConfig(
        project_root=project_root_str,
        outputs_dir=outputs_dir_str,
        cache_dir=cache_dir_str,
        seed=int(raw.get("seed", 0)),
        n_jobs=int(raw.get("n_jobs", 8)),
        cellxgene=cxg,
        geo_accession=str(raw.get("geo_accession", "GSE174188")),
        subsample=sub,
        columns=cols,
        qc=qc,
        preprocess=prep,
        scdrs=scdrs_cfg,
        ifn=ifn_cfg,
        ifn_external_control=ext_cfg,
        root_path=root_path,
        outputs_path=outputs_path,
        cache_path=cache_path,
    )
    validate_config(cfg)
    return cfg
