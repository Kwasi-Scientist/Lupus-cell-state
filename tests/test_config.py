"""Unit tests for configuration loading, validation, and LUPUS_P1_HOME path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lupus_p1.config import build_config_from_dict, load_config


def test_load_default_config_yaml() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "config.yaml")
    assert cfg.cellxgene.collection_id == "436154da-bcf1-4130-9c8b-120ff9a888f2"
    assert cfg.cellxgene.census_version == "2025-11-08"
    assert cfg.subsample.target_cells == 80000
    assert cfg.scdrs.figshare_article_id == 19312583


def test_lupus_p1_home_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUPUS_P1_HOME", str(tmp_path))
    cfg = build_config_from_dict({"outputs_dir": "custom_out", "cache_dir": "custom_cache"})
    assert cfg.root_path == tmp_path.resolve()
    assert cfg.outputs_path == (tmp_path / "custom_out").resolve()
    assert cfg.cache_path == (tmp_path / "custom_cache").resolve()


def test_invalid_target_cells_raises() -> None:
    with pytest.raises(ValueError, match="subsample.target_cells"):
        build_config_from_dict(
            {
                "subsample": {
                    "target_cells": 200000,
                    "min_target": 60000,
                    "max_target": 100000,
                }
            }
        )
