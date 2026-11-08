"""Synthetic offline fixtures (obs DataFrames and small AnnData objects) for unit tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: mark test as requiring internet access (skipped by default)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m") and "network" in config.getoption("-m"):
        return
    skip_network = pytest.mark.skip(reason="Network test skipped by default in offline test suite")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture()
def synthetic_obs_df() -> pd.DataFrame:
    """Create a synthetic obs DataFrame with 20 donors, 2 diseases, 2 ancestries, and 4 cell types."""
    rng = np.random.default_rng(42)
    rows = []
    donors = [f"donor_{i:02d}" for i in range(20)]
    for i, d_id in enumerate(donors):
        disease = "systemic lupus erythematosus" if i < 10 else "normal"
        ancestry = "European" if i % 2 == 0 else "East Asian"
        sex = "female"
        n_cells_donor = int(rng.integers(80, 250))
        for c_idx in range(n_cells_donor):
            u = rng.random()
            if u < 0.45:
                ct = "CD4-positive, alpha-beta T cell"
            elif u < 0.80:
                ct = "classical monocyte"
            elif u < 0.97:
                ct = "B cell"
            else:
                ct = "plasmablast"
            rows.append(
                {
                    "cell_id": f"{d_id}_cell_{c_idx:04d}",
                    "donor_id": d_id,
                    "disease": disease,
                    "cell_type": ct,
                    "self_reported_ethnicity": ancestry,
                    "sex": sex,
                }
            )

    df = pd.DataFrame(rows).set_index("cell_id")
    return df


@pytest.fixture()
def synthetic_anndata(synthetic_obs_df: pd.DataFrame) -> ad.AnnData:
    """Create a small synthetic AnnData object with raw counts and ISG genes."""
    rng = np.random.default_rng(123)
    obs = synthetic_obs_df.iloc[:400].copy()
    isg_genes = ["IFI27", "IFI44L", "ISG15", "MX1", "RSAD2", "SIGLEC1"]
    mt_genes = ["MT-CO1", "MT-ND1"]
    other_genes = [f"GENE_{i:03d}" for i in range(92)]
    all_genes = isg_genes + mt_genes + other_genes

    counts = rng.poisson(lam=2.0, size=(len(obs), len(all_genes))).astype(np.float32)
    sle_mask = (obs["disease"] == "systemic lupus erythematosus").to_numpy()
    counts[np.ix_(sle_mask, np.arange(len(isg_genes)))] += rng.poisson(
        lam=8.0, size=(int(sle_mask.sum()), len(isg_genes))
    )

    var = pd.DataFrame({"feature_name": all_genes}, index=all_genes)
    return ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
