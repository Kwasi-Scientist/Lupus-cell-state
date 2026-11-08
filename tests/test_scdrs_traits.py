"""Unit tests for scDRS trait regex selection (exact match vs zero or multiple match errors)."""

from __future__ import annotations

import pytest

from src.lupus_p1.scdrs_scoring import match_trait_by_regex, select_scdrs_traits


def test_select_scdrs_traits_exact_match() -> None:
    traits = [
        "PASS_Lupus",
        "UKB_460K.body_HEIGHTz",
        "PASS_Rheumatoid_Arthritis",
        "PASS_IBD",
    ]
    selected = select_scdrs_traits(
        available_traits=traits,
        sle_regex=r"(?i)lupus|\bSLE\b",
        height_regex=r"(?i)height",
        extra_trait_regexes=[r"(?i)rheumatoid"],
    )
    assert selected["sle"] == "PASS_Lupus"
    assert selected["height"] == "UKB_460K.body_HEIGHTz"
    assert selected["extra_1"] == "PASS_Rheumatoid_Arthritis"


def test_match_trait_zero_matches_raises() -> None:
    traits = ["PASS_Lupus", "UKB_460K.body_HEIGHTz"]
    with pytest.raises(ValueError, match="Zero traits matched"):
        match_trait_by_regex(traits, r"(?i)schizophrenia", label="missing_trait")


def test_match_trait_multiple_matches_raises() -> None:
    traits = ["PASS_Lupus_EUR", "PASS_Lupus_EAS", "UKB_460K.body_HEIGHTz"]
    with pytest.raises(ValueError, match="Multiple traits"):
        match_trait_by_regex(traits, r"(?i)lupus", label="SLE")
