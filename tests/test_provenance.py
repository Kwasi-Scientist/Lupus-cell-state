"""Unit tests for streaming checksum calculation and provenance.json recording."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.lupus_p1.provenance import (
    append_provenance_record,
    compute_file_checksums,
    read_provenance,
    write_output_checksums,
)


def test_compute_file_checksums_and_provenance(tmp_path: Path) -> None:
    payload = b"lupus_p1_reproducible_test_payload\n" * 100
    sample_file = tmp_path / "sample.txt"
    sample_file.write_bytes(payload)

    expected_md5 = hashlib.md5(payload).hexdigest()
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    chk = compute_file_checksums(sample_file)
    assert chk["file_size_bytes"] == len(payload)
    assert chk["md5"] == expected_md5
    assert chk["sha256"] == expected_sha256

    prov_path = tmp_path / "outputs" / "provenance" / "provenance.json"
    rec = append_provenance_record(
        prov_path,
        resource_name="unit_test_resource",
        url_or_endpoint="https://example.org/sample.txt",
        collection_id="436154da-bcf1-4130-9c8b-120ff9a888f2",
        dataset_id="test-dataset-uuid",
        census_build_requested="2025-11-08",
        census_build_resolved="2025-11-08",
        local_file_path=sample_file,
        provider_md5=expected_md5,
    )
    assert rec["provider_md5_match"] is True

    loaded = read_provenance(prov_path)
    assert len(loaded["resources"]) == 1
    assert loaded["resources"][0]["sha256"] == expected_sha256

    checksums_txt = write_output_checksums(tmp_path / "outputs")
    assert checksums_txt.is_file()
    content = checksums_txt.read_text(encoding="utf-8")
    assert "provenance/provenance.json" in content
