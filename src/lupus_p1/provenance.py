"""Streaming checksum computation, provenance.json management, and environment capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "scanpy",
    "anndata",
    "cellxgene-census",
    "tiledbsoma",
    "scdrs",
    "harmonypy",
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "requests",
    "psutil",
    "pyarrow",
    "pytest",
    "igraph",
    "leidenalg",
    "pertpy",
)


def compute_file_checksums(file_path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Compute MD5, SHA256, and byte size of a file using streaming reads."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute checksums; file does not exist: {p}")

    md5_hasher = hashlib.md5()
    sha256_hasher = hashlib.sha256()
    total_bytes = 0

    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            md5_hasher.update(chunk)
            sha256_hasher.update(chunk)

    return {
        "file_size_bytes": total_bytes,
        "md5": md5_hasher.hexdigest(),
        "sha256": sha256_hasher.hexdigest(),
    }


def get_installed_package_versions(
    package_names: tuple[str, ...] = TRACKED_PACKAGES,
) -> dict[str, str | None]:
    """Query importlib.metadata for installed versions of tracked packages."""
    versions: dict[str, str | None] = {}
    for pkg in package_names:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def get_git_commit(repo_dir: Path) -> str | None:
    """Return the current git commit SHA if available, else None."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def capture_environment_metadata(repo_dir: Path) -> dict[str, Any]:
    """Capture Python version, git commit, and package versions."""
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "git_commit": get_git_commit(repo_dir),
        "packages": get_installed_package_versions(),
    }


def read_provenance(provenance_path: Path) -> dict[str, Any]:
    """Read outputs/provenance/provenance.json or initialize a default structure."""
    if provenance_path.is_file():
        with provenance_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    pkg_versions = get_installed_package_versions()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cellxgene_census_version": pkg_versions.get("cellxgene-census"),
        "tiledbsoma_version": pkg_versions.get("tiledbsoma"),
        "environment": {
            "python_version": sys.version.replace("\n", " "),
            "packages": pkg_versions,
        },
        "resources": [],
    }


def append_provenance_record(
    provenance_path: Path,
    *,
    resource_name: str,
    url_or_endpoint: str,
    collection_id: str | None = None,
    dataset_id: str | None = None,
    geo_accession: str | None = None,
    census_build_requested: str | None = None,
    census_build_resolved: str | None = None,
    local_file_path: Path | None = None,
    provider_md5: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append or update a resource record in outputs/provenance/provenance.json."""
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    prov = read_provenance(provenance_path)

    pkg_versions = get_installed_package_versions()
    prov["cellxgene_census_version"] = pkg_versions.get("cellxgene-census")
    prov["tiledbsoma_version"] = pkg_versions.get("tiledbsoma")
    prov["updated_at"] = datetime.now(timezone.utc).isoformat()

    file_size_bytes: int | None = None
    md5_hex: str | None = None
    sha256_hex: str | None = None
    md5_match: bool | None = None

    if local_file_path is not None and Path(local_file_path).is_file():
        chk = compute_file_checksums(local_file_path)
        file_size_bytes = chk["file_size_bytes"]
        md5_hex = chk["md5"]
        sha256_hex = chk["sha256"]
        if provider_md5:
            md5_match = md5_hex.lower() == provider_md5.strip().lower()

    record: dict[str, Any] = {
        "resource_name": resource_name,
        "url_or_endpoint": url_or_endpoint,
        "collection_id": collection_id,
        "dataset_id": dataset_id,
        "geo_accession": geo_accession,
        "census_build_requested": census_build_requested,
        "census_build_resolved": census_build_resolved,
        "cellxgene_census_version": pkg_versions.get("cellxgene-census"),
        "tiledbsoma_version": pkg_versions.get("tiledbsoma"),
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "local_file": str(local_file_path) if local_file_path is not None else None,
        "file_size_bytes": file_size_bytes,
        "md5": md5_hex,
        "sha256": sha256_hex,
        "provider_supplied_md5": provider_md5,
        "provider_md5_match": md5_match,
        "extra_metadata": extra_metadata or {},
    }

    resources: list[dict[str, Any]] = prov.get("resources", [])
    replaced = False
    for idx, existing in enumerate(resources):
        if existing.get("resource_name") == resource_name:
            resources[idx] = record
            replaced = True
            break
    if not replaced:
        resources.append(record)

    prov["resources"] = resources
    with provenance_path.open("w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    return record


def write_output_checksums(outputs_dir: Path) -> Path:
    """Compute SHA256 for every file under outputs_dir and write outputs/provenance/checksums_outputs.txt."""
    checksums_path = outputs_dir / "provenance" / "checksums_outputs.txt"
    checksums_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(
        p
        for p in outputs_dir.rglob("*")
        if p.is_file() and p.resolve() != checksums_path.resolve()
    )

    lines: list[str] = []
    for file_path in all_files:
        chk = compute_file_checksums(file_path)
        rel = file_path.relative_to(outputs_dir).as_posix()
        lines.append(f"{chk['sha256']}  {rel}")

    with checksums_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))

    return checksums_path
