"""Create data-minimized provenance for model evaluation artifacts."""

import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path


def _file_record(path, rows):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "rows": int(rows),
    }


def build_experiment_provenance(split_paths, row_counts, packages):
    """Return timestamps, package versions, and hashes without storing row content."""
    package_versions = {}
    for package in packages:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "packages": package_versions,
        "splits": {
            name: _file_record(path, row_counts[name])
            for name, path in split_paths.items()
        },
    }
