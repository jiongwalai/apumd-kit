# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat result records and environment metadata for DPA4C benchmarks."""

from __future__ import (
    annotations,
)

import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import socket
import subprocess
from dataclasses import (
    asdict,
    dataclass,
)
from datetime import (
    datetime,
    timezone,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Mapping,
    )

    from .contract import (
        BenchmarkCase,
    )
    from .metrics import (
        WorkloadEstimate,
    )

RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Record software and hardware identifiers without importing heavy tools."""

    git_sha: str = ""
    gpu: str = "cpu"
    cuda: str = ""
    pytorch: str = ""
    lammps: str = ""
    model_hash: str = ""
    host: str = ""
    python: str = ""
    timestamp_utc: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible environment mapping."""
        return asdict(self)


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    """Return a short command result, or an empty string when unavailable."""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def file_sha256(path: str | Path) -> str:
    """Hash a model artifact without loading it into memory."""
    model_path = Path(path)
    if not model_path.is_file():
        return ""
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_environment(
    repository_root: str | Path | None = None,
    model_path: str | Path | None = None,
) -> EnvironmentMetadata:
    """Collect reproducibility metadata with optional lazy PyTorch probing."""
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    gpu = "cpu"
    cuda = ""
    pytorch = ""
    try:
        import torch

        pytorch = str(torch.__version__)
        cuda = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    except (ImportError, RuntimeError):
        pass

    lammps = shutil.which("lmp") or shutil.which("lammps") or ""
    if not lammps and importlib.util.find_spec("lammps") is not None:
        lammps = "python:lammps"
    return EnvironmentMetadata(
        git_sha=_command_output(["git", "rev-parse", "HEAD"], cwd=root),
        gpu=gpu,
        cuda=cuda,
        pytorch=pytorch,
        lammps=lammps,
        model_hash="" if model_path is None else file_sha256(model_path),
        host=f"{socket.gethostname()} ({platform.machine()})",
        python=platform.python_version(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def result_row(
    case: BenchmarkCase,
    estimate: WorkloadEstimate,
    *,
    environment: EnvironmentMetadata | Mapping[str, Any] | None = None,
    status: str = "not_run",
    error: str = "",
    timing: Mapping[str, Any] | None = None,
    counters: Mapping[str, Any] | None = None,
    numeric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten one case, estimate, timing, and validation result for CSV."""
    case.validate()
    row: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "error": error,
        **{key: value for key, value in case.as_dict().items() if key != "model"},
        "model": case.model.name,
        "channels": case.model.channels,
        "lmax": case.model.lmax,
        "radial_modes": case.model.radial_modes,
        "fitting_width": case.model.fitting_width,
    }
    row.update(
        {
            f"estimate_{key}": value
            for key, value in estimate.as_dict().items()
            if key != "profile"
        }
    )
    for key, value in estimate.profile.as_dict().items():
        row[f"profile_{key}"] = value
    if timing:
        row.update(timing)
    if counters:
        row.update(counters)
    if numeric:
        row.update(numeric)
    if environment is not None:
        values = (
            environment.as_dict()
            if isinstance(environment, EnvironmentMetadata)
            else dict(environment)
        )
        row.update({f"env_{key}": value for key, value in values.items()})
    return row


def _fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    """Keep stable contract fields first and append optional counters."""
    preferred = (
        "schema_version",
        "status",
        "error",
        "model",
        "path",
        "atoms",
        "edges",
        "neighbors",
        "active_types",
        "model_types",
        "mode",
        "tile",
        "precision",
        "table_spacing",
    )
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(
        present.difference(preferred)
    )


def write_results(
    rows: Iterable[Mapping[str, Any]],
    csv_path: str | Path,
    json_path: str | Path | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write normalized flat CSV and a JSON envelope."""
    records = [dict(row) for row in rows]
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(records))
        writer.writeheader()
        writer.writerows(records)
    if json_path is None:
        return
    json_destination = Path(json_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "metadata": {} if metadata is None else dict(metadata),
        "records": records,
    }
    with json_destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Read either the JSON envelope or a flat CSV result file."""
    result_path = Path(path)
    if result_path.suffix.lower() == ".json":
        with result_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported benchmark result schema")
        return [dict(row) for row in payload.get("records", [])]
    with result_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
