from __future__ import annotations

"""Project-level constants and path helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ("CNV", "DME", "DRUSEN", "NORMAL")
RAW_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path
    processed_root: Path
    outputs_root: Path
    interim_root: Path
    tables_root: Path
    figures_root: Path
    models_root: Path
    logs_root: Path


def _has_expected_split_dirs(root: Path) -> bool:
    """Return True when root contains the expected raw split directories."""
    return all((root / split).is_dir() for split in RAW_SPLITS)


def resolve_oct_root(data_root: Path) -> Path:
    """
    Locate dataset folder robustly.
    Handles the common trailing-space directory name: `OCT2017 `.
    """
    # Case 1: dataset is already extracted as data/raw/{train,val,test}
    if data_root.exists() and _has_expected_split_dirs(data_root):
        return data_root

    # Case 2: dataset is under a wrapper folder such as data/raw/OCT2017
    candidates = [data_root / "OCT2017", data_root / "OCT2017 "]
    for candidate in candidates:
        if candidate.exists() and _has_expected_split_dirs(candidate):
            return candidate

    for child in sorted(data_root.iterdir()):
        if child.is_dir() and child.name.strip() == "OCT2017" and _has_expected_split_dirs(child):
            return child

    raise FileNotFoundError(
        f"Unable to locate OCT root under {data_root}. "
        "Expected either `data/raw/{train,val,test}` or `data/raw/OCT2017/{train,val,test}`."
    )


def make_paths(project_root: Path | None = None) -> Paths:
    """Create canonical path bundle used across modules."""
    root = (project_root or Path.cwd()).resolve()
    data_root = root / "data" / "raw"
    processed_root = root / "data" / "processed"
    outputs_root = root / "outputs"
    interim_root = root / "data" / "interim"
    return Paths(
        project_root=root,
        data_root=data_root,
        processed_root=processed_root,
        outputs_root=outputs_root,
        interim_root=interim_root,
        tables_root=outputs_root / "tables",
        figures_root=outputs_root / "figures",
        models_root=outputs_root / "models",
        logs_root=outputs_root / "logs",
    )


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create required output/intermediate folders when missing."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
