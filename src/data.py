from __future__ import annotations

"""Dataset scanning, auditing, and patient-aware split generation."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from tqdm import tqdm

from src.config import CLASS_NAMES, RAW_SPLITS
from src.utils import file_sha1

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
FILENAME_PATTERN = re.compile(r"^(?P<class_name>[A-Z]+)-(?P<patient_id>\d+)-(?P<image_id>\d+)$")


@dataclass(frozen=True)
class ScanOptions:
    compute_sha1: bool = False
    max_files_per_class: int | None = None


def _iter_image_files(class_dir: Path) -> Iterable[Path]:
    """Yield supported image files in deterministic order."""
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_filename_tokens(path: Path) -> tuple[str | None, str | None]:
    """Parse patient/image IDs from filename pattern `<CLASS>-<PID>-<IMGID>`."""
    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None, None
    return match.group("patient_id"), match.group("image_id")


def scan_dataset(oct_root: Path, options: ScanOptions | None = None) -> pd.DataFrame:
    """Scan all splits/classes and return row-level dataset audit table."""
    options = options or ScanOptions()
    rows: list[dict[str, object]] = []

    for split in RAW_SPLITS:
        split_dir = oct_root / split
        if not split_dir.exists():
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            files = list(_iter_image_files(class_dir))
            if options.max_files_per_class is not None:
                files = files[: options.max_files_per_class]

            for path in tqdm(files, desc=f"scan:{split}/{class_name}", leave=False):
                patient_id, image_id = parse_filename_tokens(path)
                row = {
                    "filepath": str(path.resolve()),
                    "split": split,
                    "split_raw": split,
                    "class": class_name,
                    "class_name": class_name,
                    "patient_id": patient_id,
                    "image_id": image_id,
                    "width": None,
                    "height": None,
                    "mode": None,
                    "broken_file_flag": False,
                    "sha1": None,
                }

                try:
                    with Image.open(path) as img:
                        row["width"], row["height"] = img.size
                        row["mode"] = img.mode
                except (UnidentifiedImageError, OSError):
                    row["broken_file_flag"] = True

                if options.compute_sha1:
                    row["sha1"] = file_sha1(path)

                rows.append(row)

    return pd.DataFrame(rows)


def build_split_manifest(
    audit_df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
    raw_train_split_name: str = "train",
) -> pd.DataFrame:
    """Create final split manifest with patient-aware validation split.

    Strategy:
    - Keep raw `test` untouched as `test_final`.
    - Build `train_final/val_final` from raw `train` with grouped splitting.
    - Keep raw `val` as optional `val_raw_holdout`.
    """
    if audit_df.empty:
        raise ValueError("audit_df is empty. Run scan_dataset first.")

    if not (0 < val_fraction < 0.5):
        raise ValueError("val_fraction must be in (0, 0.5).")

    train_df = audit_df[
        (audit_df["split_raw"] == raw_train_split_name) & (~audit_df["broken_file_flag"])
    ].copy()
    if train_df.empty:
        raise ValueError("No valid training rows found in audit_df.")

    train_df["group_id"] = train_df["patient_id"].fillna(train_df["filepath"])

    unique_groups = train_df["group_id"].nunique()
    n_splits = max(2, int(round(1.0 / val_fraction)))
    n_splits = min(n_splits, unique_groups) if unique_groups >= 2 else 2

    # Prefer stratified grouped split. If data conditions are not compatible,
    # fall back to a grouped shuffle split to preserve leakage protection.
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(
            X=train_df,
            y=train_df["class_name"],
            groups=train_df["group_id"],
        )
        train_idx, val_idx = next(split_iter)
    except ValueError:
        fallback = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(fallback.split(train_df, groups=train_df["group_id"]))

    split_labels = pd.Series(index=train_df.index, data="train_final")
    split_labels.iloc[val_idx] = "val_final"
    train_df["split_final"] = split_labels

    test_df = audit_df[(audit_df["split_raw"] == "test") & (~audit_df["broken_file_flag"])].copy()
    if not test_df.empty:
        test_df["split_final"] = "test_final"

    raw_val_df = audit_df[(audit_df["split_raw"] == "val") & (~audit_df["broken_file_flag"])].copy()
    if not raw_val_df.empty:
        raw_val_df["split_final"] = "val_raw_holdout"

    manifest = pd.concat([train_df, test_df, raw_val_df], ignore_index=True, sort=False)
    manifest["split"] = manifest["split_final"]
    manifest["class"] = manifest["class_name"]
    manifest = manifest[
        [
            "filepath",
            "split",
            "split_final",
            "split_raw",
            "class",
            "class_name",
            "patient_id",
            "image_id",
        ]
    ].sort_values(by=["split_final", "class_name", "filepath"])
    return manifest.reset_index(drop=True)


def patient_overlap_table(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise patient overlap counts across final splits."""
    split_to_patients: dict[str, set[str]] = {}
    for split_name, split_df in manifest_df.groupby("split_final"):
        patients = set(split_df["patient_id"].dropna().astype(str).tolist())
        split_to_patients[split_name] = patients

    splits = sorted(split_to_patients.keys())
    rows: list[dict[str, object]] = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap = sorted(split_to_patients[a].intersection(split_to_patients[b]))
            rows.append(
                {
                    "split_a": a,
                    "split_b": b,
                    "overlap_count": len(overlap),
                    "sample_overlap_ids": ",".join(overlap[:5]),
                }
            )
    return pd.DataFrame(rows)


def _table_to_markdown_safe(df: pd.DataFrame, index: bool = True) -> str:
    """Convert table to markdown; gracefully degrade if `tabulate` is missing."""
    if df.empty:
        return "No rows."
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```text\n" + df.to_string(index=index) + "\n```"


def build_audit_report_markdown(
    audit_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    val_fraction: float,
) -> str:
    """Generate markdown summary for direct report inclusion."""
    class_count = (
        audit_df[~audit_df["broken_file_flag"]]
        .groupby(["split_raw", "class_name"])
        .size()
        .unstack(fill_value=0)
    )
    final_count = manifest_df.groupby(["split_final", "class_name"]).size().unstack(fill_value=0)
    broken_count = int(audit_df["broken_file_flag"].sum())
    total_count = len(audit_df)
    parse_ok = int(audit_df["patient_id"].notna().sum())
    parse_ratio = parse_ok / total_count if total_count else 0.0
    train_val_overlap = 0
    if not overlap_df.empty:
        mask_a = (overlap_df["split_a"] == "train_final") & (overlap_df["split_b"] == "val_final")
        mask_b = (overlap_df["split_a"] == "val_final") & (overlap_df["split_b"] == "train_final")
        overlap_rows = overlap_df[mask_a | mask_b]
        if not overlap_rows.empty:
            train_val_overlap = int(overlap_rows.iloc[0]["overlap_count"])

    lines = [
        "# Data Audit Report",
        "",
        "## Summary",
        f"- Total scanned files: {total_count}",
        f"- Broken files: {broken_count}",
        f"- Patient ID parsed: {parse_ok} ({parse_ratio:.2%})",
        f"- Final validation ratio target: {val_fraction:.2f}",
        "",
        "## Raw Split Counts",
        _table_to_markdown_safe(class_count),
        "",
        "## Final Split Counts",
        _table_to_markdown_safe(final_count),
        "",
        "## Patient Overlap Check",
        _table_to_markdown_safe(overlap_df, index=False),
        "",
        "## Split Protocol",
        "- Keep official `test` untouched for final evaluation.",
        "- Build `train_final/val_final` from official `train` using patient-aware grouped split.",
        "- Keep official `val` as `val_raw_holdout` for optional sanity checks only.",
        "",
        "## Model Selection Policy",
        "- Hyperparameter tuning and model selection must use `train_final -> val_final` only.",
        "- Primary model-selection metric is **Macro-F1**.",
        "- `test_final` is reserved for one final report evaluation only.",
        "- `val_raw_holdout` is for sanity checks, not for core model selection.",
        f"- Patient leakage check: overlap(`train_final`, `val_final`) = **{train_val_overlap}**.",
    ]
    return "\n".join(lines)
