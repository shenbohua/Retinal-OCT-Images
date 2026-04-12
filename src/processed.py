from __future__ import annotations

"""Export manifest-driven processed datasets for team sharing."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.preprocess import preprocess_for_features


def _channel_count(rgb: bool) -> int:
    return 3 if rgb else 1


def _build_output_path(
    root: Path,
    profile: str,
    split_name: str,
    class_name: str,
    stem: str,
    output_format: str,
) -> Path:
    suffix = ".png" if output_format == "png" else ".npy"
    return root / profile / split_name / class_name / f"{stem}{suffix}"


def export_processed_dataset(
    manifest_df: pd.DataFrame,
    processed_root: Path,
    profile: str,
    image_size: tuple[int, int] = (224, 224),
    output_format: str = "png",
    rgb: bool = False,
    normalize: bool = False,
    splits: Sequence[str] | None = None,
    overwrite: bool = False,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Export processed samples to disk and return processed-manifest table.

    Notes:
    - `normalize=True` is effective for `npy` output only.
    - `png` output keeps uint8 pixel values for interoperability.
    """
    if output_format not in {"png", "npy"}:
        raise ValueError("output_format must be one of {'png','npy'}.")

    selected = manifest_df.copy()
    if splits:
        selected = selected[selected["split_final"].isin(list(splits))].copy()
    if selected.empty:
        raise ValueError("No rows selected for export. Check requested splits.")
    if max_files is not None and max_files > 0:
        selected = selected.head(max_files).copy()

    rows: list[dict[str, object]] = []
    width, height = image_size
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="export-processed", leave=False):
        src = Path(str(row["filepath"]))
        split_name = str(row["split_final"])
        class_name = str(row["class_name"])
        out_path = _build_output_path(
            root=processed_root,
            profile=profile,
            split_name=split_name,
            class_name=class_name,
            stem=src.stem,
            output_format=output_format,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite or (not out_path.exists()):
            image = preprocess_for_features(src, size=image_size, normalize=False)
            if rgb:
                image = np.stack([image, image, image], axis=-1)

            if output_format == "png":
                Image.fromarray(image.astype(np.uint8)).save(out_path)
            else:
                arr = image.astype(np.float32) / 255.0 if normalize else image.astype(np.uint8)
                np.save(out_path, arr)

        rows.append(
            {
                "source_path": str(src.resolve()),
                "processed_path": str(out_path.resolve()),
                "profile": profile,
                "split_final": split_name,
                "class_name": class_name,
                "patient_id": row.get("patient_id"),
                "image_id": row.get("image_id"),
                "output_format": output_format,
                "width": width,
                "height": height,
                "channels": _channel_count(rgb),
                "normalized": bool(normalize and output_format == "npy"),
            }
        )

    return pd.DataFrame(rows)
