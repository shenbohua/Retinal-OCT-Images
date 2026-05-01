from __future__ import annotations

"""Training orchestration for traditional feature/classifier baselines."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, per_class_table, save_confusion_matrix
from src.features import (
    SIFTBOVWConfig,
    SIFTBOVWExtractor,
    compute_hog_matrix,
    compute_lbp_matrix,
    load_feature_cache,
    save_feature_cache,
)
from src.traditional_models import build_classifier, predict, save_model
from src.utils import timed


TRADITIONAL_CLASSIFIERS = ("linear_svm", "rbf_svm", "random_forest")
FEATURE_METHODS = ("hog", "lbp", "sift_bovw")


@dataclass(frozen=True)
class TraditionalTrainConfig:
    feature_name: str
    classifier_name: str
    image_size: tuple[int, int] = (224, 224)
    seed: int = 42
    sift_vocab_size: int = 128
    force_recompute_features: bool = False
    eval_split: str = "val_final"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def _fingerprint_paths(paths: Sequence[str]) -> str:
    """Build deterministic fingerprint from filepath list for cache naming."""
    digest = hashlib.sha1()
    for p in paths:
        digest.update(p.encode("utf-8"))
    return digest.hexdigest()[:12]


def _feature_cache_file(
    cache_root: Path,
    feature_name: str,
    split_name: str,
    image_paths: Sequence[str],
) -> Path:
    """Construct stable cache path for one feature/split subset."""
    fp = _fingerprint_paths(image_paths)
    return cache_root / f"{feature_name}_{split_name}_{fp}.npz"


def _select_rows(manifest_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Select one final split from manifest with explicit empty-check."""
    selected = manifest_df[manifest_df["split_final"] == split_name].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split_final={split_name}.")
    return selected


def _stratified_cap(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Cap rows with class-aware sampling to preserve label distribution."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df

    groups: list[pd.DataFrame] = []
    total = len(df)
    for class_name, g in df.groupby("class_name"):
        n = max(1, int(round(len(g) / total * max_samples)))
        groups.append(g.sample(n=min(n, len(g)), random_state=seed))
    sampled = pd.concat(groups, axis=0)

    # Guard against rounding drift.
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    elif len(sampled) < max_samples:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            extra_n = min(max_samples - len(sampled), len(remaining))
            sampled = pd.concat([sampled, remaining.sample(n=extra_n, random_state=seed)], axis=0)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _compute_features_for_split(
    feature_name: str,
    image_paths: Sequence[str],
    labels: np.ndarray,
    cache_path: Path,
    image_size: tuple[int, int],
    force_recompute: bool,
) -> np.ndarray:
    """Load cached features or compute from images for HOG/LBP."""
    if cache_path.exists() and not force_recompute:
        X_cached, y_cached, _ = load_feature_cache(cache_path)
        if len(y_cached) == len(labels):
            return X_cached

    if feature_name == "hog":
        X = compute_hog_matrix(image_paths=image_paths, image_size=image_size)
    elif feature_name == "lbp":
        X = compute_lbp_matrix(image_paths=image_paths, image_size=image_size)
    else:
        raise ValueError("_compute_features_for_split only supports hog/lbp directly.")

    save_feature_cache(cache_path, X=X, y=labels, filepaths=image_paths)
    return X


def run_traditional_experiment(
    manifest_df: pd.DataFrame,
    config: TraditionalTrainConfig,
    cache_root: Path,
    outputs_models_root: Path,
    outputs_tables_root: Path,
    outputs_figures_root: Path,
) -> dict[str, float | str]:
    """Run one end-to-end baseline experiment and save core artifacts."""
    train_df = _select_rows(manifest_df, "train_final")
    eval_df = _select_rows(manifest_df, config.eval_split)
    train_df = _stratified_cap(train_df, config.max_train_samples, seed=config.seed)
    eval_df = _stratified_cap(eval_df, config.max_eval_samples, seed=config.seed)

    train_paths = train_df["filepath"].tolist()
    eval_paths = eval_df["filepath"].tolist()
    y_train = train_df["class_name"].to_numpy()
    y_eval = eval_df["class_name"].to_numpy()

    cache_root.mkdir(parents=True, exist_ok=True)
    outputs_models_root.mkdir(parents=True, exist_ok=True)
    outputs_tables_root.mkdir(parents=True, exist_ok=True)
    outputs_figures_root.mkdir(parents=True, exist_ok=True)

    # Feature extraction branch
    notes_parts: list[str] = []
    if config.max_train_samples:
        notes_parts.append(f"train_cap={config.max_train_samples}")
    if config.max_eval_samples:
        notes_parts.append(f"eval_cap={config.max_eval_samples}")

    if config.feature_name in ("hog", "lbp"):
        X_train = _compute_features_for_split(
            feature_name=config.feature_name,
            image_paths=train_paths,
            labels=y_train,
            cache_path=_feature_cache_file(cache_root, config.feature_name, "train_final", train_paths),
            image_size=config.image_size,
            force_recompute=config.force_recompute_features,
        )
        X_eval = _compute_features_for_split(
            feature_name=config.feature_name,
            image_paths=eval_paths,
            labels=y_eval,
            cache_path=_feature_cache_file(cache_root, config.feature_name, config.eval_split, eval_paths),
            image_size=config.image_size,
            force_recompute=config.force_recompute_features,
        )
        notes = ""
    elif config.feature_name == "sift_bovw":
        codebook_path = cache_root / f"sift_bovw_codebook_k{config.sift_vocab_size}.joblib"
        if codebook_path.exists() and not config.force_recompute_features:
            extractor = SIFTBOVWExtractor.load(codebook_path)
        else:
            extractor = SIFTBOVWExtractor(
                SIFTBOVWConfig(vocab_size=config.sift_vocab_size, image_size=config.image_size)
            )
            extractor.fit(train_paths)
            extractor.save(codebook_path)

        train_cache = _feature_cache_file(cache_root, "sift_bovw", "train_final", train_paths)
        eval_cache = _feature_cache_file(cache_root, "sift_bovw", config.eval_split, eval_paths)

        if train_cache.exists() and not config.force_recompute_features:
            X_train, _, _ = load_feature_cache(train_cache)
        else:
            X_train = extractor.transform(train_paths)
            save_feature_cache(train_cache, X=X_train, y=y_train, filepaths=train_paths)

        if eval_cache.exists() and not config.force_recompute_features:
            X_eval, _, _ = load_feature_cache(eval_cache)
        else:
            X_eval = extractor.transform(eval_paths)
            save_feature_cache(eval_cache, X=X_eval, y=y_eval, filepaths=eval_paths)

        notes = f"bovw_vocab={config.sift_vocab_size}"
    else:
        raise ValueError(f"Unsupported feature: {config.feature_name}")

    # Model training + timing is separated from inference timing.
    model = build_classifier(config.classifier_name, seed=config.seed)
    with timed() as train_time:
        model.fit(X_train, y_train)
    with timed() as infer_time:
        y_pred, _ = predict(model, X_eval)

    infer_ms_per_image = (infer_time["seconds"] / len(y_eval) * 1000.0) if len(y_eval) else 0.0
    metrics = compute_metrics(y_eval, y_pred)

    run_name = f"{config.feature_name}_{config.classifier_name}_{config.eval_split}"
    model_path = outputs_models_root / f"{run_name}.joblib"
    save_model(model, model_path)

    labels = sorted(manifest_df["class_name"].dropna().unique().tolist())
    per_class_df = per_class_table(y_eval, y_pred, labels=labels)
    per_class_df.to_csv(outputs_tables_root / f"per_class_{run_name}.csv", index=False)

    # Persist per-sample predictions for downstream error analysis.
    pred_df = pd.DataFrame(
        {
            "filepath": eval_paths,
            "y_true": y_eval,
            "y_pred": y_pred,
            "is_correct": (y_eval == y_pred),
        }
    )
    pred_path = outputs_tables_root / f"predictions_{run_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    save_confusion_matrix(
        y_true=y_eval,
        y_pred=y_pred,
        labels=labels,
        path=outputs_figures_root / f"confusion_{run_name}.png",
        title=f"Confusion Matrix: {run_name}",
    )

    if notes:
        notes_parts.append(notes)

    return {
        "feature": config.feature_name,
        "classifier": config.classifier_name,
        "eval_split": config.eval_split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "primary_metric_macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "train_time_sec": train_time["seconds"],
        "inference_time_ms_per_image": infer_ms_per_image,
        "notes": ";".join(notes_parts),
        "model_path": str(model_path.resolve()),
        "predictions_path": str(pred_path.resolve()),
    }


def run_traditional_matrix(
    manifest_df: pd.DataFrame,
    cache_root: Path,
    outputs_models_root: Path,
    outputs_tables_root: Path,
    outputs_figures_root: Path,
    seed: int = 42,
    eval_split: str = "val_final",
    image_size: tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Run full feature/classifier matrix and export ranked summary table."""
    rows = []
    for feature_name in FEATURE_METHODS:
        for classifier_name in TRADITIONAL_CLASSIFIERS:
            cfg = TraditionalTrainConfig(
                feature_name=feature_name,
                classifier_name=classifier_name,
                image_size=image_size,
                seed=seed,
                eval_split=eval_split,
            )
            row = run_traditional_experiment(
                manifest_df=manifest_df,
                config=cfg,
                cache_root=cache_root,
                outputs_models_root=outputs_models_root,
                outputs_tables_root=outputs_tables_root,
                outputs_figures_root=outputs_figures_root,
            )
            rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    outputs_tables_root.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(outputs_tables_root / "traditional_results.csv", index=False)
    if not result_df.empty:
        result_df.head(1).to_csv(outputs_tables_root / "traditional_best_by_macro_f1.csv", index=False)
    return result_df
