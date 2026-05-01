from __future__ import annotations

"""End-to-end orchestration for DL framework training and evaluation."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES
from src.dl_models import DLFramework, DLRunConfig
from src.evaluate import compute_metrics, per_class_table, save_confusion_matrix
from src.utils import timed


def _select_rows(manifest_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Select one split from manifest and fail loudly when empty."""
    selected = manifest_df[manifest_df["split_final"] == split_name].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split_final={split_name}.")
    return selected


def _stratified_cap(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Cap a dataframe with class-aware sampling for quick DL smoke runs."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df

    groups: list[pd.DataFrame] = []
    total = len(df)
    for _class_name, g in df.groupby("class_name"):
        n = max(1, int(round(len(g) / total * max_samples)))
        groups.append(g.sample(n=min(n, len(g)), random_state=seed))
    sampled = pd.concat(groups, axis=0)

    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    elif len(sampled) < max_samples:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            extra_n = min(max_samples - len(sampled), len(remaining))
            sampled = pd.concat([sampled, remaining.sample(n=extra_n, random_state=seed)], axis=0)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_dl_experiment(
    manifest_df: pd.DataFrame,
    config: DLRunConfig,
    outputs_models_root: Path,
    outputs_tables_root: Path,
    outputs_figures_root: Path,
) -> dict[str, float | str]:
    """Train DL model and export report-ready artifacts."""
    train_df = _select_rows(manifest_df, "train_final")
    eval_df = _select_rows(manifest_df, config.eval_split)
    train_df = _stratified_cap(train_df, config.max_train_samples, config.random_seed)
    eval_df = _stratified_cap(eval_df, config.max_eval_samples, config.random_seed)

    train_paths = train_df["filepath"].tolist()
    eval_paths = eval_df["filepath"].tolist()
    train_classes = train_df["class_name"].tolist()
    eval_classes = eval_df["class_name"].tolist()

    framework = DLFramework(config=config)
    model = framework.build_model(num_classes=len(CLASS_NAMES))
    optimizer = framework.make_optimizer(model)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_train_idx = np.array([class_to_idx[c] for c in train_classes], dtype=np.int64)
    class_weights = (
        framework.class_weight_from_labels(y_train_idx, num_classes=len(CLASS_NAMES))
        if config.use_class_weight
        else None
    )
    criterion = framework.make_criterion(class_weights=class_weights)

    train_loader = framework.make_loader(
        filepaths=train_paths,
        class_names=train_classes,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=True,
        shuffle=True,
    )
    eval_loader = framework.make_loader(
        filepaths=eval_paths,
        class_names=eval_classes,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=False,
        shuffle=False,
    )

    outputs_models_root.mkdir(parents=True, exist_ok=True)
    outputs_tables_root.mkdir(parents=True, exist_ok=True)
    outputs_figures_root.mkdir(parents=True, exist_ok=True)

    run_name = f"dl_{config.model_name}_{config.eval_split}"
    checkpoint_path = outputs_models_root / f"{run_name}_best.pt"
    config_dump = json.loads(json.dumps(asdict(config), default=str))
    history_rows: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0

    with timed() as train_timer:
        for epoch in range(1, config.epochs + 1):
            train_metrics = framework.train_one_epoch(model, train_loader, optimizer, criterion)
            eval_metrics = framework.validate(model, eval_loader, criterion)
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": eval_metrics["loss"],
                    "val_macro_f1": eval_metrics["macro_f1"],
                    "val_accuracy": eval_metrics["accuracy"],
                }
            )
            if eval_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = eval_metrics["macro_f1"]
                best_epoch = epoch
                framework.save_checkpoint(
                    state={
                        "epoch": epoch,
                        "model_name": config.model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_macro_f1": best_macro_f1,
                        "config": config_dump,
                    },
                    filename=checkpoint_path.name,
                )

    # Load best checkpoint before final evaluation export.
    import torch

    ckpt = torch.load(checkpoint_path, map_location=framework.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    with timed() as infer_timer:
        y_true_idx, y_pred_idx, pred_paths = framework.predict(model, eval_loader)

    y_true = np.array([CLASS_NAMES[i] for i in y_true_idx])
    y_pred = np.array([CLASS_NAMES[i] for i in y_pred_idx])
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    infer_ms_per_image = (infer_timer["seconds"] / len(y_true) * 1000.0) if len(y_true) else 0.0

    history_path = outputs_tables_root / f"history_{run_name}.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    per_class_df = per_class_table(y_true=y_true, y_pred=y_pred, labels=CLASS_NAMES)
    per_class_path = outputs_tables_root / f"per_class_{run_name}.csv"
    per_class_df.to_csv(per_class_path, index=False)

    pred_df = pd.DataFrame(
        {
            "filepath": pred_paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "is_correct": (y_true == y_pred),
        }
    )
    pred_path = outputs_tables_root / f"predictions_{run_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=CLASS_NAMES,
        path=outputs_figures_root / f"confusion_{run_name}.png",
        title=f"Confusion Matrix: {run_name}",
    )

    return {
        "feature": "end_to_end",
        "classifier": config.model_name,
        "eval_split": config.eval_split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "primary_metric_macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "train_time_sec": train_timer["seconds"],
        "inference_time_ms_per_image": infer_ms_per_image,
        "notes": f"best_epoch={best_epoch};batch_size={config.batch_size};image_size={config.image_size}",
        "model_path": str(checkpoint_path.resolve()),
        "predictions_path": str(pred_path.resolve()),
        "history_path": str(history_path.resolve()),
    }
