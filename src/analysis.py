from __future__ import annotations

"""Post-run analysis artifacts for reporting and comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def build_traditional_summary(tables_root: Path) -> pd.DataFrame:
    """Aggregate result CSVs and sort by Macro-F1."""
    frames = []
    for path in sorted(tables_root.glob("result_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df.assign(result_file=path.name, run_type="validation"))
    for path in sorted(tables_root.glob("final_test_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df.assign(result_file=path.name, run_type="final_test"))

    if not frames:
        return pd.DataFrame()

    summary = pd.concat(frames, ignore_index=True, sort=False)
    return summary.sort_values(by=["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)


def build_dl_comparison_template(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a unified comparison table with placeholders for DL results."""
    keep_cols = [
        "run_type",
        "feature",
        "classifier",
        "eval_split",
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "train_time_sec",
        "inference_time_ms_per_image",
    ]
    base = summary_df[keep_cols].copy()
    trad = base[base["feature"] != "end_to_end"].copy()
    trad.insert(0, "family", "traditional")
    trad.insert(1, "model_name", trad["feature"] + "+" + trad["classifier"])

    dl_actual = base[base["feature"] == "end_to_end"].copy()
    if not dl_actual.empty:
        dl_actual.insert(0, "family", "deep_learning")
        dl_actual.insert(1, "model_name", dl_actual["classifier"])

    dl_rows = pd.DataFrame(
        [
            {
                "family": "deep_learning",
                "model_name": "resnet18_todo",
                "run_type": "validation",
                "feature": "end_to_end",
                "classifier": "n/a",
                "eval_split": "val_final",
                "accuracy": "",
                "macro_f1": "",
                "macro_precision": "",
                "macro_recall": "",
                "train_time_sec": "",
                "inference_time_ms_per_image": "",
            },
            {
                "family": "deep_learning",
                "model_name": "vgg16_todo",
                "run_type": "validation",
                "feature": "end_to_end",
                "classifier": "n/a",
                "eval_split": "val_final",
                "accuracy": "",
                "macro_f1": "",
                "macro_precision": "",
                "macro_recall": "",
                "train_time_sec": "",
                "inference_time_ms_per_image": "",
            },
        ]
    )
    existing_dl_names = set(dl_actual["model_name"].tolist()) if not dl_actual.empty else set()
    dl_rows = dl_rows[~dl_rows["model_name"].isin(existing_dl_names)]

    frames = [trad]
    if not dl_actual.empty:
        frames.append(dl_actual)
    frames.append(dl_rows)
    return pd.concat(frames, ignore_index=True)


def write_policy_note(path: Path) -> None:
    """Write evaluation protocol notes for report consistency."""
    text = "\n".join(
        [
            "# Evaluation Protocol Notes",
            "",
            "- Hyperparameter tuning and model selection use `train_final -> val_final` only.",
            "- Primary selection metric is **Macro-F1**.",
            "- `test_final` is used exactly once for final reporting.",
            "- `val_raw_holdout` is used for sanity checks only.",
            "- `train_final` and `val_final` patient overlap is zero.",
            "- Official train/test patient overlap exists; this is a reported limitation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _top_confusion_pairs(pred_df: pd.DataFrame, top_n_pairs: int) -> pd.DataFrame:
    wrong = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
    if wrong.empty:
        return pd.DataFrame(columns=["y_true", "y_pred", "count"])
    pair_counts = (
        wrong.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .head(top_n_pairs)
    )
    return pair_counts


def save_error_gallery(
    pred_df: pd.DataFrame,
    figures_root: Path,
    run_name: str,
    top_n_pairs: int = 3,
    samples_per_pair: int = 8,
) -> list[Path]:
    """Save misclassification galleries for the most frequent confusion pairs."""
    figures_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    pair_counts = _top_confusion_pairs(pred_df, top_n_pairs=top_n_pairs)
    if pair_counts.empty:
        return saved

    wrong = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
    for _, row in pair_counts.iterrows():
        y_true, y_pred = row["y_true"], row["y_pred"]
        sub = wrong[(wrong["y_true"] == y_true) & (wrong["y_pred"] == y_pred)].head(samples_per_pair)
        n = len(sub)
        if n == 0:
            continue

        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (_, sample) in enumerate(sub.iterrows()):
            ax = axes[i]
            img = Image.open(sample["filepath"]).convert("L")
            ax.imshow(img, cmap="gray")
            ax.set_title(f"T:{sample['y_true']} -> P:{sample['y_pred']}")
            ax.axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Top Misclassification: {y_true} -> {y_pred} ({n} samples)")
        fig.tight_layout()
        out = figures_root / f"errors_{run_name}_{y_true}_to_{y_pred}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        saved.append(out)

    return saved
