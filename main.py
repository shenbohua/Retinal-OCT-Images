from __future__ import annotations

"""CLI entrypoint for data audit and traditional baseline experiments.

Design choice:
- Keep one stable entry (`main.py`) for reproducibility in team workflows.
- Route each sub-command to a focused function so teammates can extend safely.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis import (
    build_dl_comparison_template,
    build_traditional_summary,
    save_error_gallery,
    write_policy_note,
)
from src.config import ensure_dirs, make_paths, resolve_oct_root
from src.data import (
    ScanOptions,
    build_audit_report_markdown,
    build_split_manifest,
    patient_overlap_table,
    scan_dataset,
)
from src.dl_models import DLRunConfig
from src.dl_train import run_dl_experiment
from src.processed import export_processed_dataset
from src.train import TraditionalTrainConfig, run_traditional_experiment, run_traditional_matrix
from src.utils import dump_json, set_seed, setup_logger


def _save_distribution_figure(manifest_df: pd.DataFrame, out_path: Path) -> None:
    """Save a report-ready class-distribution bar chart by final split."""
    pivot = manifest_df.groupby(["split_final", "class_name"]).size().unstack(fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_xlabel("Split")
    ax.set_ylabel("Image count")
    ax.set_title("Class Distribution by Final Split")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_distribution_figure_logy(manifest_df: pd.DataFrame, out_path: Path) -> None:
    """Save distribution chart on log y-axis to avoid tiny splits being invisible."""
    pivot = manifest_df.groupby(["split_final", "class_name"]).size().unstack(fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_yscale("log")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image count (log scale)")
    ax.set_title("Class Distribution by Final Split (log y)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_distribution_figure_faceted(manifest_df: pd.DataFrame, out_path: Path) -> None:
    """Save split-faceted bar charts for easier per-split inspection."""
    pivot = manifest_df.groupby(["split_final", "class_name"]).size().unstack(fill_value=0)
    splits = list(pivot.index)
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        values = pivot.loc[split_name]
        ax.bar(values.index.tolist(), values.values.tolist())
        ax.set_title(split_name)
        ax.set_xlabel("Class")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Image count")

    fig.suptitle("Class Distribution by Split (Faceted)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def command_audit(args: argparse.Namespace) -> None:
    """Build dataset audit artifacts and patient-aware split manifest."""
    paths = make_paths()
    ensure_dirs(
        [
            paths.tables_root,
            paths.figures_root,
            paths.models_root,
            paths.logs_root,
            paths.interim_root,
        ]
    )
    logger = setup_logger(paths.logs_root / "audit.log")
    set_seed(args.seed)

    oct_root = resolve_oct_root(paths.data_root)
    logger.info("Scanning dataset at: %s", oct_root)
    audit_df = scan_dataset(
        oct_root=oct_root,
        options=ScanOptions(
            compute_sha1=args.compute_sha1,
            max_files_per_class=args.max_files_per_class,
        ),
    )
    audit_path = paths.tables_root / "dataset_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    logger.info("Saved dataset audit: %s", audit_path)

    manifest_df = build_split_manifest(
        audit_df=audit_df,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    manifest_path = paths.tables_root / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Saved split manifest: %s", manifest_path)

    overlap_df = patient_overlap_table(manifest_df)
    overlap_path = paths.tables_root / "patient_overlap_check.csv"
    overlap_df.to_csv(overlap_path, index=False)
    logger.info("Saved patient overlap table: %s", overlap_path)

    report_text = build_audit_report_markdown(
        audit_df=audit_df,
        manifest_df=manifest_df,
        overlap_df=overlap_df,
        val_fraction=args.val_fraction,
    )
    report_path = paths.tables_root / "data_audit_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved audit report: %s", report_path)

    figure_path = paths.figures_root / "data_distribution.png"
    _save_distribution_figure(manifest_df, figure_path)
    logger.info("Saved distribution figure: %s", figure_path)
    figure_log_path = paths.figures_root / "data_distribution_logy.png"
    _save_distribution_figure_logy(manifest_df, figure_log_path)
    logger.info("Saved log-scale distribution figure: %s", figure_log_path)
    figure_facet_path = paths.figures_root / "data_distribution_faceted.png"
    _save_distribution_figure_faceted(manifest_df, figure_facet_path)
    logger.info("Saved faceted distribution figure: %s", figure_facet_path)

    dump_json(
        {
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "compute_sha1": args.compute_sha1,
            "max_files_per_class": args.max_files_per_class,
            "oct_root": str(oct_root.resolve()),
        },
        paths.logs_root / "audit_config.json",
    )


def command_train_trad(args: argparse.Namespace) -> None:
    """Run one traditional baseline experiment (feature + classifier)."""
    paths = make_paths()
    ensure_dirs(
        [
            paths.tables_root,
            paths.figures_root,
            paths.models_root,
            paths.logs_root,
            paths.interim_root,
        ]
    )
    logger = setup_logger(paths.logs_root / "train_traditional.log")
    set_seed(args.seed)

    manifest_path = paths.tables_root / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.csv not found. Run `python main.py audit` first.")

    manifest_df = pd.read_csv(manifest_path)
    if args.eval_split == "test_final":
        raise ValueError(
            "`test_final` is reserved for final reporting only. "
            "Use `python main.py final-test ... --confirm-final-report`."
        )
    cfg = TraditionalTrainConfig(
        feature_name=args.feature,
        classifier_name=args.classifier,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        sift_vocab_size=args.sift_vocab_size,
        force_recompute_features=args.force_recompute_features,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    result = run_traditional_experiment(
        manifest_df=manifest_df,
        config=cfg,
        cache_root=paths.interim_root / "features",
        outputs_models_root=paths.models_root,
        outputs_tables_root=paths.tables_root,
        outputs_figures_root=paths.figures_root,
    )
    result_df = pd.DataFrame([result])
    out_path = paths.tables_root / f"result_{args.feature}_{args.classifier}_{args.eval_split}.csv"
    result_df.to_csv(out_path, index=False)
    logger.info("Saved result row: %s", out_path)


def command_final_test(args: argparse.Namespace) -> None:
    """Run final-report evaluation on official test split (single-run intent)."""
    if not args.confirm_final_report:
        raise ValueError("Add `--confirm-final-report` to run final test evaluation.")

    paths = make_paths()
    ensure_dirs(
        [
            paths.tables_root,
            paths.figures_root,
            paths.models_root,
            paths.logs_root,
            paths.interim_root,
        ]
    )
    logger = setup_logger(paths.logs_root / "final_test.log")
    set_seed(args.seed)

    manifest_path = paths.tables_root / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.csv not found. Run `python main.py audit` first.")

    manifest_df = pd.read_csv(manifest_path)
    cfg = TraditionalTrainConfig(
        feature_name=args.feature,
        classifier_name=args.classifier,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        sift_vocab_size=args.sift_vocab_size,
        force_recompute_features=args.force_recompute_features,
        eval_split="test_final",
    )

    result = run_traditional_experiment(
        manifest_df=manifest_df,
        config=cfg,
        cache_root=paths.interim_root / "features",
        outputs_models_root=paths.models_root,
        outputs_tables_root=paths.tables_root,
        outputs_figures_root=paths.figures_root,
    )
    result_df = pd.DataFrame([result])
    out_path = paths.tables_root / f"final_test_{args.feature}_{args.classifier}.csv"
    result_df.to_csv(out_path, index=False)
    logger.info("Saved final-test result row: %s", out_path)


def command_train_matrix(args: argparse.Namespace) -> None:
    """Run the full 3x3 traditional experiment matrix."""
    paths = make_paths()
    ensure_dirs(
        [
            paths.tables_root,
            paths.figures_root,
            paths.models_root,
            paths.logs_root,
            paths.interim_root,
        ]
    )
    logger = setup_logger(paths.logs_root / "train_matrix.log")
    set_seed(args.seed)

    manifest_path = paths.tables_root / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.csv not found. Run `python main.py audit` first.")

    manifest_df = pd.read_csv(manifest_path)
    result_df = run_traditional_matrix(
        manifest_df=manifest_df,
        cache_root=paths.interim_root / "features",
        outputs_models_root=paths.models_root,
        outputs_tables_root=paths.tables_root,
        outputs_figures_root=paths.figures_root,
        seed=args.seed,
        eval_split="val_final",
        image_size=(args.image_size, args.image_size),
    )
    logger.info("Saved matrix results with %d rows", len(result_df))


def command_train_dl(args: argparse.Namespace) -> None:
    """Run DL framework training on train_final -> eval_split."""
    paths = make_paths()
    ensure_dirs(
        [
            paths.tables_root,
            paths.figures_root,
            paths.models_root,
            paths.logs_root,
            paths.interim_root,
        ]
    )
    logger = setup_logger(paths.logs_root / "train_dl.log")
    set_seed(args.seed)

    manifest_path = paths.tables_root / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.csv not found. Run `python main.py audit` first.")

    manifest_df = pd.read_csv(manifest_path)
    cfg = DLRunConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        checkpoint_dir=paths.models_root,
        random_seed=args.seed,
        use_class_weight=(not args.no_class_weight),
        device=args.device,
    )
    result = run_dl_experiment(
        manifest_df=manifest_df,
        config=cfg,
        outputs_models_root=paths.models_root,
        outputs_tables_root=paths.tables_root,
        outputs_figures_root=paths.figures_root,
    )
    out_path = paths.tables_root / f"result_dl_{args.model}_{args.eval_split}.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    logger.info("Saved DL result row: %s", out_path)


def command_analysis_artifacts(args: argparse.Namespace) -> None:
    """Generate summary tables, DL template, policy note, and error galleries."""
    paths = make_paths()
    ensure_dirs([paths.tables_root, paths.figures_root, paths.logs_root])
    logger = setup_logger(paths.logs_root / "analysis_artifacts.log")

    summary_df = build_traditional_summary(paths.tables_root)
    if summary_df.empty:
        raise FileNotFoundError("No result_*.csv files found in outputs/tables.")

    summary_path = paths.tables_root / "traditional_summary_sorted.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary table: %s", summary_path)

    all_val_summary = summary_df[summary_df["eval_split"] == "val_final"].copy()
    all_val_summary_path = paths.tables_root / "all_validation_summary_sorted.csv"
    all_val_summary.to_csv(all_val_summary_path, index=False)
    logger.info("Saved all-validation summary table: %s", all_val_summary_path)

    val_summary = all_val_summary[all_val_summary["feature"] != "end_to_end"].copy()
    val_summary_path = paths.tables_root / "traditional_validation_summary_sorted.csv"
    val_summary.to_csv(val_summary_path, index=False)
    logger.info("Saved validation-only summary table: %s", val_summary_path)

    dl_template = build_dl_comparison_template(summary_df)
    dl_template_path = paths.tables_root / "comparison_with_dl_template.csv"
    dl_template.to_csv(dl_template_path, index=False)
    logger.info("Saved DL comparison template: %s", dl_template_path)

    policy_path = paths.tables_root / "report_policy_notes.md"
    write_policy_note(policy_path)
    logger.info("Saved policy notes: %s", policy_path)

    run_name = args.run_name
    pred_path = paths.tables_root / f"predictions_{run_name}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. "
            "Re-run the target experiment to generate predictions."
        )

    pred_df = pd.read_csv(pred_path)
    saved = save_error_gallery(
        pred_df=pred_df,
        figures_root=paths.figures_root,
        run_name=run_name,
        top_n_pairs=args.top_pairs,
        samples_per_pair=args.samples_per_pair,
    )
    logger.info("Saved %d error gallery figures.", len(saved))


def command_export_processed(args: argparse.Namespace) -> None:
    """Export manifest-driven processed dataset files for team collaboration."""
    paths = make_paths()
    ensure_dirs([paths.tables_root, paths.logs_root, paths.processed_root])
    logger = setup_logger(paths.logs_root / "export_processed.log")

    manifest_path = paths.tables_root / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.csv not found. Run `python main.py audit` first.")
    manifest_df = pd.read_csv(manifest_path)

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    processed_df = export_processed_dataset(
        manifest_df=manifest_df,
        processed_root=paths.processed_root,
        profile=args.profile,
        image_size=(args.image_size, args.image_size),
        output_format=args.output_format,
        rgb=args.rgb,
        normalize=args.normalize,
        splits=split_list,
        overwrite=args.overwrite,
        max_files=args.max_files,
    )
    out_manifest = paths.tables_root / f"processed_manifest_{args.profile}.csv"
    processed_df.to_csv(out_manifest, index=False)
    logger.info("Saved processed manifest: %s", out_manifest)
    logger.info(
        "Processed export complete. profile=%s, files=%d, root=%s",
        args.profile,
        len(processed_df),
        paths.processed_root / args.profile,
    )


def build_parser() -> argparse.ArgumentParser:
    """Define CLI commands used in coursework experiments."""
    parser = argparse.ArgumentParser(description="Retinal OCT coursework pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Scan dataset and build split manifest")
    audit.add_argument("--compute-sha1", action="store_true", help="Compute SHA1 per image")
    audit.add_argument("--max-files-per-class", type=int, default=None, help="Debug option for quick runs")
    audit.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction from raw train split")
    audit.add_argument("--seed", type=int, default=42)
    audit.set_defaults(func=command_audit)

    train_trad = subparsers.add_parser("train-trad", help="Run one traditional feature+classifier experiment")
    train_trad.add_argument("--feature", choices=["hog", "lbp", "sift_bovw"], required=True)
    train_trad.add_argument("--classifier", choices=["linear_svm", "rbf_svm", "random_forest"], required=True)
    train_trad.add_argument("--eval-split", default="val_final", choices=["val_final", "val_raw_holdout"])
    train_trad.add_argument("--image-size", type=int, default=224)
    train_trad.add_argument("--sift-vocab-size", type=int, default=128)
    train_trad.add_argument("--force-recompute-features", action="store_true")
    train_trad.add_argument("--max-train-samples", type=int, default=None)
    train_trad.add_argument("--max-eval-samples", type=int, default=None)
    train_trad.add_argument("--seed", type=int, default=42)
    train_trad.set_defaults(func=command_train_trad)

    final_test = subparsers.add_parser("final-test", help="Run final report evaluation on test_final")
    final_test.add_argument("--feature", choices=["hog", "lbp", "sift_bovw"], required=True)
    final_test.add_argument("--classifier", choices=["linear_svm", "rbf_svm", "random_forest"], required=True)
    final_test.add_argument("--image-size", type=int, default=224)
    final_test.add_argument("--sift-vocab-size", type=int, default=128)
    final_test.add_argument("--force-recompute-features", action="store_true")
    final_test.add_argument("--seed", type=int, default=42)
    final_test.add_argument("--confirm-final-report", action="store_true")
    final_test.set_defaults(func=command_final_test)

    matrix = subparsers.add_parser("train-matrix", help="Run full traditional experiment matrix")
    matrix.add_argument("--image-size", type=int, default=224)
    matrix.add_argument("--seed", type=int, default=42)
    matrix.set_defaults(func=command_train_matrix)

    train_dl = subparsers.add_parser("train-dl", help="Run DL framework (ResNet/VGG scaffold)")
    train_dl.add_argument("--model", choices=["resnet18", "resnet34", "vgg16"], default="resnet18")
    train_dl.add_argument("--eval-split", choices=["val_final", "val_raw_holdout"], default="val_final")
    train_dl.add_argument("--epochs", type=int, default=5)
    train_dl.add_argument("--batch-size", type=int, default=32)
    train_dl.add_argument("--learning-rate", type=float, default=1e-3)
    train_dl.add_argument("--weight-decay", type=float, default=1e-4)
    train_dl.add_argument("--image-size", type=int, default=224)
    train_dl.add_argument("--num-workers", type=int, default=0)
    train_dl.add_argument("--max-train-samples", type=int, default=None)
    train_dl.add_argument("--max-eval-samples", type=int, default=None)
    train_dl.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train_dl.add_argument("--no-class-weight", action="store_true")
    train_dl.add_argument("--seed", type=int, default=42)
    train_dl.set_defaults(func=command_train_dl)

    analysis = subparsers.add_parser("analysis-artifacts", help="Build report-ready analysis artifacts")
    analysis.add_argument(
        "--run-name",
        default="hog_linear_svm_test_final",
        help="Run suffix used in predictions_<run-name>.csv",
    )
    analysis.add_argument("--top-pairs", type=int, default=3, help="Number of top confusion pairs to visualize")
    analysis.add_argument("--samples-per-pair", type=int, default=8, help="Samples per confusion pair figure")
    analysis.set_defaults(func=command_analysis_artifacts)

    export_processed = subparsers.add_parser(
        "export-processed",
        help="Export processed dataset files (for teammate sharing)",
    )
    export_processed.add_argument("--profile", type=str, default="oct224_gray_png_v1")
    export_processed.add_argument("--image-size", type=int, default=224)
    export_processed.add_argument("--output-format", choices=["png", "npy"], default="png")
    export_processed.add_argument("--rgb", action="store_true", help="Export 3-channel images")
    export_processed.add_argument(
        "--normalize",
        action="store_true",
        help="Store normalized float arrays (effective for npy only)",
    )
    export_processed.add_argument(
        "--splits",
        type=str,
        default="train_final,val_final,test_final",
        help="Comma-separated split_final names to export",
    )
    export_processed.add_argument("--overwrite", action="store_true")
    export_processed.add_argument("--max-files", type=int, default=None)
    export_processed.set_defaults(func=command_export_processed)

    return parser


def main() -> None:
    """Program entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
