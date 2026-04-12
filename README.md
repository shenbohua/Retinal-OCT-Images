# Retinal-OCT-Images

Coursework pipeline for retinal OCT classification with:
- Data audit and patient-aware split manifest
- Traditional feature extraction (HOG / LBP / SIFT-BoVW)
- Baseline classifiers (Linear SVM / RBF SVM / Random Forest)
- Lightweight deep-learning framework interfaces for team integration

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

1. Build data audit and split manifest:
```bash
python main.py audit --val-fraction 0.15 --seed 42
```

2. Run one baseline:
```bash
python main.py train-trad --feature hog --classifier linear_svm --eval-split val_final
```

3. Run full traditional matrix:
```bash
python main.py train-matrix
```

4. Optional sanity check on tiny raw val:
```bash
python main.py train-trad --feature hog --classifier linear_svm --eval-split val_raw_holdout
```

5. Final report evaluation on official test split:
```bash
python main.py final-test --feature hog --classifier linear_svm --confirm-final-report
```

6. Build report-ready analysis artifacts:
```bash
python main.py analysis-artifacts --run-name hog_linear_svm_test_final --top-pairs 3 --samples-per-pair 8
```

7. Run DL framework scaffold (for B to plug into):
```bash
python main.py train-dl --model resnet18 --eval-split val_final --epochs 5 --batch-size 32
```

8. Quick DL smoke run (small subset):
```bash
python main.py train-dl --model resnet18 --eval-split val_final --epochs 1 --batch-size 16 --image-size 128 --max-train-samples 2000 --max-eval-samples 600
```

9. Export processed dataset for teammate sharing:
```bash
python main.py export-processed --profile oct224_gray_png_v1 --image-size 224 --output-format png --splits train_final,val_final,test_final
```

## Key outputs

- `outputs/tables/dataset_audit.csv`
- `outputs/tables/split_manifest.csv`
- `outputs/tables/data_audit_report.md`
- `outputs/tables/traditional_results.csv`
- `outputs/tables/traditional_best_by_macro_f1.csv`
- `outputs/tables/traditional_summary_sorted.csv`
- `outputs/tables/comparison_with_dl_template.csv`
- `outputs/tables/report_policy_notes.md`
- `outputs/tables/result_dl_*.csv`
- `outputs/tables/history_dl_*.csv`
- `outputs/tables/processed_manifest_*.csv`
- `outputs/figures/confusion_*.png`
- `outputs/figures/errors_*.png`

## Data folders for collaboration

- `data/raw`: immutable original dataset downloaded from source.
- `data/interim`: regenerable intermediate caches (e.g., handcrafted feature `.npz`).
- `data/processed`: shareable, manifest-aligned processed files exported by `export-processed`.

## Evaluation policy

- Hyperparameter tuning uses `train_final -> val_final` only.
- Primary model selection metric is **Macro-F1**.
- `test_final` is reserved for final reporting only.
- `val_raw_holdout` is a sanity-check split, not for core model selection.
