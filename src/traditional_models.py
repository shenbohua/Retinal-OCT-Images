from __future__ import annotations

"""Traditional classifier factory and model I/O helpers."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC


def build_classifier(name: str, seed: int = 42) -> Any:
    """Create baseline classifier by name."""
    key = name.lower()
    if key == "linear_svm":
        return LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    if key == "rbf_svm":
        return SVC(
            C=5.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def fit_predict(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit model on training data and predict on evaluation data."""
    model.fit(X_train, y_train)
    return predict(model, X_eval)


def predict(model: Any, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict labels and return confidence scores when available."""
    pred = model.predict(X_eval)

    if hasattr(model, "predict_proba"):
        return pred, model.predict_proba(X_eval)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_eval)
        return pred, score
    return pred, None


def save_model(model: Any, path: str | Path) -> None:
    """Serialize trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
