from __future__ import annotations

"""Deep-learning framework for team integration (ResNet/VGG training scaffold)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES
from src.evaluate import compute_metrics

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class DLRunConfig:
    model_name: str = "resnet18"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 0
    eval_split: str = "val_final"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    checkpoint_dir: Path = Path("outputs/models")
    checkpoint_name: str = "dl_best.pt"
    random_seed: int = 42
    use_class_weight: bool = True
    device: str = "auto"


def _require_torch() -> None:
    """Raise actionable import error when torch stack is unavailable."""
    if torch is None or nn is None or models is None or transforms is None:
        raise ImportError(
            "PyTorch stack is required for DL commands. "
            "Install `torch` and `torchvision` first."
        ) from _TORCH_IMPORT_ERROR


class OCTManifestDataset(Dataset):
    """Dataset backed by split_manifest rows (filepath + class_name)."""

    def __init__(
        self,
        filepaths: list[str],
        class_names: list[str],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        _require_torch()
        self.filepaths = filepaths
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.augment = augment
        self.transform = self._build_transform(image_size=image_size, augment=augment)

    @staticmethod
    def _build_transform(image_size: int, augment: bool) -> Any:
        _require_torch()
        ops = [
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=8),
                ]
            )
        ops.extend(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[Any, int, str]:
        path = self.filepaths[idx]
        class_name = self.class_names[idx]
        label = self.class_to_idx[class_name]
        with Image.open(path) as img:
            image = img.convert("L")
        x = self.transform(image)
        return x, label, path


class DLFramework:
    """Reusable DL runner: model factory, loops, metrics, and checkpointing."""

    def __init__(self, config: DLRunConfig) -> None:
        _require_torch()
        self.config = config
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        _require_torch()
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def build_model(self, num_classes: int) -> Any:
        _require_torch()
        name = self.config.model_name.lower()
        if name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model.to(self.device)
        if name == "resnet34":
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model.to(self.device)
        if name == "vgg16":
            model = models.vgg16(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model.to(self.device)
        raise ValueError(f"Unsupported model_name: {self.config.model_name}")

    def make_optimizer(self, model: Any) -> Any:
        _require_torch()
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def make_criterion(self, class_weights: np.ndarray | None = None) -> Any:
        _require_torch()
        if class_weights is None:
            return nn.CrossEntropyLoss()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        return nn.CrossEntropyLoss(weight=weight_tensor)

    def train_one_epoch(self, model: Any, loader: Any, optimizer: Any, criterion: Any) -> dict[str, float]:
        _require_torch()
        model.train()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        for x, y, _paths in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())

        metrics = self._metrics_from_indices(y_true, y_pred)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    def validate(self, model: Any, loader: Any, criterion: Any) -> dict[str, float]:
        _require_torch()
        model.eval()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for x, y, _paths in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                loss = criterion(logits, y)

                losses.append(float(loss.item()))
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())

        metrics = self._metrics_from_indices(y_true, y_pred)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    def predict(self, model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return true/pred labels and filepaths for one dataloader."""
        _require_torch()
        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        all_paths: list[str] = []
        with torch.no_grad():
            for x, y, paths in loader:
                x = x.to(self.device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                all_paths.extend(list(paths))
        return np.array(y_true), np.array(y_pred), all_paths

    @staticmethod
    def class_weight_from_labels(y_indices: np.ndarray, num_classes: int) -> np.ndarray:
        """Compute inverse-frequency class weights."""
        counts = np.bincount(y_indices, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = counts.sum() / (num_classes * counts)
        return inv.astype(np.float32)

    @staticmethod
    def _metrics_from_indices(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        true_names = np.array([CLASS_NAMES[i] for i in y_true])
        pred_names = np.array([CLASS_NAMES[i] for i in y_pred])
        return compute_metrics(true_names, pred_names)

    def save_checkpoint(self, state: dict[str, Any], filename: str) -> Path:
        _require_torch()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / filename
        torch.save(state, path)
        return path

    @staticmethod
    def make_loader(
        filepaths: list[str],
        class_names: list[str],
        image_size: int,
        batch_size: int,
        num_workers: int,
        augment: bool,
        shuffle: bool,
    ) -> Any:
        """Build one DataLoader from filepath/class lists."""
        _require_torch()
        dataset = OCTManifestDataset(
            filepaths=filepaths,
            class_names=class_names,
            image_size=image_size,
            augment=augment,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
