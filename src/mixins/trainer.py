import os
import random
from typing import Mapping, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from clearml import Task
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainerConfig
from src.mixins.clearml_logger import ClearMLLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._set_seed(TrainerConfig.seed)

        self.model = model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.class_weights = class_weights

        self.criterion = None
        self.optimizer = None

        self.best_train_metrics = {}
        self.best_val_metrics = {}
        self.best_train_outputs = {}
        self.best_val_outputs = {}
        self.best_threshold = 0.5

        os.makedirs(TrainerConfig.save_dir, exist_ok=True)
        self.best_path = os.path.join(TrainerConfig.save_dir, TrainerConfig.save_best_name)

        self.best_score = -np.inf if TrainerConfig.monitor_mode == "max" else np.inf
        self.best_epoch = -1

        self.task = Task.init(
            project_name=TrainerConfig.project_name,
            task_name=TrainerConfig.task_name,
            auto_connect_frameworks={"pytorch": True},
            auto_connect_arg_parser=False,
        )
        self.logger = self.task.get_logger()
        self.clearml = ClearMLLogger(self.logger)

        self.task.connect(
            {
                "seed": TrainerConfig.seed,
                "lr": TrainerConfig.lr,
                "weight_decay": TrainerConfig.weight_decay,
                "batch_size": TrainerConfig.batch_size,
                "num_workers": TrainerConfig.num_workers,
                "epochs": TrainerConfig.epochs,
                "use_speed": TrainerConfig.use_speed,
                "monitor_metric": TrainerConfig.monitor_metric,
                "monitor_mode": TrainerConfig.monitor_mode,
                "overfit_patience": getattr(TrainerConfig, "overfit_patience", None),
                "overfit_f1_gap": getattr(TrainerConfig, "overfit_f1_gap", None),
                "overfit_loss_gap": getattr(TrainerConfig, "overfit_loss_gap", None),
                "overfit_warmup_epochs": getattr(TrainerConfig, "overfit_warmup_epochs", 0),
                "device": str(self.device),
            }
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob=None, threshold: Optional[float] = None) -> dict:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_prob is not None and len(set(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)

        if threshold is not None:
            metrics["threshold"] = threshold

        return metrics

    @staticmethod
    def _extract_probabilities(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) > 1:
            return torch.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits.view(-1))

    @staticmethod
    def _predict_from_probabilities(probs: list[float], threshold: float) -> list[int]:
        return (np.array(probs) >= threshold).astype(int).tolist()

    @staticmethod
    def _find_best_threshold_for_f1(y_true, y_prob) -> float:
        if len(set(y_true)) < 2:
            return 0.5

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if len(thresholds) == 0:
            return 0.5

        f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_scores))
        return float(thresholds[best_idx])

    def _save_checkpoint(self, epoch: int, score: float) -> None:
        serializable_config = {
            key: value for key, value in vars(TrainerConfig).items() if not key.startswith("__") and not callable(value)
        }

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": score,
            "config": serializable_config,
        }
        torch.save(payload, self.best_path)

    def _is_better(self, score: float) -> bool:
        if TrainerConfig.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _set_optimizer_lr(self, lr: float) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized. Call fit() first.")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def _copy_epoch_outputs(outputs: dict) -> dict:
        copied = {}
        for key, value in outputs.items():
            copied[key] = value.copy() if hasattr(value, "copy") else value
        return copied

    def train_one_epoch(
        self,
        epoch: int,
        train_loader,
    ) -> dict:
        self.model.train()
        if self.criterion is None or self.optimizer is None:
            raise RuntimeError("Trainer is not initialized. Call fit() before training.")

        total_loss = 0.0
        all_targets = []
        all_probs = []

        progress = tqdm(train_loader, desc=f"train {epoch}", leave=False)

        for batch in progress:
            signal = batch["signal"].to(self.device)
            target = batch["target"].to(self.device)

            speed = None
            if TrainerConfig.use_speed:
                speed = batch["speed"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(signal, speed) if TrainerConfig.use_speed else self.model(signal)
            loss = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * signal.size(0)

            probs_pos = self._extract_probabilities(logits)
            all_targets.extend(target.detach().cpu().tolist())
            all_probs.extend(probs_pos.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / len(train_loader.dataset),
            "targets": all_targets,
            "probs": all_probs,
        }

    @torch.no_grad()
    def eval_one_epoch(
        self,
        epoch: int,
        val_loader,
    ) -> dict:
        self.model.eval()

        total_loss = 0.0
        all_targets = []
        all_probs = []

        progress = tqdm(val_loader, desc=f"eval {epoch}", leave=False)

        for batch in progress:
            signal = batch["signal"].to(self.device)
            target = batch["target"].to(self.device)

            speed = None
            if TrainerConfig.use_speed:
                speed = batch["speed"].to(self.device)

            logits = self.model(signal, speed) if TrainerConfig.use_speed else self.model(signal)
            loss = self.criterion(logits, target)

            total_loss += loss.item() * signal.size(0)

            probs_pos = self._extract_probabilities(logits)
            all_targets.extend(target.detach().cpu().tolist())
            all_probs.extend(probs_pos.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / len(val_loader.dataset),
            "targets": all_targets,
            "probs": all_probs,
        }

    def fit(
        self,
        epochs: int = 20,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        loss_fn: Type[nn.Module] = nn.CrossEntropyLoss,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        class_weights: Optional[torch.Tensor] = None,
        lr_by_epoch: Optional[Mapping[int, float]] = None,
        patience: Optional[int] = None,
        overfit_patience: Optional[int] = None,
        overfit_f1_gap: Optional[float] = None,
        overfit_loss_gap: Optional[float] = None,
        overfit_warmup_epochs: Optional[int] = None,
    ):
        if class_weights is None:
            class_weights = self.class_weights
        else:
            class_weights = class_weights.to(self.device)

        effective_lr = TrainerConfig.lr if lr is None else lr
        effective_weight_decay = TrainerConfig.weight_decay if weight_decay is None else weight_decay

        self.criterion = loss_fn(weight=class_weights)
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=effective_lr,
            weight_decay=effective_weight_decay,
        )

        self.clearml.log_model_architecture(self.model, self.device, train_loader)

        if patience is not None and patience < 1:
            raise ValueError("patience must be >= 1 or None")

        effective_overfit_patience = (
            getattr(TrainerConfig, "overfit_patience", None) if overfit_patience is None else overfit_patience
        )
        effective_overfit_f1_gap = (
            getattr(TrainerConfig, "overfit_f1_gap", 0.08) if overfit_f1_gap is None else overfit_f1_gap
        )
        effective_overfit_loss_gap = (
            getattr(TrainerConfig, "overfit_loss_gap", 0.10) if overfit_loss_gap is None else overfit_loss_gap
        )
        effective_overfit_warmup_epochs = (
            getattr(TrainerConfig, "overfit_warmup_epochs", 0)
            if overfit_warmup_epochs is None
            else overfit_warmup_epochs
        )

        if effective_overfit_patience is not None and effective_overfit_patience < 1:
            raise ValueError("overfit_patience must be >= 1 or None")
        if effective_overfit_f1_gap < 0:
            raise ValueError("overfit_f1_gap must be >= 0")
        if effective_overfit_loss_gap < 0:
            raise ValueError("overfit_loss_gap must be >= 0")
        if effective_overfit_warmup_epochs < 0:
            raise ValueError("overfit_warmup_epochs must be >= 0")

        epochs_without_improvement = 0
        epochs_with_overfitting = 0

        for epoch in range(1, epochs + 1):
            if lr_by_epoch and epoch in lr_by_epoch:
                self._set_optimizer_lr(lr_by_epoch[epoch])

            train_outputs = self.train_one_epoch(epoch, train_loader)
            val_outputs = self.eval_one_epoch(epoch, val_loader)

            threshold = self._find_best_threshold_for_f1(val_outputs["targets"], val_outputs["probs"])

            train_preds = self._predict_from_probabilities(train_outputs["probs"], threshold)
            val_preds = self._predict_from_probabilities(val_outputs["probs"], threshold)

            train_metrics = self.compute_metrics(
                train_outputs["targets"],
                train_preds,
                train_outputs["probs"],
                threshold=threshold,
            )
            train_metrics["loss"] = train_outputs["loss"]

            val_metrics = self.compute_metrics(
                val_outputs["targets"],
                val_preds,
                val_outputs["probs"],
                threshold=threshold,
            )
            val_metrics["loss"] = val_outputs["loss"]

            f1_gap = float(train_metrics["f1"] - val_metrics["f1"])
            loss_gap = float(val_metrics["loss"] - train_metrics["loss"])
            self.clearml.log_metrics(
                {
                    "f1_gap": f1_gap,
                    "loss_gap": loss_gap,
                },
                stage="train_val_gap",
                epoch=epoch,
            )

            self.clearml.log_metrics(train_metrics, stage="train", epoch=epoch)
            self.clearml.log_metrics(val_metrics, stage="val", epoch=epoch)

            train_outputs["preds"] = train_preds
            val_outputs["preds"] = val_preds

            current_score = val_metrics[TrainerConfig.monitor_metric]

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_f1={train_metrics['f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_recall={val_metrics['recall']:.4f} "
                f"f1_gap={f1_gap:.4f} "
                f"loss_gap={loss_gap:.4f}"
            )

            overfit_detected = (
                effective_overfit_patience is not None
                and epoch > effective_overfit_warmup_epochs
                and f1_gap >= effective_overfit_f1_gap
                and loss_gap >= effective_overfit_loss_gap
            )

            if overfit_detected:
                epochs_with_overfitting += 1
                self.clearml.report_text(
                    f"[OVERFIT_WARN] epoch={epoch}, f1_gap={f1_gap:.4f}, loss_gap={loss_gap:.4f}, "
                    f"counter={epochs_with_overfitting}/{effective_overfit_patience}"
                )
            else:
                epochs_with_overfitting = 0

            if self._is_better(current_score):
                self.best_score = current_score
                self.best_epoch = epoch
                self.best_train_metrics = train_metrics.copy()
                self.best_val_metrics = val_metrics.copy()
                self.best_train_outputs = self._copy_epoch_outputs(train_outputs)
                self.best_val_outputs = self._copy_epoch_outputs(val_outputs)
                self.best_threshold = threshold
                self._save_checkpoint(epoch, current_score)
                epochs_without_improvement = 0

                self.clearml.report_text(
                    f"[BEST] epoch={epoch}, {TrainerConfig.monitor_metric}={current_score:.6f}, threshold={threshold:.4f}"
                )
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                message = f"[EARLY_STOP] epoch={epoch}, no improvement in {epochs_without_improvement} epoch(s)"
                print(message)
                self.clearml.report_text(message)
                break

            if (
                effective_overfit_patience is not None
                and epochs_with_overfitting >= effective_overfit_patience
            ):
                message = (
                    f"[OVERFIT_STOP] epoch={epoch}, train/val gap exceeded thresholds for "
                    f"{epochs_with_overfitting} epoch(s)"
                )
                print(message)
                self.clearml.report_text(message)
                break

        print(
            f"Best epoch: {self.best_epoch} | "
            f"best_{TrainerConfig.monitor_metric}={self.best_score:.4f} | "
            f"threshold={self.best_threshold:.4f}"
        )

        if self.best_epoch > 0:
            self.clearml.log_metrics(self.best_train_metrics, stage="train_best", epoch=self.best_epoch)
            self.clearml.log_metrics(self.best_val_metrics, stage="val_best", epoch=self.best_epoch)

            self.clearml.log_confusion_matrix(
                self.best_train_outputs["targets"],
                self.best_train_outputs["preds"],
                stage="train_best",
                epoch=self.best_epoch,
            )
            self.clearml.log_confusion_matrix(
                self.best_val_outputs["targets"],
                self.best_val_outputs["preds"],
                stage="val_best",
                epoch=self.best_epoch,
            )

            self.clearml.log_probability_distribution(
                self.best_train_outputs["targets"],
                self.best_train_outputs["probs"],
                stage="train_best",
                epoch=self.best_epoch,
            )
            self.clearml.log_probability_distribution(
                self.best_val_outputs["targets"],
                self.best_val_outputs["probs"],
                stage="val_best",
                epoch=self.best_epoch,
            )

            self.clearml.log_pr_curve(
                self.best_train_outputs["targets"],
                self.best_train_outputs["probs"],
                stage="train_best",
                epoch=self.best_epoch,
            )
            self.clearml.log_pr_curve(
                self.best_val_outputs["targets"],
                self.best_val_outputs["probs"],
                stage="val_best",
                epoch=self.best_epoch,
            )

            self.clearml.log_roc_curve(
                self.best_train_outputs["targets"],
                self.best_train_outputs["probs"],
                stage="train_best",
                epoch=self.best_epoch,
            )
            self.clearml.log_roc_curve(
                self.best_val_outputs["targets"],
                self.best_val_outputs["probs"],
                stage="val_best",
                epoch=self.best_epoch,
            )

            try:
                self.clearml.log_embedding_space(
                    model=self.model,
                    val_loader=val_loader,
                    device=self.device,
                    best_threshold=self.best_threshold,
                    epoch=self.best_epoch,
                )
            except Exception as exc:
                self.clearml.report_text(f"[WARN] embedding plot logging failed: {exc}")

        self.task.upload_artifact("best_model_path", artifact_object=self.best_path)

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_model_path": self.best_path,
        }
