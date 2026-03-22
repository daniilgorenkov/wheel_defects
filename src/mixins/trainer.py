import os
import random
from config import TrainerConfig
from typing import Mapping, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from clearml import Task
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    def compute_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def _log_metrics_to_clearml(self, metrics: dict, stage: str, epoch: int) -> None:
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title=metric_name,
                series=stage,
                value=float(value),
                iteration=epoch,
            )

    def _log_confusion_matrix_to_clearml(self, y_true, y_pred, stage: str, epoch: int) -> None:
        if not TrainerConfig.log_confusion_matrix:
            return

        cm = confusion_matrix(y_true, y_pred)
        self.logger.report_confusion_matrix(
            title=f"{stage}_confusion_matrix",
            series="cm",
            iteration=epoch,
            matrix=cm,
            xaxis="predicted",
            yaxis="actual",
        )

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

    def train_one_epoch(
        self,
        epoch: int,
        train_loader,
    ) -> dict:
        self.model.train()
        if self.criterion is None or self.optimizer is None:
            raise RuntimeError("Trainer is not initialized. Call fit() before training.")

        total_loss = 0.0
        all_preds = []
        all_targets = []

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

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(target.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.compute_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(train_loader.dataset)

        self._log_metrics_to_clearml(metrics, stage="train", epoch=epoch)
        self._log_confusion_matrix_to_clearml(all_targets, all_preds, stage="train", epoch=epoch)

        return metrics

    @torch.no_grad()
    def eval_one_epoch(
        self,
        epoch: int,
        val_loader,
    ) -> dict:
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

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

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(target.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.compute_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(val_loader.dataset)

        self._log_metrics_to_clearml(metrics, stage="val", epoch=epoch)
        self._log_confusion_matrix_to_clearml(all_targets, all_preds, stage="val", epoch=epoch)

        return metrics

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

        if patience is not None and patience < 1:
            raise ValueError("patience must be >= 1 or None")

        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            if lr_by_epoch and epoch in lr_by_epoch:
                self._set_optimizer_lr(lr_by_epoch[epoch])

            train_metrics = self.train_one_epoch(epoch, train_loader)
            val_metrics = self.eval_one_epoch(epoch, val_loader)

            current_score = val_metrics[TrainerConfig.monitor_metric]

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_f1={train_metrics['f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_recall={val_metrics['recall']:.4f}"
            )

            if self._is_better(current_score):
                self.best_score = current_score
                self.best_epoch = epoch
                self._save_checkpoint(epoch, current_score)
                epochs_without_improvement = 0

                self.logger.report_text(f"[BEST] epoch={epoch}, {TrainerConfig.monitor_metric}={current_score:.6f}")
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                message = f"[EARLY_STOP] epoch={epoch}, " f"no improvement in {epochs_without_improvement} epoch(s)"
                print(message)
                self.logger.report_text(message)
                break

        print(f"Best epoch: {self.best_epoch} | " f"best_{TrainerConfig.monitor_metric}={self.best_score:.4f}")

        self.task.upload_artifact("best_model_path", artifact_object=self.best_path)

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_model_path": self.best_path,
        }
