from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from config import TrainerConfig
from src.mixins.plot_utils import ModelGraphSaver, TargetSpaceVisualizer


class ClearMLLogger:
    """Encapsulates all ClearML logging for the Trainer."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def report_text(self, text: str) -> None:
        self.logger.report_text(text)

    # ------------------------------------------------------------------
    # Scalar metrics
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict, stage: str, epoch: int) -> None:
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title=metric_name,
                series=stage,
                value=float(value),
                iteration=epoch,
            )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def log_confusion_matrix(self, y_true, y_pred, stage: str, epoch: int) -> None:
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

    # ------------------------------------------------------------------
    # Distribution / curve plots
    # ------------------------------------------------------------------

    def log_probability_distribution(self, y_true, y_prob, stage: str, epoch: int) -> None:
        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(y_prob_arr[y_true_arr == 0], bins=20, alpha=0.6, label="class 0")
        ax.hist(y_prob_arr[y_true_arr == 1], bins=20, alpha=0.6, label="class 1")
        ax.set_title(f"{stage} probability distribution")
        ax.set_xlabel("P(class=1)")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.2)

        self.logger.report_matplotlib_figure(
            title=f"{stage}_probability_distribution",
            series="probability_distribution",
            iteration=epoch,
            figure=fig,
        )
        plt.close(fig)

    def log_pr_curve(self, y_true, y_prob, stage: str, epoch: int) -> None:
        if len(set(y_true)) < 2:
            return

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, label=f"AP={ap:.4f}")
        ax.set_title(f"{stage} PR curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.2)

        self.logger.report_matplotlib_figure(
            title=f"{stage}_pr_curve",
            series="pr_curve",
            iteration=epoch,
            figure=fig,
        )
        plt.close(fig)

    def log_roc_curve(self, y_true, y_prob, stage: str, epoch: int) -> None:
        if len(set(y_true)) < 2:
            return

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(f"{stage} ROC curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)

        self.logger.report_matplotlib_figure(
            title=f"{stage}_roc_curve",
            series="roc_curve",
            iteration=epoch,
            figure=fig,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Embedding space
    # ------------------------------------------------------------------

    def log_embedding_space(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        best_threshold: float,
        epoch: int,
    ) -> None:
        method = getattr(TrainerConfig, "embedding_plot_method", "tsne")
        random_state = int(getattr(TrainerConfig, "embedding_plot_random_state", TrainerConfig.seed))
        perplexity = float(getattr(TrainerConfig, "embedding_plot_tsne_perplexity", 30.0))
        max_iter = int(getattr(TrainerConfig, "embedding_plot_tsne_max_iter", 1000))

        visualizer = TargetSpaceVisualizer(
            method=method,
            random_state=random_state,
            standardize=True,
        )
        saver = ModelGraphSaver(logger=self.logger, close_after_save=True)
        saver.save_model_plot(
            visualizer=visualizer,
            model=model,
            dataloader=val_loader,
            device=device,
            title="val_best_embedding_space",
            series=f"embedding_{method}",
            iteration=epoch,
            show_decision_boundary=True,
            decision_threshold=best_threshold,
            tsne_perplexity=perplexity,
            tsne_max_iter=max_iter,
        )

    # ------------------------------------------------------------------
    # Model architecture summary
    # ------------------------------------------------------------------

    def log_model_architecture(
        self,
        model: torch.nn.Module,
        device: torch.device,
        train_loader: DataLoader,
    ) -> None:
        """Log a compact, readable model architecture summary to ClearML."""
        self.logger.report_text(f"[MODEL]\n{model}")

        if train_loader is None:
            self.logger.report_text("[WARN] train_loader is None, architecture graph logging skipped")
            return

        try:
            batch = next(iter(train_loader))
        except StopIteration:
            self.logger.report_text("[WARN] train_loader is empty, architecture graph logging skipped")
            return
        except Exception as exc:
            self.logger.report_text(f"[WARN] cannot read sample batch for architecture graph: {exc}")
            return

        signal = batch["signal"].to(device)
        speed = batch.get("speed")
        if speed is not None:
            speed = speed.to(device)
        if signal.size(0) > 1:
            signal = signal[:1]
        if speed is not None and speed.size(0) > 1:
            speed = speed[:1]

        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            summary_lines = [
                f"Model: {model.__class__.__name__}",
                f"Input signal shape: {tuple(signal.shape)}",
                f"Input speed shape: {tuple(speed.shape) if speed is not None else 'N/A'}",
                f"Total params: {total_params:,}",
                f"Trainable params: {trainable_params:,}",
                "",
                "Top-level blocks:",
            ]

            children = list(model.named_children())
            if not children:
                summary_lines.append("- (no child modules)")
            else:
                for name, module in children:
                    block_params = sum(p.numel() for p in module.parameters())
                    summary_lines.append(f"- {name}: {module.__class__.__name__} ({block_params:,} params)")

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.axis("off")
            ax.text(
                0.01,
                0.99,
                "\n".join(summary_lines),
                va="top",
                ha="left",
                family="monospace",
                fontsize=10,
                wrap=True,
            )
            ax.set_title("Model architecture summary")

            self.logger.report_matplotlib_figure(
                title="model_architecture_graph",
                series="architecture",
                iteration=0,
                figure=fig,
            )
            plt.close(fig)
        except Exception as exc:
            self.logger.report_text(f"[WARN] model architecture graph logging failed: {exc}")
