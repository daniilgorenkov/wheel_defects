from __future__ import annotations

from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class TargetSpaceVisualizer:
    """Visualize target distribution in 2D space using PCA or t-SNE."""

    def __init__(
        self,
        method: str = "tsne",
        random_state: int = 42,
        include_speed: bool = False,
        resample_length: int | None = None,
        standardize: bool = True,
    ):
        if method not in {"tsne", "pca"}:
            raise ValueError("method must be either 'tsne' or 'pca'")

        self.method = method
        self.random_state = random_state
        self.include_speed = include_speed
        self.resample_length = resample_length
        self.standardize = standardize

    def fit_transform(
        self,
        samples: list[dict],
        tsne_perplexity: float = 30.0,
        tsne_max_iter: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (embedding_2d, targets)."""
        features, targets = self._build_matrix(samples)

        if self.standardize:
            features = StandardScaler().fit_transform(features)

        if self.method == "pca":
            reducer = PCA(n_components=2, random_state=self.random_state)
            embedding = reducer.fit_transform(features)
            return embedding, targets

        safe_perplexity = self._safe_perplexity(tsne_perplexity, len(features))
        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
            max_iter=tsne_max_iter,
        )
        embedding = reducer.fit_transform(features)
        return embedding, targets

    def plot(
        self,
        samples: list[dict],
        title: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        alpha: float = 0.8,
        s: int = 45,
        tsne_perplexity: float = 30.0,
        tsne_max_iter: int = 1000,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Build embedding and render a scatter plot colored by target."""
        embedding, targets = self.fit_transform(
            samples=samples,
            tsne_perplexity=tsne_perplexity,
            tsne_max_iter=tsne_max_iter,
        )

        fig, ax = plt.subplots(figsize=figsize)
        unique_targets = np.unique(targets)
        cmap = plt.cm.get_cmap("tab10", len(unique_targets))

        for idx, target in enumerate(unique_targets):
            mask = targets == target
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                alpha=alpha,
                s=s,
                color=cmap(idx),
                label=f"target={target}",
            )

        method_name = self.method.upper() if self.method == "pca" else "t-SNE"
        ax.set_title(title or f"Target distribution in {method_name} space")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Classes")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def plot_model_embeddings(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str | torch.device | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        alpha: float = 0.8,
        s: int = 45,
        decision_threshold: float = 0.5,
        show_decision_boundary: bool = True,
        tsne_perplexity: float = 30.0,
        tsne_max_iter: int = 1000,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Build 2D projection from learned encoder embeddings."""
        embeddings, targets, probs, preds = self._extract_model_embedding_outputs(
            model=model,
            dataloader=dataloader,
            device=device,
            decision_threshold=decision_threshold,
        )

        if self.standardize:
            embeddings = StandardScaler().fit_transform(embeddings)

        if self.method == "pca":
            reducer = PCA(n_components=2, random_state=self.random_state)
            projection = reducer.fit_transform(embeddings)
        else:
            safe_perplexity = self._safe_perplexity(tsne_perplexity, len(embeddings))
            reducer = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                random_state=self.random_state,
                init="pca",
                learning_rate="auto",
                max_iter=tsne_max_iter,
            )
            projection = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=figsize)

        if show_decision_boundary and len(np.unique(preds)) > 1:
            self._draw_decision_boundary(ax=ax, points_2d=projection, predicted_labels=preds)

        unique_targets = np.unique(targets)
        cmap = plt.cm.get_cmap("tab10", len(unique_targets))

        for idx, target in enumerate(unique_targets):
            mask = targets == target
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                alpha=alpha,
                s=s,
                color=cmap(idx),
                label=f"target={target}",
            )

        errors_mask = preds != targets
        if np.any(errors_mask):
            ax.scatter(
                projection[errors_mask, 0],
                projection[errors_mask, 1],
                s=s * 1.8,
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
                label="misclassified",
            )

        method_name = self.method.upper() if self.method == "pca" else "t-SNE"
        error_rate = float(np.mean(errors_mask))
        ax.set_title(title or f"Learned embeddings in {method_name} space | error={error_rate:.2%}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Classes")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def _build_matrix(self, samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        if not samples:
            raise ValueError("samples is empty")

        self._validate_sample_keys(samples)

        signal_lengths = [len(np.asarray(sample["X"], dtype=float).ravel()) for sample in samples]
        target_length = self.resample_length or int(np.median(signal_lengths))
        if target_length <= 1:
            raise ValueError("Signals are too short for projection")

        features = []
        targets = []

        for sample in samples:
            signal = np.asarray(sample["X"], dtype=float).ravel()
            signal = self._resample_1d(signal, target_length)

            if self.include_speed:
                speed = np.asarray(sample["speed"], dtype=float).ravel()
                speed_feature = np.array([float(np.mean(speed))], dtype=float)
                feature_vector = np.concatenate([signal, speed_feature])
            else:
                feature_vector = signal

            features.append(feature_vector)
            targets.append(sample["target"])

        features_arr = np.vstack(features)
        targets_arr = np.asarray(targets)
        return features_arr, targets_arr

    @staticmethod
    def _safe_perplexity(requested_perplexity: float, n_samples: int) -> float:
        max_allowed = max(2.0, float(n_samples - 1))
        return float(min(requested_perplexity, max_allowed))

    @staticmethod
    def _resample_1d(signal: np.ndarray, target_length: int) -> np.ndarray:
        if len(signal) == target_length:
            return signal

        if len(signal) < 2:
            return np.full(shape=(target_length,), fill_value=float(signal[0]) if len(signal) else 0.0)

        x_old = np.linspace(0.0, 1.0, num=len(signal))
        x_new = np.linspace(0.0, 1.0, num=target_length)
        return np.interp(x_new, x_old, signal)

    @staticmethod
    @torch.no_grad()
    def _extract_model_embeddings(
        model: torch.nn.Module,
        dataloader,
        device: str | torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings, targets, _, _ = TargetSpaceVisualizer._extract_model_embedding_outputs(
            model=model,
            dataloader=dataloader,
            device=device,
        )
        return embeddings, targets

    @staticmethod
    def _prepare_speed_for_baseline(speed: torch.Tensor) -> torch.Tensor:
        if speed.ndim == 1:
            return speed.unsqueeze(1)
        return speed

    @staticmethod
    def _prepare_speed_for_lite(speed: torch.Tensor) -> torch.Tensor:
        if speed.ndim == 1:
            return speed.unsqueeze(1).unsqueeze(1)
        if speed.ndim == 2:
            return speed.unsqueeze(1)
        return speed

    @staticmethod
    @torch.no_grad()
    def _extract_model_embedding_outputs(
        model: torch.nn.Module,
        dataloader,
        device: str | torch.device | None = None,
        decision_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)

        if not (0.0 <= decision_threshold <= 1.0):
            raise ValueError("decision_threshold must be in [0, 1]")

        was_training = model.training
        model.eval()

        all_embeddings = []
        all_targets = []
        all_probs = []
        all_preds = []

        for batch in dataloader:
            signal = batch["signal"].to(device)
            target = batch["target"].to(device)
            speed = batch.get("speed")
            speed = speed.to(device) if speed is not None else None

            logits = None
            if speed is not None:
                try:
                    logits = model(signal, speed)
                except TypeError:
                    logits = model(signal)
            else:
                logits = model(signal)

            if logits.ndim == 2 and logits.size(1) > 1:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits.view(-1))
            preds = (probs >= decision_threshold).long()

            if hasattr(model, "signal_head") and hasattr(model, "speed_head"):
                signal_lite = signal.unsqueeze(1) if signal.ndim == 2 else signal
                emb_signal = model.signal_head(signal_lite)
                if speed is None:
                    raise ValueError("Dataloader must provide speed for this model")
                speed_lite = TargetSpaceVisualizer._prepare_speed_for_lite(speed.float())
                emb_speed = model.speed_head(speed_lite)
                emb = torch.cat([emb_signal, emb_speed], dim=1)
            elif hasattr(model, "short_block") and hasattr(model, "long_block") and hasattr(model, "speed_head"):
                emb = model.extract_features(signal, speed)
            else:
                raise ValueError(
                    "Unsupported model architecture for embedding extraction. "
                    "Expected either encoder/speed_branch or signal_head/speed_head."
                )

            all_embeddings.append(emb.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

        if was_training:
            model.train()

        if not all_embeddings:
            raise ValueError("Dataloader is empty")

        embeddings = np.concatenate(all_embeddings, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        return embeddings, targets, probs, preds

    @staticmethod
    def _draw_decision_boundary(ax: plt.Axes, points_2d: np.ndarray, predicted_labels: np.ndarray) -> None:
        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

        x_pad = (x_max - x_min) * 0.08 + 1e-6
        y_pad = (y_max - y_min) * 0.08 + 1e-6
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 220),
            np.linspace(y_min, y_max, 220),
        )
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

        surrogate = LogisticRegression(random_state=42, max_iter=1500)
        surrogate.fit(points_2d, predicted_labels)

        z_prob = surrogate.predict_proba(grid_points)[:, 1].reshape(grid_x.shape)
        ax.contourf(grid_x, grid_y, z_prob, levels=np.linspace(0.0, 1.0, 15), cmap="RdBu", alpha=0.16)
        ax.contour(grid_x, grid_y, z_prob, levels=[0.5], colors="black", linewidths=1.0, linestyles="--")

    @staticmethod
    def _validate_sample_keys(samples: Iterable[dict]) -> None:
        required = {"X", "speed", "target"}
        missing_counter = Counter()

        for sample in samples:
            missing = required - set(sample.keys())
            for key in missing:
                missing_counter[key] += 1

        if missing_counter:
            details = ", ".join(f"{key}: {count}" for key, count in missing_counter.items())
            raise ValueError(f"Samples are missing required keys -> {details}")


class ModelGraphSaver:
    """Build and send model embedding plots to ClearML."""

    def __init__(
        self,
        logger,
        close_after_save: bool = True,
    ):
        self.logger = logger
        self.close_after_save = close_after_save

    def save_model_plot(
        self,
        visualizer: TargetSpaceVisualizer,
        model: torch.nn.Module,
        dataloader,
        device: str | torch.device | None = None,
        title: str = "model_embeddings",
        series: str = "embedding_space",
        iteration: int = 0,
        **plot_kwargs,
    ) -> None:
        """Render model embeddings and report the figure to ClearML."""
        fig, _ = visualizer.plot_model_embeddings(
            model=model,
            dataloader=dataloader,
            device=device,
            **plot_kwargs,
        )
        self.save_figure(
            fig=fig,
            title=title,
            series=series,
            iteration=iteration,
        )

    def save_figure(self, fig: plt.Figure, title: str, series: str, iteration: int) -> None:
        """Report a prepared matplotlib figure to ClearML."""
        if not title:
            raise ValueError("title must not be empty")
        if not series:
            raise ValueError("series must not be empty")

        self.logger.report_matplotlib_figure(
            title=title,
            series=series,
            iteration=iteration,
            figure=fig,
        )
        if self.close_after_save:
            plt.close(fig)
