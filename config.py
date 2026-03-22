import torch
import numpy as np


class TrainerConfig:
    project_name: str = "wheel-defects"
    task_name: str = "baseline-cnn"
    seed: int = 101

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 20

    use_speed: bool = True
    monitor_metric: str = "f1"
    monitor_mode: str = "max"  # "max" or "min"

    save_dir: str = "./checkpoints"
    save_best_name: str = "best_model.pt"

    log_confusion_matrix: bool = True

    device = "cuda" if torch.cuda.is_available() else "cpu"


class ProcessorConfig:
    train_size = 0.8
    batch_size = 32


class ModelConfig:
    n_classes = 2
    embedding_dim = 64
    base_channels = 16
    use_speed = True
    dropout = 0.2


class PreprocessorConfig:
    USE_COLS: list = ["datatime", "accz", "spd"]
    WHEEL_DIAMETER = 0.960  # м
    CIRCUMFERENCE = np.pi * WHEEL_DIAMETER
    GAP_RESET_SEC = 0.5
    MIN_SPEED_KMH = 0.1
    MIN_POINTS_PER_REV = 8
    NORMALIZE_SIGNAL = False
