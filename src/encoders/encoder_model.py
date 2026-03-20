import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()

        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.drop(x)
        x = x + residual
        x = self.pool(x)
        return x


class SignalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, dropout=dropout),
            ConvBlock(base_channels, base_channels * 2, kernel_size=5, dropout=dropout),
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        x = self.backbone(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class WheelBaseline(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        embedding_dim: int = 128,
        base_channels: int = 32,
        use_speed: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.use_speed = use_speed
        self.encoder = SignalEncoder(
            in_channels=1,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        if use_speed:
            self.speed_branch = nn.Sequential(
                nn.Linear(1, 16),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = embedding_dim + 16
        else:
            head_in = embedding_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, signal: torch.Tensor, speed: torch.Tensor | None = None) -> torch.Tensor:
        signal_emb = self.encoder(signal)

        if self.use_speed:
            if speed is None:
                raise ValueError("speed is required when use_speed=True")
            if speed.ndim == 1:
                speed = speed.unsqueeze(1)
            speed_emb = self.speed_branch(speed.float())
            features = torch.cat([signal_emb, speed_emb], dim=1)
        else:
            features = signal_emb

        logits = self.head(features)
        return logits
