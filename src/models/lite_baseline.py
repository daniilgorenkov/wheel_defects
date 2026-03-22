import torch
from torch import nn


class FeaturesConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        use_norm: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        if self.use_norm:
            self.bn = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()  # ReLU(), SiLU(), GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)  # для получения фиксированного размера фичей

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)  # (batch_size, out_channels)
        return x


class LiteBaseline(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        use_norm: bool,
        num_groups: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        # фичи  из ускорений и обработка сигнала скорости буду идти отдельно и после препроцесса обхединячтся
        self.signal_head = FeaturesConvBlock(
            1,
            out_channels,
            kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.speed_head = FeaturesConvBlock(
            1,
            out_channels,
            kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            use_norm=use_norm,
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )

    def forward(self, signal: torch.Tensor, speed: torch.Tensor):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        if speed.ndim == 1:
            speed = speed.unsqueeze(1).unsqueeze(1)  # [B] -> [B, 1, 1]
        elif speed.ndim == 2:
            speed = speed.unsqueeze(1)  # [B, L] -> [B, 1, L]

        signal_features = self.signal_head(signal)  # (B, out_channels)
        speed_features = self.speed_head(speed)  # (B, out_channels)

        combined = torch.cat([signal_features, speed_features], dim=1)
        output = self.classifier(combined)

        return output
