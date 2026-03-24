import torch
from torch import nn


class SpeedHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)  # BatchNorm1d(out_channels)
        self.act = nn.GELU()  # ReLU(), SiLU(), GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)  # для получения фиксированного размера фичей

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)  # (batch_size, out_channels)
        return x


class DeepFeaturesConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm3 = nn.GroupNorm(num_groups, out_channels)
        self.act3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)

        self.skip_proj = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        skip_connection = self.skip_proj(x)  # сохраняем вход для пропуска

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + skip_connection  # добавляем пропуск
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.pool(x).squeeze(-1)  # (batch_size, out_channels)

        return x


class LiteBaseline(nn.Module):
    def __init__(
        self,
        out_channels_signal: int,
        kernel_size_signal: int,
        num_groups_signal: int = None,
        out_channels_speed: int = 2,
        kernel_size_speed: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # фичи  из ускорений и обработка сигнала скорости буду идти отдельно и после препроцесса обхединячтся
        self.signal_head = DeepFeaturesConvBlock(
            1,
            out_channels_signal,
            kernel_size_signal,
            num_groups=num_groups_signal,
            dropout=dropout,
        )
        self.speed_head = SpeedHead(1, out_channels_speed, kernel_size_speed, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels_signal + out_channels_speed, out_channels_signal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_signal, 2),
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
