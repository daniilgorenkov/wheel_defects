import torch
from torch import nn
from src.models.lite_baseline import SpeedHead


class ShortFeaturesConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dropout: float = 0.2,
        kernel_size: int = 3,
        pooling: bool = True,
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

        self.pooling = pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
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

        if self.pooling:
            x = self.pool(x).squeeze(-1)  # (batch_size, out_channels)
            return x  # [B,C]
        else:
            return x  # [B,C,T]


class LongFeaturesConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dropout: float = 0.2,
        kernel_size: int = 7,
        pooling: bool = True,
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

        self.pooling = pooling
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

        if self.pooling:
            x = self.pool(x).squeeze(-1)  # (batch_size, out_channels)
            return x  # [B,C]
        else:
            return x  # [B,T,C]


class ThreeHeadModel(nn.Module):
    def __init__(
        self,
        out_channels_signal: int,
        num_groups_signal: int = None,
        out_channels_speed: int = 2,
        kernel_size_speed: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.short_block = ShortFeaturesConvBlock(1, out_channels_signal, num_groups_signal, dropout)
        self.long_block = LongFeaturesConvBlock(1, out_channels_signal, num_groups_signal, dropout)

        self.speed_head = SpeedHead(1, out_channels_speed, kernel_size_speed, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels_signal * 2 + out_channels_speed, out_channels_signal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_signal, 2),
        )

    def extract_features(self, signal: torch.Tensor, speed: torch.Tensor):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        if speed.ndim == 1:
            speed = speed.unsqueeze(1).unsqueeze(1)  # [B] -> [B, 1, 1]
        elif speed.ndim == 2:
            speed = speed.unsqueeze(1)  # [B, L] -> [B, 1, L]

        short_features = self.short_block(signal)  # [B,C]
        long_features = self.long_block(signal)  # [B,C]

        speed_features = self.speed_head(speed)  # [B,speed_out]

        return torch.cat([short_features, long_features, speed_features], dim=1)  # [B,2C+speed_out]

    def forward(self, signal: torch.Tensor, speed: torch.Tensor):

        combined_features = self.extract_features(signal, speed)  # [B,128+2]

        output = self.classifier(combined_features)  # [B,num_classes]

        return output
