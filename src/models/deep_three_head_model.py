import torch
from torch import nn
from src.models.three_head_model import LongFeaturesConvBlock, ShortFeaturesConvBlock
from src.models.lite_baseline import SpeedHead


class MaxPoolConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class DeepThreeHeadModel(nn.Module):

    def __init__(
        self,
        out_channels_signal: int,
        num_groups_signal: int = None,
        out_channels_speed: int = 2,
        kernel_size_speed: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.short_block = ShortFeaturesConvBlock(
            1,
            out_channels_signal,
            num_groups_signal,
            dropout,
            pooling=False,
        )  # [B,C,O]

        self.long_block = LongFeaturesConvBlock(
            1,
            out_channels_signal,
            num_groups_signal,
            dropout,
            pooling=False,
        )  # [B,C,O]

        self.speed_head = SpeedHead(1, out_channels_speed, kernel_size_speed, dropout=dropout)  # [B,O]

        self.maxpool_conv = MaxPoolConvBlock(
            out_channels_signal * 2, out_channels_signal * 2, kernel_size=3, stride=1
        )  # [B,2C,O]

        self.classifier = nn.Sequential(
            nn.Linear(out_channels_signal * 2 + out_channels_speed, out_channels_signal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_signal, 2),
        )

        self.signal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, signal: torch.Tensor, speed: torch.Tensor):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        if speed.ndim == 1:
            speed = speed.unsqueeze(1).unsqueeze(1)  # [B] -> [B, 1, 1]
        elif speed.ndim == 2:
            speed = speed.unsqueeze(1)  # [B, L] -> [B, 1, L]

        short_features = self.short_block(signal)  # [B,C,O]
        long_features = self.long_block(signal)  # [B,C,O]

        speed_features = self.speed_head(speed)  # [B,O]

        combined_features = torch.cat([short_features, long_features], dim=1)  # [B,2C,O]
        combined_features = self.maxpool_conv(combined_features)  # [B,2C,O]

        combined_features = self.signal_pool(combined_features).squeeze(-1)  # [B,2C*O]
        features = torch.cat([combined_features, speed_features], dim=1)  # [B,2C+speed]

        output = self.classifier(features)  # [B,2]

        return output
