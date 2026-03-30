import math
import torch
from torch import nn
from src.models.three_head_model import ShortFeaturesConvBlock, LongFeaturesConvBlock
from src.models.lite_baseline import SpeedHead


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNTransformerEncoder(nn.Module):

    def __init__(
        self,
        out_channels_signal: int,
        num_groups_signal: int = 4,
        out_channels_speed: int = 2,
        kernel_size_speed: int = 3,
        nheads: int = 4,
        enc_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        assert (out_channels_signal * 2) % nheads == 0, "d_model must be divisible by nheads"

        self.short_block = ShortFeaturesConvBlock(
            1, out_channels_signal, num_groups_signal, dropout, pooling=False
        )  # [B,T,C]
        self.long_block = LongFeaturesConvBlock(
            1, out_channels_signal, num_groups_signal, dropout, pooling=False
        )  # [B,T,C]

        self.pos_encoder = PositionalEncoding(out_channels_signal, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_channels_signal,
                nhead=nheads,
                dim_feedforward=out_channels_signal * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=enc_layers,
        )
        self.speed_head = SpeedHead(1, out_channels_speed, kernel_size_speed, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels_signal + out_channels_speed, out_channels_signal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels_signal, 2),
        )

    def forward(self, signal: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:

        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        if speed.ndim == 1:
            speed = speed.unsqueeze(1).unsqueeze(1)  # [B] -> [B, 1, 1]
        elif speed.ndim == 2:
            speed = speed.unsqueeze(1)

        # short_features = self.short_block(signal)  # [B,T,C]
        long_features = self.long_block(signal)  # [B,T,C]

        # signal_features = torch.cat([short_features, long_features], dim=2)  # [B,T,C*2]
        signal_features = self.pos_encoder(long_features)  # [B,T,C*2]
        signal_features = self.transformer_encoder(signal_features)  # [B,T,C*2]
        signal_features = signal_features.mean(dim=1)  # [B,C*2]

        speed_features = self.speed_head(speed)  # [B,2]

        combined_features = torch.cat([signal_features, speed_features], dim=1)  # [B,128+2]

        output = self.classifier(combined_features)  # [B,num_classes]

        return output
