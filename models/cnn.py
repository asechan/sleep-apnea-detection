# models/cnn.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for spectrogram classification.
    Input: (B, 1, n_mels, T)
    Output: logits (B,)
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 1, base_filters: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x).squeeze(dim=-1)  # (B,)
        return logits