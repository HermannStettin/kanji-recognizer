import torch
from torch import nn
import numpy as np
import torch.utils.data

class M7_1(nn.Module):
    def __init__(self, in_channels, num_classes=160):
        super(M7_1, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=3, stride=1, padding=1), # (64, 64, 64)
            nn.BatchNorm3d(64, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool3d(kernel_size=2, stride=2), # (32, 32, 64)

            nn.Conv3d(64, 128, kernel_size=3, padding=1), # (32, 32, 128)
            nn.BatchNorm3d(128, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool3d(kernel_size=2, stride=2), # (16, 16, 128)

            nn.Conv3d(128, 512, kernel_size=3, padding=1), # (16, 16, 512)
            nn.BatchNorm3d(512, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv3d(512, 512, kernel_size=3, padding=1), # (16, 16, 512)
            nn.BatchNorm3d(512, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool3d(kernel_size=2, stride=2), # (8, 8, 512)
        )

        self.linear_layers = nn.Sequential(
            # nn.BatchNorm1d(8*8*2*512),
            nn.Dropout(0.5),
            nn.Linear(in_features=8*8*3*512, out_features=4096),
            nn.ReLU(),

            # nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),

            # nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
