import torch
import torch.nn as nn

from nn.base_nn import ConvBlockV1


class BaselineCNN(nn.Module):
    """
    Baseline CNN for 3-class cough classification.

    Classes:
        0 = Healthy
        1 = Symptomatic
        2 = COVID-19
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # First convolution block
        self.conv1 = ConvBlockV1(
            in_channels=1,
            out_channels=16,
            kernel=3,
            stride=1,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(2)

        # Second convolution block
        self.conv2 = ConvBlockV1(
            in_channels=16,
            out_channels=32,
            kernel=3,
            stride=1,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(2)

        # Reduce each feature map to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3-class classifier
        self.classifier = nn.Linear(
            32,
            num_classes
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
