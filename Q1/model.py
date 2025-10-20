import torch
import torch.nn as nn

# ---------------------------------------------------------
# Model definition file
# Defines a simple Convolutional Neural Network (CNN)
# for image classification on the MNIST dataset.
# ---------------------------------------------------------


class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        """
        CNN architecture for MNIST classification.
        :param in_channels: number of input channels (1 for grayscale)
        :param num_classes: number of output classes (10 digits)
        """

        super().__init__()


        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
