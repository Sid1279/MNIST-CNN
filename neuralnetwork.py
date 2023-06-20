import torch.nn as nn


class CIFARClassifier(nn.Module):
    def __init__(self):
        super(CIFARClassifier, self).__init__()  # Invoke parent class initializer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()  # ReLU activation
        self.pool = nn.MaxPool2d(
            kernel_size=2
        )  # downsizes into 2x2 while preserving important features
        self.flatten = nn.Flatten()  # flattens the 2D input tensor into a 1D tensor
        self.fc1 = nn.Linear(
            16 * 16 * 16, 256
        )  # first layer: 16*16*16 inputs, 64 outputs
        self.fc2 = nn.Linear(256, 2)  # output layer: 64 inputs, 2 outputs

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
