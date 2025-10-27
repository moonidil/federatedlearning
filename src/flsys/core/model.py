from torch import nn


class MnistNet(nn.Module):
    """CNN model for MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # 1 input channel (grayscale) n 32 output channels
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),  # 32 in, 64 out channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduces spatial dimensions by half
            nn.Dropout(0.25),  # Regularisation
            nn.Flatten(),  # Flatten for dense layers
            nn.Linear(9216, 128),  # 9216=64*12*12 (after conv and pooling)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        return self.net(x)
