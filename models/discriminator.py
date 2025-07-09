import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator architecture for PS-GAN:
    C64-C128-C256-C512
    Followed by a linear layer to generate a 1-dimensional output and a Sigmoid function.
    """
    def __init__(self, input_channels=3, input_size=56):
        """
        Initialize the discriminator

        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            input_size: Size of the input image/patch (default: 56 for 56x56 patches)
        """
        super(Discriminator, self).__init__()

        # Calculate the size after convolutions to determine the flattened feature size
        # Each C layer uses stride 2 and padding 1, so the spatial size halves
        feature_size = input_size // (2**4)  # After 4 downsampling layers
        flattened_features = 512 * feature_size * feature_size

        # Sequential model for the convolutional layers
        # C64-C128-C256-C512
        self.conv_layers = nn.Sequential(
            # C64: Input -> Conv -> LeakyReLU
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # C128: C64 -> Conv -> BatchNorm -> LeakyReLU
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # C256: C128 -> Conv -> BatchNorm -> LeakyReLU
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # C512: C256 -> Conv -> BatchNorm -> LeakyReLU
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final linear layer and sigmoid 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input image or patch

        Returns:
            1-dimensional output after sigmoid (probability of being real)
        """
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output