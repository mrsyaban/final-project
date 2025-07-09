import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """
    Generator architecture for PS-GAN, following the U-Net architecture in the paper:
    - Encoder: C16-C32-C64-C128
    - Decoder: C64-C32-C16-C3
    Where each Ck is a Convolution-LayerNorm-LeakyReLU layer with k filters
    and (4x4) spatial filters with stride 2.
    """
    def __init__(self, input_channels=3, output_channels=3, input_size=56, dropout_prob=0.3):
        super(Generator, self).__init__()

        # Parameters
        self.leaky_slope = 0.2
        self.dropout_prob = dropout_prob

        # Calculate output sizes for each layer (for dynamic LayerNorm)
        def calc_output_size(size, kernel_size=4, stride=2, padding=1):
            return int(np.floor((size + 2 * padding - kernel_size) / stride + 1))

        # Calculate sizes for each layer
        size1 = calc_output_size(input_size)  # 56 -> 28
        size2 = calc_output_size(size1)      # 28 -> 14
        size3 = calc_output_size(size2)      # 14 -> 7
        size4 = calc_output_size(size3)      # 7 -> 3 or 4 (depending on exact calculations)

        print(f"Calculated layer sizes: {input_size} -> {size1} -> {size2} -> {size3} -> {size4}")

        # Encoder (downsampling)
        # C16
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            # nn.LayerNorm([16, size1, size1]),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C32
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            # nn.LayerNorm([32, size2, size2]),
            # nn.InstanceNorm2d(32),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C64
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.LayerNorm([64, size3, size3]),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C128
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.LayerNorm([128, size4, size4]),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(self.leaky_slope)
        )

        # Decoder (upsampling) with skip connections
        # Dropout is applied after LeakyReLU in decoders
        # C64
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.LayerNorm([64, size3-1, size3-1]),
            # nn.InstanceNorm2d(64),
            nn.Dropout2d(self.dropout_prob),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C32
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            # nn.LayerNorm([32, size2, size2]),
            # nn.InstanceNorm2d(32),
            nn.Dropout2d(self.dropout_prob),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C16
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            # nn.LayerNorm([16, size1, size1]),
            # nn.InstanceNorm2d(16),
            nn.Dropout2d(self.dropout_prob),
            nn.LeakyReLU(self.leaky_slope)
        )

        # C3
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

        # Dropout2d layers (to be used in decoder)
        self.drop1 = nn.Dropout2d(self.dropout_prob)
        self.drop2 = nn.Dropout2d(self.dropout_prob)
        self.drop3 = nn.Dropout2d(self.dropout_prob)

    def forward(self, x):
        """
        Forward pass with U-Net skip connections and dropout in decoder path.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections and dropout
        d1 = self.dec1(e4)
        if d1.size() != e3.size():
            d1 = F.interpolate(d1, size=e3.size()[2:], mode='bilinear', align_corners=False)
        d1_cat = torch.cat([d1, e3], dim=1)

        d2 = self.dec2(d1_cat)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=False)
        d2_cat = torch.cat([d2, e2], dim=1)

        d3 = self.dec3(d2_cat)
        if d3.size() != e1.size():
            d3 = F.interpolate(d3, size=e1.size()[2:], mode='bilinear', align_corners=False)
        d3_cat = torch.cat([d3, e1], dim=1)

        d4 = self.dec4(d3_cat)

        return d4

    # def forward(self, x):
    #     """
    #     Forward pass with U-Net skip connections and focus on black pixel modification.
    #     The generator explicitly knows which pixels to modify through mask input.
    #     """
    #     # Create a mask where black pixels are 1 and white pixels are 0
    #     if x.shape[1] > 1:  # If more than one channel
    #         x_grayscale = x.mean(dim=1, keepdim=True)
    #     else:
    #         x_grayscale = x
        
    #     # Create binary mask based on pixel intensity
    #     threshold = 0.0 if x_grayscale.min() < 0 else 0.5
    #     mask = (x_grayscale < threshold).float()  # 1 for black pixels, 0 for white pixels
        
    #     # Store the original input
    #     original = x
        
    #     # Concatenate the mask to the input as an additional channel
    #     # This explicitly tells the generator which pixels it should focus on
    #     x_with_mask = torch.cat([x, mask.expand_as(x_grayscale)], dim=1)
        
    #     # Modify the first encoder to accept the additional channel
    #     # Using a temporary convolution layer that matches the first encoder but with +1 input channels
    #     temp_conv = nn.Conv2d(x.shape[1] + 1, 16, kernel_size=4, stride=2, padding=1).to(x.device)
    #     if hasattr(self, 'temp_weights_initialized') and self.temp_weights_initialized:
    #         pass  # Keep using the same temp_conv if already initialized
    #     else:
    #         # Initialize the weights to match the original first layer (for the original channels)
    #         with torch.no_grad():
    #             # Copy weights for original channels
    #             original_weights = self.enc1[0].weight
    #             temp_conv.weight[:, :x.shape[1], :, :] = original_weights
    #             # Initialize the weights for the mask channel to small random values
    #             temp_conv.weight[:, x.shape[1]:, :, :] = torch.randn_like(temp_conv.weight[:, 0:1, :, :]) * 0.01
    #             temp_conv.bias = nn.Parameter(self.enc1[0].bias.clone())
    #         self.temp_weights_initialized = True
        
    #     # Run first convolution with the extra mask channel
    #     e1_features = temp_conv(x_with_mask)
    #     # Continue with the rest of the encoder layer operations
    #     for layer in list(self.enc1)[1:]:  # Skip the first Conv2d layer which we replaced
    #         e1_features = layer(e1_features)
    #     e1 = e1_features
        
    #     # Continue with the normal encoder path
    #     e2 = self.enc2(e1)
    #     e3 = self.enc3(e2)
    #     e4 = self.enc4(e3)

    #     # Decoder with skip connections and dropout (no changes needed here)
    #     d1 = self.dec1(e4)
    #     if d1.size() != e3.size():
    #         d1 = F.interpolate(d1, size=e3.size()[2:], mode='bilinear', align_corners=False)
    #     d1_cat = torch.cat([d1, e3], dim=1)

    #     d2 = self.dec2(d1_cat)
    #     if d2.size() != e2.size():
    #         d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=False)
    #     d2_cat = torch.cat([d2, e2], dim=1)

    #     d3 = self.dec3(d2_cat)
    #     if d3.size() != e1.size():
    #         d3 = F.interpolate(d3, size=e1.size()[2:], mode='bilinear', align_corners=False)
    #     d3_cat = torch.cat([d3, e1], dim=1)

    #     d4 = self.dec4(d3_cat)
        
    #     # Still apply the mask at the end for absolute guarantee
    #     output = mask * d4 + (1 - mask) * original
        
    #     return output