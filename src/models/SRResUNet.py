# src/models/SRResUNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=32, num_residuals=2, upscale_factor=4):
        """
        SRResUNet for super-resolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_filters (int): Number of filters in the initial convolutional layer.
            num_residuals (int): Number of residual blocks in the encoder/decoder blocks.
            upscale_factor (int): Factor by which to upscale the image (e.g., 2 for 2x SR).
        """
        super(SRResUNet, self).__init__()

        self.in_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)

        # Encoder
        self.encoder1 = EncoderBlock(num_filters, num_filters * 2, num_residuals) # Downsample
        self.encoder2 = EncoderBlock(num_filters * 2, num_filters * 4, num_residuals) # Downsample
        self.encoder3 = EncoderBlock(num_filters * 4, num_filters * 8, num_residuals) # Downsample

        self.bridge_conv = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=3, padding=1)

        # Decoder
        self.decoder3 = DecoderBlock(num_filters * 8, num_filters * 4, num_residuals) # Upsample
        self.decoder2 = DecoderBlock(num_filters * 4, num_filters * 2, num_residuals) # Upsample
        self.decoder1 = DecoderBlock(num_filters * 2, num_filters, num_residuals) # Upsample

        # Output layer with upscaling
        # self.out_conv = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1) # Original - no upscaling

        # Upscaling with ConvTranspose2d:
        self.upscale_factor = upscale_factor
        self.out_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * (upscale_factor ** 2), kernel_size=3, padding=1), # Prepare for pixel shuffle
            nn.PixelShuffle(upscale_factor), # Upscale H and W
            nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1) # Final channel adjustment
        )

    def forward(self, x):
        # Encoder path
        #print(f"Input shape: {x.shape}")
        x1 = F.relu(self.in_conv(x))
        #print(f"After in_conv: {x1.shape}")
        x2 = self.encoder1(x1)
        #print(f"After encoder1: {x2.shape}")
        x3 = self.encoder2(x2)
        #print(f"After encoder2: {x3.shape}")
        x4 = self.encoder3(x3)
        #print(f"After encoder3: {x4.shape}")

        # Bridge
        x = F.relu(self.bridge_conv(x4))
        #print(f"After bridge_conv: {x.shape}")

        # Decoder path
        x = self.decoder3(x, x3)
        #print(f"After decoder3: {x.shape}")
        x = self.decoder2(x, x2)
        #print(f"After decoder2: {x.shape}")
        x = self.decoder1(x, x1)
        #print(f"After decoder1: {x.shape}")

        # Output layer
        out = self.out_conv(x)
        #print(f"Output shape: {out.shape}")
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) # Stride 2 for downsampling
        self.residuals = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_residuals)])
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.residuals(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # Stride 2 for upsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # in_channels is doubled after concatenation
        self.residuals = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_residuals)])
        self.prelu = nn.PReLU()

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.prelu(x)
        x = self.residuals(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        x = self.prelu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return x

if __name__ == '__main__':
    # Example usage:
    upscale = 2
    input_height = 32
    input_width = 64
    in_channels = 3
    batch_size = 1

    input_tensor = torch.randn(batch_size, in_channels, input_height, input_width) # BCHW
    model = SRResUNet(in_channels=in_channels, out_channels=in_channels, num_filters=32, num_residuals=2, upscale_factor=upscale)
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}") # Expect: [1, 3, 64, 128] if upscale_factor = 2

