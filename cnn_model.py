"""CNN Models for RGB to Hyperspectral Conversion"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with optional residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True, use_relu=True):
        super(ConvBlock, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class HSCNN(nn.Module):
    """
    Hyperspectral CNN (HSCNN) - Simple CNN for RGB to Hyperspectral conversion
    
    Architecture:
    - Input: RGB image (3 channels)
    - Output: Hyperspectral image (30 channels)
    """
    
    def __init__(self, num_bands=30, input_channels=3, hidden_channels=64, use_bn=True):
        super(HSCNN, self).__init__()
        
        self.num_bands = num_bands
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_bn = use_bn
        
        # Feature extraction layers
        self.layer1 = ConvBlock(input_channels, hidden_channels, kernel_size=3, use_bn=use_bn)
        self.layer2 = ConvBlock(hidden_channels, hidden_channels * 2, kernel_size=3, use_bn=use_bn)
        self.layer3 = ConvBlock(hidden_channels * 2, hidden_channels * 2, kernel_size=3, use_bn=use_bn)
        
        # Reconstruction layers
        self.layer4 = ConvBlock(hidden_channels * 2, hidden_channels, kernel_size=3, use_bn=use_bn)
        self.layer5 = ConvBlock(hidden_channels, num_bands, kernel_size=3, use_bn=False, use_relu=False)
        
    def forward(self, x):
        # Encoder
        x = self.layer1(x)  # 3 -> 64
        x = self.layer2(x)  # 64 -> 128
        x = self.layer3(x)  # 128 -> 128
        
        # Decoder
        x = self.layer4(x)  # 128 -> 64
        x = self.layer5(x)  # 64 -> 30 (output bands)
        
        return x


class ResidualHSCNN(nn.Module):
    """
    Hyperspectral CNN with Residual Connections
    """
    
    def __init__(self, num_bands=30, input_channels=3, hidden_channels=64, use_bn=True):
        super(ResidualHSCNN, self).__init__()
        
        self.num_bands = num_bands
        self.input_channels = input_channels
        
        # Initial layer to match dimensions if needed
        self.initial_conv = ConvBlock(input_channels, hidden_channels, kernel_size=3, use_bn=use_bn)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(hidden_channels, hidden_channels * 2, use_bn)
        self.res_block2 = self._make_residual_block(hidden_channels * 2, hidden_channels * 2, use_bn)
        self.res_block3 = self._make_residual_block(hidden_channels * 2, hidden_channels, use_bn)
        
        # Final output layer
        self.output_conv = ConvBlock(hidden_channels, num_bands, kernel_size=3, use_bn=False, use_relu=False)
        
    def _make_residual_block(self, in_channels, out_channels, use_bn):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, use_bn=use_bn),
            ConvBlock(out_channels, out_channels, kernel_size=3, use_bn=use_bn)
        )
    
    def forward(self, x):
        residual = x
        
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_conv(x)
        
        # Resize residual to match output channels (RGB -> 30 bands)
        # Use bilinear interpolation to expand the 3 channels to 30
        residual_expanded = F.interpolate(residual, size=x.shape[-2:], mode='bilinear', align_corners=False)
        residual_expanded = torch.cat([residual_expanded] * 10, dim=1)  # Repeat to match 30 channels
        
        return x


class DenseHSCNN(nn.Module):
    """
    Dense CNN for RGB to Hyperspectral conversion with multiple skip connections
    """
    
    def __init__(self, num_bands=30, input_channels=3, hidden_channels=64, use_bn=True):
        super(DenseHSCNN, self).__init__()
        
        self.num_bands = num_bands
        
        # Dense blocks with concatenation
        self.conv1 = ConvBlock(input_channels, hidden_channels, kernel_size=3, use_bn=use_bn)
        
        # Block 1
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3, use_bn=use_bn)
        self.conv3 = ConvBlock(hidden_channels * 2, hidden_channels, kernel_size=3, use_bn=use_bn)
        
        # Block 2
        self.conv4 = ConvBlock(hidden_channels * 3, hidden_channels * 2, kernel_size=3, use_bn=use_bn)
        self.conv5 = ConvBlock(hidden_channels * 4, hidden_channels, kernel_size=3, use_bn=use_bn)
        
        # Output layer
        self.output_conv = ConvBlock(hidden_channels * 5, num_bands, kernel_size=3, use_bn=False, use_relu=False)
    
    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = self.conv2(x1)
        x2_concat = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x2_concat)
        
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        
        out = self.output_conv(torch.cat([x1, x2, x3, x4, x5], dim=1))
        
        return out


def create_model(architecture="HSCNN", num_bands=30, input_channels=3, hidden_channels=64, use_bn=True):
    """Factory function to create models"""
    
    if architecture == "HSCNN":
        return HSCNN(num_bands=num_bands, input_channels=input_channels, 
                     hidden_channels=hidden_channels, use_bn=use_bn)
    elif architecture == "ResidualHSCNN":
        return ResidualHSCNN(num_bands=num_bands, input_channels=input_channels,
                             hidden_channels=hidden_channels, use_bn=use_bn)
    elif architecture == "DenseHSCNN":
        return DenseHSCNN(num_bands=num_bands, input_channels=input_channels,
                          hidden_channels=hidden_channels, use_bn=use_bn)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model("HSCNN", num_bands=30).to(device)
    print(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
