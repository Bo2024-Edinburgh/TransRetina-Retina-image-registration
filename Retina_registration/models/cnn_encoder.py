"""
CNN Encoder Components for Medical Image Registration

This module implements the CNN encoder part of the hybrid CNN-Transformer
architecture, following a ResNet-like structure for feature extraction.

"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN Encoder with ResNet-like architecture for feature extraction.
    
    This encoder follows a hierarchical feature extraction approach with
    multiple stages of downsampling, generating multi-scale features
    suitable for skip connections in the decoder.
    
    Architecture:
    - Initial conv + maxpool: 768×768 -> 192×192
    - Stage 1: 64 channels, 192×192
    - Stage 2: 128 channels, 96×96  
    - Stage 3: 256 channels, 48×48
    - Stage 4: 512 channels, 24×24 (fed to transformer)
    
    Args:
        in_channels (int): Number of input channels. Default: 6 (concat of two 3-channel images)
        
    Input:
        x (torch.Tensor): Input tensor [B, in_channels, 768, 768]
        
    Output:
        Tuple[torch.Tensor, List[torch.Tensor]]: 
            - Final features [B, 512, 24, 24]
            - List of intermediate features for skip connections
    """
    
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Initial convolution and max pooling
        # 768×768 -> 384×384 -> 192×192
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Hierarchical feature extraction stages
        # Stage 1: 64 channels, maintain 192×192 resolution
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        
        # Stage 2: 128 channels, downsample to 96×96
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        
        # Stage 3: 256 channels, downsample to 48×48
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        
        # Stage 4: 512 channels, downsample to 24×24 (input to transformer)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage with multiple ResNet blocks.
        
        Args:
            in_channels (int): Input channels for the stage
            out_channels (int): Output channels for the stage
            num_blocks (int): Number of ResNet blocks in the stage
            stride (int): Stride for the first block (for downsampling)
            
        Returns:
            nn.Sequential: Sequential container of ResNet blocks
        """
        layers = []
        
        # First block with potential stride change
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        # Remaining blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride):
        """Basic ResNet block"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        return block
    
    def forward(self, x):
        """
        Forward pass through the CNN encoder.
        
        Args:
            x (torch.Tensor): Input tensor [B, in_channels, 768, 768]
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - features: Final feature map [B, 512, 24, 24]
                - skip_features: List of intermediate features for skip connections
                  [conv1_out, stage1_out, stage2_out, stage3_out, stage4_out]
        """
        # Store intermediate features for skip connections
        features = []
        
        # Initial convolution: [B, in_channels, 768, 768] -> [B, 64, 192, 192]
        x = self.conv1(x)  # [B, 64, 192, 192]
        features.append(x)
        
        # Stage 1: [B, 64, 192, 192] -> [B, 64, 192, 192]
        x = self.stage1(x)  # [B, 64, 192, 192]
        features.append(x)
        
        # Stage 2: [B, 64, 192, 192] -> [B, 128, 96, 96]
        x = self.stage2(x)  # [B, 128, 96, 96]
        features.append(x)
        
        # Stage 3: [B, 128, 96, 96] -> [B, 256, 48, 48]
        x = self.stage3(x)  # [B, 256, 48, 48]
        features.append(x)
        
        # Stage 4: [B, 256, 48, 48] -> [B, 512, 24, 24]
        x = self.stage4(x)  # [B, 512, 24, 24]
        features.append(x)
        
        return x, features