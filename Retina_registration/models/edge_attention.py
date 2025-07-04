"""
Attention Mechanisms for Medical Image Registration

This module implements various attention mechanisms used in the TransRetina
architecture for medical image registration tasks.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class EdgeGuidedAttentionModule(nn.Module):
    """
    Edge-guided Attention Module (EAM) for enhancing features using edge information.
    
    This module uses edge maps (e.g., from Canny edge detection) to guide the
    attention mechanism, enhancing features at edge locations which are crucial
    for medical image registration.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        reduction_ratio (int): Channel reduction ratio for efficiency. Default: 8
        
    Input:
        feature_map (torch.Tensor): Feature map [B, in_channels, H, W]
        edge_map (torch.Tensor): Edge map [B, H, W] or [B, 1, H, W]
        
    Output:
        torch.Tensor: Enhanced features [B, out_channels, H, W]
    """
    
    def __init__(self, in_channels, out_channels, reduction_ratio=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Edge feature processing branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()  # Generate attention weights
        )
        
        # Feature fusion branch using depthwise separable convolution
        self.feature_fusion = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        
        # Channel adjustment for residual connection
        if in_channels != out_channels:
            self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_adjust = nn.Identity()
            
        # Layer normalization
        self.layer_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
    def forward(self, feature_map, edge_map):
        """
        Forward pass of edge-guided attention.
        
        Args:
            feature_map (torch.Tensor): Input features [B, in_channels, H, W]
            edge_map (torch.Tensor): Edge map [B, H, W] or [B, 1, H, W]
            
        Returns:
            torch.Tensor: Enhanced features [B, out_channels, H, W]
        """
        # Ensure edge_map has correct dimensions
        if edge_map.dim() == 3:
            edge_map = edge_map.unsqueeze(1)  # [B, 1, H, W]
            
        # Generate edge-guided attention weights
        edge_attention = self.edge_conv(edge_map)  # [B, in_channels, H, W]
        
        # Apply attention weights
        attended_features = feature_map * edge_attention
        
        # Feature fusion
        fused_features = self.feature_fusion(attended_features)
        
        # Residual connection with channel adjustment
        residual = self.channel_adjust(feature_map)
        
        # Learnable weighted fusion
        output = self.alpha * fused_features + self.beta * residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class PixelAffinityAttentionHead(nn.Module):
    """
    Pixel Affinity Guidance (PAG) Attention Head.
    
    This module computes pixel-wise affinity scores between fixed and moving images
    to guide the registration process. It generates an affinity matrix that captures
    the similarity between corresponding pixels.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        use_gpu (bool): Whether to use GPU. Default: False
        device (str): Device to use. Default: "cpu"
        
    Input:
        decode_feature (torch.Tensor): Decoded features [B, in_channels, H, W]
        
    Output:
        torch.Tensor: Affinity score matrix [B, N, N] where N = (H/8) * (W/8)
    """
    
    def __init__(self, in_channels, out_channels, use_gpu=False, device="cpu"):
        super().__init__()
        
        self.down_sample = nn.MaxPool2d(kernel_size=8)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if use_gpu:
            self.down_sample = self.down_sample.to(device)
            self.conv3x3 = self.conv3x3.to(device)
            
    def forward(self, decode_feature):
        """
        Forward pass to compute pixel affinity matrix.
        
        Args:
            decode_feature (torch.Tensor): Input features [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Affinity score matrix [B, N, N]
        """
        batch_size = decode_feature.shape[0]

        # 3×3 convolution for dimension reduction
        decode_feature2 = self.conv3x3(decode_feature)  # [B, out_channels, H, W]

        # Split features into fixed and moving parts
        decode_fix, decode_mov = decode_feature2.chunk(2, dim=1)  # [B, out_channels//2, H, W]

        # Downsample and permute dimensions
        fix_feature = self.down_sample(decode_fix).permute(0, 2, 3, 1)  # [B, H//8, W//8, out_channels//2]
        mov_feature = self.down_sample(decode_mov).permute(0, 2, 3, 1)  # [B, H//8, W//8, out_channels//2]

        # Flatten spatial dimensions
        fix_feature = fix_feature.view(batch_size, -1, decode_fix.shape[1])  # [B, N, out_channels//2]
        mov_feature = mov_feature.view(batch_size, -1, decode_fix.shape[1])  # [B, N, out_channels//2]
        
        # Compute affinity matrix via batch matrix multiplication
        attention_score_matrix = torch.bmm(fix_feature, mov_feature.permute(0, 2, 1))  # [B, N, N]
        
        return attention_score_matrix