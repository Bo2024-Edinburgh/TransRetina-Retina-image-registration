"""
TransUNet Model for Medical Image Segmentation/Registration

This module implements the TransUNet architecture, which combines the strengths
of CNN encoders for local feature extraction and Transformer encoders for 
global context modeling.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.cnn_encoder import CNNEncoder
from models.transformer import TransformerEncoder



class TransRetina(nn.Module):
    """
    TransUNet: Transformer-Enhanced U-Net for Medical Image Analysis.
    
    This architecture combines:
    1. CNN encoder for local feature extraction and multi-scale representations
    2. Transformer encoder for global context modeling
    3. CNN decoder with skip connections for precise localization
    
    The model processes images through a hierarchical CNN encoder, applies
    transformer attention at the bottleneck, and reconstructs features through
    a decoder with skip connections.
    
    Args:
        in_channels (int): Number of input channels. Default: 6
        img_size (int): Input image size. Default: 768
        patch_size (int): Patch size for transformer (not used with CNN features). Default: 1
        embed_dim (int): Transformer embedding dimension. Default: 512
        depth (int): Number of transformer layers. Default: 12
        heads (int): Number of attention heads. Default: 16
        mlp_dim (int): MLP dimension in transformer. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
        
    Input:
        x (torch.Tensor): Input tensor [B, in_channels, img_size, img_size]
        
    Output:
        torch.Tensor: Feature map [B, 64, img_size, img_size]
    """
    
    def __init__(self, in_channels=6, img_size=768, patch_size=1, embed_dim=512, 
                 depth=12, heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        
        # Store hyperparameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN Encoder for hierarchical feature extraction
        self.cnn_encoder = CNNEncoder(in_channels)
        
        # Calculate feature map dimensions after CNN encoder
        # CNN encoder reduces spatial dimensions by factor of 32
        # 768 -> 384 -> 192 -> 96 -> 48 -> 24
        self.feature_size = img_size // 32  # 24x24
        num_patches = self.feature_size ** 2  # 576 patches
        
        # Patch embedding projection for transformer input
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer encoder for global context modeling
        self.transformer = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=embed_dim // heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # Decoder for feature reconstruction
        self.decoder = self._build_decoder()
        
    def _build_decoder(self):
        """
        Build the decoder with upsampling and skip connections.
        
        Returns:
            nn.ModuleDict: Dictionary containing decoder components
        """
        decoder = nn.ModuleDict()
        
        # Transform transformer output back to CNN feature space
        decoder['reshape'] = nn.Conv2d(self.embed_dim, 512, kernel_size=1)
        
        # Upsampling stages with skip connections
        # Stage 1: 512 -> 256, 24x24 -> 48x48
        decoder['up1'] = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        decoder['conv1'] = conv_block(512, 256)  # 256+256=512 from skip connection
        
        # Stage 2: 256 -> 128, 48x48 -> 96x96
        decoder['up2'] = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        decoder['conv2'] = conv_block(256, 128)  # 128+128=256
        
        # Stage 3: 128 -> 64, 96x96 -> 192x192
        decoder['up3'] = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        decoder['conv3'] = conv_block(128, 64)  # 64+64=128
        
        # Stage 4: 64 -> 64, 192x192 -> 384x384
        decoder['up4'] = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        decoder['conv4'] = conv_block(128, 64)  # 64+64=128
        
        # Stage 5: 64 -> 64, 384x384 -> 768x768
        decoder['up5'] = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        return decoder



    def _conv_block(self, in_channels, out_channels):
        """
        Standard convolution block with BatchNorm and ReLU.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Convolution kernel size. Default: 3
            
        Returns:
            nn.Sequential: Convolution block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through TransUNet.
        
        Args:
            x (torch.Tensor): Input tensor [B, in_channels, img_size, img_size]
            
        Returns:
            torch.Tensor: Output feature map [B, 64, img_size, img_size]
        """
        # Step 1: CNN encoding with skip connections
        cnn_features, skip_features = self.cnn_encoder(x)  # [B, 512, 24, 24]
        
        # Prepare for transformer
        B, C, H, W = cnn_features.shape
        
        # Patch embedding
        x_embed = self.patch_embed(cnn_features)  # [B, embed_dim, 24, 24]
        x_embed = rearrange(x_embed, 'b c h w -> b (h w) c')  # [B, 576, embed_dim]
        
        # Add positional embedding
        x_embed += self.pos_embedding
        
        # Transformer
        x_trans = self.transformer(x_embed)  # [B, 576, embed_dim]
        
        # Reshape back to feature map
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=H, w=W)  # [B, embed_dim, 24, 24]
        
        # Decoder with skip connections
        x = self.decoder['reshape'](x_trans)  # [B, 512, 24, 24]
        
        # Upsampling with skip connections (reverse order of skip_features)
        skip_features = skip_features[::-1]  # Reverse for decoder
        
        # Stage 1: 24x24 -> 48x48
        x = self.decoder['up1'](x)  # [B, 256, 48, 48]
        x = torch.cat([x, skip_features[1]], dim=1)  # skip from stage3
        x = self.decoder['conv1'](x)  # [B, 256, 48, 48]
        
        # Stage 2: 48x48 -> 96x96
        x = self.decoder['up2'](x)  # [B, 128, 96, 96]
        x = torch.cat([x, skip_features[2]], dim=1)  # skip from stage2
        x = self.decoder['conv2'](x)  # [B, 128, 96, 96]
        
        # Stage 3: 96x96 -> 192x192
        x = self.decoder['up3'](x)  # [B, 64, 192, 192]
        x = torch.cat([x, skip_features[3]], dim=1)  # skip from stage1
        x = self.decoder['conv3'](x)  # [B, 64, 192, 192]
        
        # Stage 4: 192x192 -> 384x384
        x = self.decoder['up4'](x)  # [B, 64, 384, 384]
        # Resize skip connection to match
        skip_resized = F.interpolate(skip_features[4], size=(384, 384), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_resized], dim=1)
        x = self.decoder['conv4'](x)  # [B, 64, 384, 384]
        
        # Stage 5: 384x384 -> 768x768
        x = self.decoder['up5'](x)  # [B, 64, 768, 768]
        
        return x