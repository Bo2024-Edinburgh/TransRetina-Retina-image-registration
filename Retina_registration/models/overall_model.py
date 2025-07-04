import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transretina import TransRetina


def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
    """
    Constructs the final convolutional block for deformation field prediction.
    
    This block consists of two consecutive convolution-batchnorm-relu operations
    to generate the final displacement vector field for image registration.
    
    Args:
        in_channels (int): Number of input channels from the feature maps
        mid_channel (int): Number of intermediate channels in the first conv layer
        out_channels (int): Number of output channels (typically 2 for 2D displacement)
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
    
    Returns:
        torch.nn.Sequential: Sequential block containing conv-bn-relu layers
    """
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, 
                       out_channels=mid_channel, padding=1),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, 
                       out_channels=out_channels, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )
    return block


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network for image warping based on VoxelMorph architecture.
    
    This module performs spatial transformation by applying displacement fields
    to warp moving images according to the predicted deformation field.
    
    Attributes:
        mode (str): Interpolation mode for grid sampling
        device (str): Device for computation (cpu/cuda)
        grid (torch.Tensor): Base coordinate grid for transformation
    """
    
    def __init__(self, size, device="cpu", mode='bilinear'):
        """
        Initialize the spatial transformer.
        
        Args:
            size (tuple): Spatial dimensions of the input images (height, width)
            device (str, optional): Computation device. Defaults to "cpu".
            mode (str, optional): Grid sampling interpolation mode. Defaults to 'bilinear'.
        """
        super().__init__()

        self.mode = mode
        self.device = device
        
        # Create base coordinate grid for spatial transformation
        vectors = [torch.arange(0, s) for s in size]  # Generate coordinate vectors
        grids = torch.meshgrid(vectors)               # Create meshgrid (grid_y, grid_x)
        grid = torch.stack(grids)                     # Stack to [2, H, W]
        grid = torch.unsqueeze(grid, 0)               # Add batch dimension [1, 2, H, W]
        grid = grid.type(torch.FloatTensor).to(self.device)

        # Register as buffer to avoid gradient computation
        self.register_buffer('grid', grid)

    def forward(self, moving_image, flow):
        """
        Apply spatial transformation to the moving image using displacement field.
        
        Args:
            moving_image (torch.Tensor): Input moving image [B, C, H, W]
            flow (torch.Tensor): Displacement field [B, 2, H, W]
        
        Returns:
            torch.Tensor: Warped/registered image [B, C, H, W]
        """
        B, _, H, W = flow.shape

        # Generate coordinate grid for current batch
        vectors = [torch.arange(0, s, device=flow.device) for s in (H, W)]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).float()  # [B, 2, H, W]
        
        # Apply displacement field to base coordinates
        new_locs = grid + flow   # Add displacement vectors [B, 2, H, W]
        
        # Normalize coordinates to [-1, 1] range for grid_sample
        new_locs[:, 0, :, :] = 2 * (new_locs[:, 0, :, :] / (H - 1) - 0.5)
        new_locs[:, 1, :, :] = 2 * (new_locs[:, 1, :, :] / (W - 1) - 0.5)

        # Reshape and swap x,y coordinates for grid_sample format
        new_locs = new_locs.permute(0, 2, 3, 1)  # [B, H, W, 2]
        new_locs = new_locs[..., [1, 0]]         # Swap x and y coordinates

        # Perform bilinear sampling to warp the moving image
        return F.grid_sample(moving_image, new_locs, align_corners=True, mode=self.mode)


class RetinalTransUNet(nn.Module):
    """
    Retinal Image Registration Network using TransRetina architecture.
    
    This network performs deformable registration of retinal images by combining
    TransRetina for feature extraction with spatial transformation for image warping.
    Supports optional edge-guided attention (EAM) and pixel affinity guidance (PAG).
    
    Architecture Components:
    - TransRetina: Feature extraction from concatenated moving and fixed images
    - Edge-guided Attention Module (EAM): Leverages edge information for better alignment
    - Pixel Affinity Guidance (PAG): Computes pixel-wise similarity scores
    - Spatial Transformer: Warps moving image using predicted deformation field
    """
    
    def __init__(self, in_channels, config_train, use_gpu=False, device="cpu"):
        """
        Initialize the RetinalTransUNet registration network.
        
        Args:
            in_channels (int): Number of input channels (typically 6 for concatenated RGB pairs)
            config_train (dict): Training configuration containing model parameters
            use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
            device (str, optional): Computation device. Defaults to "cpu".
        """
        super(RetinalTransUNet, self).__init__()

        # Initialize TransRetina backbone for feature extraction
        self.TransRetina = TransRetina(
            in_channels=in_channels, 
            img_size=config_train['train']["model_image_width"],
            patch_size=1,
            embed_dim=512,
            depth=12,
            heads=16,
            mlp_dim=2048,
            dropout=0.1
        )
        
        # Final convolution block to predict 2D displacement field
        # Output: 2 channels for (dx, dy) displacement vectors
        self.final_layer = final_block(64, 32, 2)

        # Initialize spatial transformer for image warping
        shape = (config_train['train']["model_image_width"], 
                config_train['train']["model_image_width"])
        self.transformer = SpatialTransformer(shape, device)
        
        # Configuration flags for optional modules
        self.Em_map = config_train['train']["Em_map"]    # Edge-guided attention flag
        self.PAG_map = config_train['train']["PAG_map"]  # Pixel affinity guidance flag
        
        # Initialize optional modules based on configuration
        if self.PAG_map:
            # Pixel Affinity Guidance head for computing similarity matrix [B, N, N]
            self.at_head = PAG_attention_head(64, 30)
            
        if self.Em_map:
            # Edge-guided attention module for incorporating edge information
            self.guide_attention = AttentionModule(64, 64, 8)
        
        # Move modules to specified device if GPU is enabled
        if use_gpu:
            self.TransRetina = self.TransRetina.to(device)
            self.transformer = self.transformer.to(device)
            self.final_layer = self.final_layer.to(device)
            if self.Em_map:
                self.guide_attention = self.guide_attention.to(device)
            if self.PAG_map:
                self.at_head = self.at_head.to(device)

    def forward(self, moving_image, fixed_image, guide_map):
        """
        Forward pass for retinal image registration.
        
        Args:
            moving_image (torch.Tensor): Moving image to be registered [B, 3, H, W]
            fixed_image (torch.Tensor): Reference fixed image [B, 3, H, W]
            guide_map (torch.Tensor): Edge guidance map [B, H, W]
        
        Returns:
            tuple: Registration results containing:
                - registered_image (torch.Tensor): Warped moving image [B, 3, H, W]
                - deformation_matrix (torch.Tensor): Displacement field [B, 2, H, W]
                - attention_score_matrix (torch.Tensor, optional): Pixel affinity scores [B, N, N]
        """
        # Concatenate moving and fixed images along channel dimension
        # Input shapes: moving_image [B, 3, H, W], fixed_image [B, 3, H, W]
        # Output shape: x [B, 6, H, W]
        x = torch.cat([moving_image, fixed_image], dim=1)

        # Extract multi-scale features using TransRetina backbone
        decode_block1 = self.TransRetina(x)  # Output: [B, 64, H, W]

        # Apply edge-guided attention if enabled
        if self.Em_map: 
            # Edge-guided Attention Module (EAM)
            # Incorporates edge information to enhance feature representation
            # Input: decode_block1 [B, 64, H, W], guide_map [B, H, W]
            # Output: decode_block0 [B, 64, H, W]
            decode_block0 = self.guide_attention(decode_block1, guide_map)

            # Generate displacement field from edge-enhanced features
            deformation_matrix = self.final_layer(decode_block0)  # [B, 2, H, W]

            # Compute pixel affinity scores if PAG is enabled
            if self.PAG_map:
                # Pixel Affinity Guidance (PAG) branch
                # Computes pairwise similarity scores between pixels
                # Output: [B, N*N] where N = H*W (flattened spatial dimensions)
                attention_score_matrix = self.at_head(decode_block0)
        else:
            # Direct displacement field prediction without edge guidance
            deformation_matrix = self.final_layer(decode_block1)  # [B, 2, H, W]
            
            # Compute pixel affinity scores from raw features if PAG is enabled
            if self.PAG_map:
                attention_score_matrix = self.at_head(decode_block1)

        # Apply spatial transformation to warp the moving image
        # Input: moving_image [B, 3, H, W], deformation_matrix [B, 2, H, W]
        # Output: registered_image [B, 3, H, W]
        registered_image = self.transformer(moving_image, deformation_matrix)
        
        # Return results based on configuration
        if self.PAG_map:
            return registered_image, deformation_matrix, attention_score_matrix
        else:
            return registered_image, deformation_matrix