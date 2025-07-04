"""
Transformer Components for Medical Image Registration

This module implements Transformer-based components including feed-forward networks,
transformer blocks, and encoder architectures for the TransRetina model.

"""

import torch
import torch.nn as nn



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Transformer blocks.
    
    This implementation follows the standard Transformer architecture with
    scaled dot-product attention across multiple heads.
    
    Args:
        dim (int): Input dimension of the features
        heads (int): Number of attention heads. Default: 8
        dim_head (int): Dimension of each attention head. Default: 64
        dropout (float): Dropout probability. Default: 0.0
        
    Input:
        x (torch.Tensor): Input tensor of shape [B, N, dim]
        
    Output:
        torch.Tensor: Output tensor of shape [B, N, dim]
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, dim]
            
        Returns:
            torch.Tensor: Attention output [B, N, dim]
        """
        # Generate Q, K, V matrices
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) used in Transformer blocks.
    
    Implements a two-layer MLP with GELU activation and dropout for
    non-linear feature transformation in Transformer architectures.
    
    Args:
        dim (int): Input and output dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout probability. Default: 0.0
        
    Input:
        x (torch.Tensor): Input tensor [B, N, dim]
        
    Output:
        torch.Tensor: Output tensor [B, N, dim]
    """
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, dim]
            
        Returns:
            torch.Tensor: Transformed tensor [B, N, dim]
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block with Multi-Head Attention and Feed-Forward Network.
    
    Implements the standard transformer block with:
    - Layer normalization (pre-norm)
    - Multi-head self-attention
    - Residual connections
    - Feed-forward network
    
    Args:
        dim (int): Feature dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension per attention head
        mlp_dim (int): Hidden dimension in feed-forward network
        dropout (float): Dropout probability. Default: 0.0
        
    Input:
        x (torch.Tensor): Input tensor [B, N, dim]
        
    Output:
        torch.Tensor: Output tensor [B, N, dim]
    """
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, dim]
            
        Returns:
            torch.Tensor: Output tensor [B, N, dim]
        """
        # Multi-head attention with residual connection
        x = self.attn(self.norm1(x)) + x
        
        # Feed-forward network with residual connection
        x = self.ffn(self.norm2(x)) + x
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple Transformer blocks.
    
    Stacks multiple transformer blocks to create a deep transformer encoder
    for processing sequential or spatial feature representations.
    
    Args:
        dim (int): Feature dimension
        depth (int): Number of transformer blocks
        heads (int): Number of attention heads per block
        dim_head (int): Dimension per attention head
        mlp_dim (int): Hidden dimension in feed-forward networks
        dropout (float): Dropout probability. Default: 0.0
        
    Input:
        x (torch.Tensor): Input tensor [B, N, dim]
        
    Output:
        torch.Tensor: Encoded tensor [B, N, dim]
    """
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x):
        """
        Forward pass through all transformer blocks.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, dim]
            
        Returns:
            torch.Tensor: Encoded tensor [B, N, dim]
        """
        for transformer_block in self.layers:
            x = transformer_block(x)
        return x