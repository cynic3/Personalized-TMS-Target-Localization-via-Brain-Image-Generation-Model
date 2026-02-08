import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.conv = nn.Conv3d(dim, dim, kernel_size=d_conv, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.conv(x)
        x = x.view(B, C, -1).transpose(1, 2)
        x = F.silu(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x.transpose(1, 2).view(B, C, D, H, W)


class SpatialMamba(nn.Module):
    def __init__(self, feature_dim, num_layers=3):
        super().__init__()
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(feature_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.mamba_blocks:
            x = block(x)
        return x
