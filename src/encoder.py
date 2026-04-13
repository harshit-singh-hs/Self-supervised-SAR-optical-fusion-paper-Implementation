import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Attention(nn.Module):

    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, qkv_bias, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):

    def __init__(self, dim=384, depth=11, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, use_grad_checkpoint=True):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            if self.use_grad_checkpoint and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    B, N, D = 2, 129, 384

    print("\n── TransformerBlock ──────────────────────────────────────────────")
    block = TransformerBlock(dim=D, num_heads=6)
    x     = torch.randn(B, N, D)
    out   = block(x)
    print(f"  input : {x.shape}")
    print(f"  output: {out.shape}")
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()

    print("\n── ViTEncoder (11 blocks, grad checkpoint off for test) ──────────")
    encoder = ViTEncoder(dim=D, depth=11, num_heads=6, use_grad_checkpoint=False)
    x       = torch.randn(B, N, D)
    out     = encoder(x)
    print(f"  input : {x.shape}")
    print(f"  output: {out.shape}")
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()

    total_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  encoder params: {total_params:.1f}M")

    print("\n── Backward pass ─────────────────────────────────────────────────")
    loss = out.sum()
    loss.backward()
    print("  backward pass completed without error")

    