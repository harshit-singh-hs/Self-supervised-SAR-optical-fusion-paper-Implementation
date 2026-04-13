import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q    = nn.Linear(dim, dim, bias=qkv_bias)
        self.k    = nn.Linear(dim, dim, bias=qkv_bias)
        self.v    = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, D = x.shape
        _, M, _ = context.shape

        Q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        return out


class XAttnEncoder(nn.Module):

    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()

        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_y = nn.LayerNorm(dim)
        self.ca_x2y  = CrossAttention(dim, num_heads, qkv_bias, attn_drop=drop)
        self.ca_y2x  = CrossAttention(dim, num_heads, qkv_bias, attn_drop=drop)

        self.norm2   = nn.LayerNorm(dim)
        mlp_hidden   = int(dim * mlp_ratio)
        self.mlp     = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, y):
        x_n = self.norm1_x(x)
        y_n = self.norm1_y(y)

        ca_xy = self.ca_x2y(x_n, y_n)
        ca_yx = self.ca_y2x(y_n, x_n)

        fused = torch.cat([x, y], dim=1) + torch.cat([ca_xy, ca_yx], dim=1)
        fused = fused + self.mlp(self.norm2(fused))
        return fused


class XAttnDecoder(nn.Module):

    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()

        self.norm1_sar = nn.LayerNorm(dim)
        self.norm1_opt = nn.LayerNorm(dim)
        self.ca_sar    = CrossAttention(dim, num_heads, qkv_bias, attn_drop=drop)
        self.ca_opt    = CrossAttention(dim, num_heads, qkv_bias, attn_drop=drop)

        self.norm2_sar = nn.LayerNorm(dim)
        self.norm2_opt = nn.LayerNorm(dim)
        mlp_hidden     = int(dim * mlp_ratio)
        self.mlp_sar   = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )
        self.mlp_opt   = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )

    def forward(self, z_sar, z_opt):
        z_sar_cross = z_sar + self.ca_sar(self.norm1_sar(z_sar), self.norm1_opt(z_opt))
        z_opt_cross = z_opt + self.ca_opt(self.norm1_opt(z_opt), self.norm1_sar(z_sar))

        z_sar_out = z_sar_cross + self.mlp_sar(self.norm2_sar(z_sar_cross))
        z_opt_out = z_opt_cross + self.mlp_opt(self.norm2_opt(z_opt_cross))

        return z_sar_out, z_opt_out


if __name__ == '__main__':
    B, N, D = 2, 64, 384
    heads   = 6

    x = torch.randn(B, N, D)
    y = torch.randn(B, N, D)

    print("\n── CrossAttention ────────────────────────────────────────────────")
    ca  = CrossAttention(dim=D, num_heads=heads)
    out = ca(x, y)
    print(f"  input x  : {x.shape}")
    print(f"  context y: {y.shape}")
    print(f"  output   : {out.shape}")
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()

    print("\n── XAttnEncoder ──────────────────────────────────────────────────")
    xae   = XAttnEncoder(dim=D, num_heads=heads)
    fused = xae(x, y)
    print(f"  input x      : {x.shape}")
    print(f"  input y      : {y.shape}")
    print(f"  fused output : {fused.shape}   (token dim doubled, feature dim unchanged)")
    assert fused.shape == (B, N * 2, D), f"Expected (2,128,384), got {fused.shape}"
    assert not torch.isnan(fused).any()

    print("\n── XAttnDecoder ──────────────────────────────────────────────────")
    xad             = XAttnDecoder(dim=D, num_heads=heads)
    z_sar           = torch.randn(B, N, D)
    z_opt           = torch.randn(B, N, D)
    out_sar, out_opt = xad(z_sar, z_opt)
    print(f"  z_sar input  : {z_sar.shape}")
    print(f"  z_opt input  : {z_opt.shape}")
    print(f"  z_sar output : {out_sar.shape}")
    print(f"  z_opt output : {out_opt.shape}")
    assert out_sar.shape == (B, N, D)
    assert out_opt.shape == (B, N, D)
    assert not torch.isnan(out_sar).any()
    assert not torch.isnan(out_opt).any()

    print("\n── Backward pass ─────────────────────────────────────────────────")
    loss = fused.sum() + out_sar.sum() + out_opt.sum()
    loss.backward()
    print("  backward pass completed without error")
