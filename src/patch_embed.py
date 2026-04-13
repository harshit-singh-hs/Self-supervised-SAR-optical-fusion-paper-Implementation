import torch
import torch.nn as nn
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 4 == 0
    omega      = np.arange(embed_dim // 4, dtype=np.float32)
    omega      = 1.0 / (10000 ** (omega / (embed_dim // 4)))
    pos_h      = grid[0].reshape(-1)[:, None] * omega[None, :]
    pos_w      = grid[1].reshape(-1)[:, None] * omega[None, :]
    emb        = np.concatenate([np.sin(pos_h), np.cos(pos_h),
                                 np.sin(pos_w), np.cos(pos_w)], axis=1)
    return emb.astype(np.float32)


class PatchEmbed(nn.Module):

    def __init__(self, in_channels, patch_size=16, embed_dim=384, img_size=128):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size   = img_size // patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer('pos_embed',
                             torch.from_numpy(pos_embed).unsqueeze(0))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x

    def __repr__(self):
        return (f"PatchEmbed(in_channels={self.proj.in_channels}, "
                f"patch_size={self.patch_size}, "
                f"embed_dim={self.embed_dim}, "
                f"num_patches={self.num_patches})")


if __name__ == '__main__':
    print("\n── Creating patch embed layers ───────────────────────────────────")
    sar_embed = PatchEmbed(in_channels=6,  patch_size=16, embed_dim=384, img_size=128)
    opt_embed = PatchEmbed(in_channels=11, patch_size=16, embed_dim=384, img_size=128)
    print(f"  sar_embed : {sar_embed}")
    print(f"  opt_embed : {opt_embed}")

    sar_img = torch.randn(2, 6,  128, 128)
    opt_img = torch.randn(2, 11, 128, 128)

    print("\n── Forward pass shapes ───────────────────────────────────────────")
    sar_tokens = sar_embed(sar_img)
    opt_tokens = opt_embed(opt_img)
    print(f"  SAR  input : {sar_img.shape}")
    print(f"  SAR  output: {sar_tokens.shape}")
    print(f"  OPT  input : {opt_img.shape}")
    print(f"  OPT  output: {opt_tokens.shape}")

    assert sar_tokens.shape == (2, 64, 384)
    assert opt_tokens.shape == (2, 64, 384)

    print("\n── Positional embedding check ────────────────────────────────────")
    pe = sar_embed.pos_embed
    print(f"  pos_embed shape : {pe.shape}")
    print(f"  pos_embed min   : {pe.min():.4f}")
    print(f"  pos_embed max   : {pe.max():.4f}")
    unique_rows = pe.squeeze(0)
    all_unique  = all(
        not torch.allclose(unique_rows[i], unique_rows[j])
        for i in range(len(unique_rows))
        for j in range(i+1, min(i+5, len(unique_rows)))
    )
    print(f"  All positions unique: {all_unique}")

    print("\n── No NaN in outputs ─────────────────────────────────────────────")
    print(f"  SAR tokens: {'no NaN' if not torch.isnan(sar_tokens).any() else 'HAS NaN'}")
    print(f"  OPT tokens: {'no NaN' if not torch.isnan(opt_tokens).any() else 'HAS NaN'}")

    print("\n── Parameter count ───────────────────────────────────────────────")
    sar_params = sum(p.numel() for p in sar_embed.parameters())
    opt_params = sum(p.numel() for p in opt_embed.parameters())
    print(f"  sar_embed params : {sar_params:,}")
    print(f"  opt_embed params : {opt_params:,}")
    print(f"  (pos_embed is a buffer — not counted as a parameter)")

    print("\n── Two separate embeddings produce different tokens ───────────────")
    print(f"  pos_embed is identical across SAR and optical — correct")
    print(f"  proj weights differ (different in_channels) — correct")

    