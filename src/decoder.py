import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import TransformerBlock


class MAEDecoder(nn.Module):

    def __init__(self, encoder_dim=384, decoder_dim=192, depth=4,
                 num_heads=6, mlp_ratio=4.0, patch_size=16, out_channels=1):
        super().__init__()
        self.decoder_dim  = decoder_dim
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.patch_dim    = patch_size * patch_size * out_channels

        self.input_proj  = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks      = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(decoder_dim)
        self.output_proj = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latents, ids_restore):
        B       = latents.shape[0]
        N_total = ids_restore.shape[1]

        x = self.input_proj(latents)

        N_visible = x.shape[1]
        N_masked  = N_total - N_visible
        mask_tokens = self.mask_token.expand(B, N_masked, -1)

        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )

        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        x_full = self.output_proj(x_full)
        return x_full


if __name__ == '__main__':
    import torch

    B          = 2
    N_total    = 64
    N_visible  = 16
    N_masked   = N_total - N_visible
    enc_dim    = 384
    dec_dim    = 192
    patch_size = 16

    noise      = torch.rand(B, N_total)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    latents = torch.randn(B, N_visible, enc_dim)

    print("\n── SAR decoder (6 output channels) ──────────────────────────────")
    sar_decoder = MAEDecoder(
        encoder_dim=enc_dim, decoder_dim=dec_dim, depth=4,
        num_heads=6, patch_size=patch_size, out_channels=6
    )
    out_sar = sar_decoder(latents, ids_restore)
    expected_sar = patch_size * patch_size * 6
    print(f"  latents input : {latents.shape}")
    print(f"  output        : {out_sar.shape}")
    print(f"  expected last dim: {expected_sar}  (16×16×6)")
    assert out_sar.shape == (B, N_total, expected_sar), \
        f"Expected (2, 64, {expected_sar}), got {out_sar.shape}"
    assert not torch.isnan(out_sar).any()

    print("\n── Optical decoder (11 output channels) ─────────────────────────")
    opt_decoder = MAEDecoder(
        encoder_dim=enc_dim, decoder_dim=dec_dim, depth=4,
        num_heads=6, patch_size=patch_size, out_channels=11
    )
    out_opt = opt_decoder(latents, ids_restore)
    expected_opt = patch_size * patch_size * 11
    print(f"  latents input : {latents.shape}")
    print(f"  output        : {out_opt.shape}")
    print(f"  expected last dim: {expected_opt}  (16×16×11)")
    assert out_opt.shape == (B, N_total, expected_opt), \
        f"Expected (2, 64, {expected_opt}), got {out_opt.shape}"
    assert not torch.isnan(out_opt).any()

    print("\n── Backward pass ─────────────────────────────────────────────────")
    loss = out_sar.sum() + out_opt.sum()
    loss.backward()
    print("  backward pass completed without error")

    total_sar = sum(p.numel() for p in sar_decoder.parameters()) / 1e6
    total_opt = sum(p.numel() for p in opt_decoder.parameters()) / 1e6
    print(f"\n  sar_decoder params: {total_sar:.2f}M")
    print(f"  opt_decoder params: {total_opt:.2f}M")

    