import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patch_embed import PatchEmbed
from src.cross_attn  import XAttnEncoder, XAttnDecoder
from src.encoder     import ViTEncoder
from src.decoder     import MAEDecoder


class FusMAE(nn.Module):

    def __init__(
        self,
        img_size            = 128,
        patch_size          = 16,
        sar_channels        = 6,
        opt_channels        = 11,
        embed_dim           = 384,
        encoder_depth       = 11,
        decoder_dim         = 192,
        decoder_depth       = 4,
        num_heads           = 6,
        mlp_ratio           = 4.0,
        mask_ratio          = 0.75,
        masking             = 'consistent',
        fusion              = 'xae',
        use_grad_checkpoint = True,
    ):
        super().__init__()

        self.mask_ratio  = mask_ratio
        self.masking     = masking
        self.fusion      = fusion
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Modality-specific patch embedding — separate Conv2d per modality
        # because SAR has 6 channels and optical has 11 channels
        self.sar_embed = PatchEmbed(sar_channels, patch_size, embed_dim, img_size)
        self.opt_embed = PatchEmbed(opt_channels, patch_size, embed_dim, img_size)

        # XAttnEncoder replaces the first transformer block (paper section 3.1)
        self.xattn_encoder = XAttnEncoder(embed_dim, num_heads, mlp_ratio)

        # CLS token appended after XAttnEncoder, before ViTEncoder (follows MultiMAE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Remaining 11 standard transformer blocks (total = 1 XAttn + 11 standard = 12)
        self.encoder = ViTEncoder(
            embed_dim, encoder_depth, num_heads,
            mlp_ratio, use_grad_checkpoint=use_grad_checkpoint
        )

        # XAttnDecoder for feature-level fusion — only used in XAD variant (paper section 3.2)
        if fusion == 'xad':
            self.xattn_decoder = XAttnDecoder(embed_dim, num_heads, mlp_ratio)

        # Separate lightweight decoder per modality (paper section 3.2)
        self.sar_decoder = MAEDecoder(
            embed_dim, decoder_dim, decoder_depth,
            num_heads, mlp_ratio, patch_size, sar_channels
        )
        self.opt_decoder = MAEDecoder(
            embed_dim, decoder_dim, decoder_depth,
            num_heads, mlp_ratio, patch_size, opt_channels
        )

        # Layer norms applied to modality-biased latents before decoding
        self.sar_norm = nn.LayerNorm(embed_dim)
        self.opt_norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def random_masking(self, x, mask_ratio):
        B, N, D  = x.shape
        N_keep   = int(N * (1 - mask_ratio))

        noise       = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep  = ids_shuffle[:, :N_keep]
        x_visible = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :N_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    def patchify(self, imgs, channels):
        P = self.patch_size
        H = W = imgs.shape[2] // P
        x = imgs.reshape(imgs.shape[0], channels, H, P, W, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(imgs.shape[0], H * W, P * P * channels)
        return x

    def compute_loss(self, imgs, pred, mask, channels):
        target = self.patchify(imgs, channels)

        # Per-patch normalisation — normalise each patch to zero mean unit variance
        # This is standard in MAE and forces the model to learn structure not brightness
        # Without this the model just predicts the mean pixel value of each patch
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        # Loss computed only on masked patches — not visible ones
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, sar, optical):

        # Step 1 — modality-specific patch embedding
        sar_tokens = self.sar_embed(sar)
        opt_tokens = self.opt_embed(optical)

        # Step 2 — masking
        if self.masking == 'consistent':
            # Same patch positions masked in both modalities
            # Paper hypothesis: easier for cross-attention to find correspondences
            noise       = torch.rand(sar_tokens.shape[0], self.num_patches,
                                     device=sar_tokens.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            N_keep      = int(self.num_patches * (1 - self.mask_ratio))
            ids_keep    = ids_shuffle[:, :N_keep]

            B, N, D = sar_tokens.shape
            sar_vis = torch.gather(
                sar_tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )
            opt_vis = torch.gather(
                opt_tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )

            mask = torch.ones(B, N, device=sar_tokens.device)
            mask[:, :N_keep] = 0
            sar_mask = opt_mask = torch.gather(mask, 1, ids_restore)
            sar_restore = opt_restore = ids_restore

        else:
            # Independent masking — different random positions per modality
            sar_vis, sar_mask, sar_restore = self.random_masking(
                sar_tokens, self.mask_ratio
            )
            opt_vis, opt_mask, opt_restore = self.random_masking(
                opt_tokens, self.mask_ratio
            )

        # Step 3 — XAttnEncoder: cross-attention between visible tokens of both modalities
        fused = self.xattn_encoder(sar_vis, opt_vis)

        # Step 4 — append CLS token and run through remaining 11 transformer blocks
        B     = fused.shape[0]
        cls   = self.cls_token.expand(B, -1, -1)
        fused = torch.cat([cls, fused], dim=1)
        fused = self.encoder(fused)

        # Step 5 — split encoder output back into modality-biased latents
        # CLS token at position 0 is discarded for reconstruction
        # First N_vis tokens are SAR-biased, next N_vis are optical-biased
        fused_no_cls = fused[:, 1:, :]
        N_vis        = sar_vis.shape[1]
        z_sar        = self.sar_norm(fused_no_cls[:, :N_vis, :])
        z_opt        = self.opt_norm(fused_no_cls[:, N_vis:, :])

        # Step 6 — XAttnDecoder for feature-level fusion (XAD variant only)
        # paper equation 10, 11
        if self.fusion == 'xad':
            z_sar, z_opt = self.xattn_decoder(z_sar, z_opt)

        # Step 7 — modality-specific MAE decoders reconstruct all patches
        pred_sar = self.sar_decoder(z_sar, sar_restore)
        pred_opt = self.opt_decoder(z_opt, opt_restore)

        # Step 8 — MSE loss on masked patches only with per-patch normalisation
        loss_sar = self.compute_loss(sar, pred_sar, sar_mask, sar.shape[1])
        loss_opt = self.compute_loss(optical, pred_opt, opt_mask, optical.shape[1])
        loss     = loss_sar + loss_opt

        return loss, pred_sar, pred_opt, sar_mask, opt_mask


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n── Device: {device} ───────────────────────────────────────────────")

    sar     = torch.randn(2, 6,  128, 128).to(device)
    optical = torch.randn(2, 11, 128, 128).to(device)

    print("\n── XAE variant (consistent masking) ─────────────────────────────")
    model_xae = FusMAE(
        fusion='xae', masking='consistent',
        use_grad_checkpoint=False
    ).to(device)
    loss, pred_sar, pred_opt, mask_sar, mask_opt = model_xae(sar, optical)
    print(f"  loss         : {loss.item():.6f}")
    print(f"  pred_sar     : {pred_sar.shape}")
    print(f"  pred_opt     : {pred_opt.shape}")
    print(f"  masked ratio : {mask_sar.mean():.2f}  (should be 0.75)")
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert pred_sar.shape == (2, 64, 16 * 16 * 6)
    assert pred_opt.shape == (2, 64, 16 * 16 * 11)

    print("\n── XAD variant (consistent masking) ─────────────────────────────")
    model_xad = FusMAE(
        fusion='xad', masking='consistent',
        use_grad_checkpoint=False
    ).to(device)
    loss_xad, _, _, _, _ = model_xad(sar, optical)
    print(f"  loss         : {loss_xad.item():.6f}")
    assert not torch.isnan(loss_xad)

    print("\n── Independent masking variant ───────────────────────────────────")
    model_ind = FusMAE(
        fusion='xae', masking='independent',
        use_grad_checkpoint=False
    ).to(device)
    loss_ind, _, _, _, _ = model_ind(sar, optical)
    print(f"  loss         : {loss_ind.item():.6f}")
    assert not torch.isnan(loss_ind)

    print("\n── Backward pass ─────────────────────────────────────────────────")
    loss.backward()
    print("  backward pass completed without error")

    print("\n── Parameter count ───────────────────────────────────────────────")
    total = sum(p.numel() for p in model_xae.parameters()) / 1e6
    print(f"  total params : {total:.1f}M")

    print("\n── GPU memory ────────────────────────────────────────────────────")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  allocated    : {mem:.2f} GB")

    