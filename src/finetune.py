import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patch_embed import PatchEmbed
from src.cross_attn  import XAttnEncoder
from src.encoder     import ViTEncoder


class SegmentationHead(nn.Module):

    def __init__(self, embed_dim=384, patch_size=16, img_size=128):
        super().__init__()
        self.patch_size  = patch_size
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, patch_size * patch_size),
        )

    def forward(self, tokens):
        B           = tokens.shape[0]
        patch_preds = self.head(tokens)
        patch_preds = patch_preds.reshape(B, self.grid_size, self.grid_size,
                                          self.patch_size, self.patch_size)
        patch_preds = patch_preds.permute(0, 1, 3, 2, 4)
        mask_pred   = patch_preds.reshape(B, 1,
                                          self.grid_size * self.patch_size,
                                          self.grid_size * self.patch_size)
        return mask_pred


class FusMAEFinetune(nn.Module):

    def __init__(self, pretrained_path, config, device, freeze_encoder=True):
        super().__init__()
        m = config['model']

        self.sar_embed     = PatchEmbed(m['sar_channels'], m['patch_size'],
                                        m['embed_dim'], m['img_size'])
        self.opt_embed     = PatchEmbed(m['opt_channels'], m['patch_size'],
                                        m['embed_dim'], m['img_size'])
        self.xattn_encoder = XAttnEncoder(m['embed_dim'], m['num_heads'], m['mlp_ratio'])
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, m['embed_dim']))
        self.encoder       = ViTEncoder(m['embed_dim'], m['encoder_depth'],
                                        m['num_heads'], m['mlp_ratio'],
                                        use_grad_checkpoint=False)
        self.seg_head      = SegmentationHead(m['embed_dim'], m['patch_size'], m['img_size'])

        self._load_pretrained(pretrained_path, device)

        if freeze_encoder:
            for p in self.sar_embed.parameters():     p.requires_grad = False
            for p in self.opt_embed.parameters():     p.requires_grad = False
            for p in self.xattn_encoder.parameters(): p.requires_grad = False
            for p in self.encoder.parameters():       p.requires_grad = False
            self.cls_token.requires_grad = False

    def _load_pretrained(self, path, device):
        ckpt      = torch.load(path, map_location=device)
        state     = ckpt['model']
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"  pretrained weights loaded: {loaded} tensors  skipped: {skipped}")

    def forward(self, sar, optical):
        sar_tokens = self.sar_embed(sar)
        opt_tokens = self.opt_embed(optical)

        fused = self.xattn_encoder(sar_tokens, opt_tokens)

        B     = fused.shape[0]
        cls   = self.cls_token.expand(B, -1, -1)
        fused = torch.cat([cls, fused], dim=1)
        fused = self.encoder(fused)

        tokens_no_cls = fused[:, 1:, :]
        N             = tokens_no_cls.shape[1]
        half          = N // 2
        z_fused       = (tokens_no_cls[:, :half, :] + tokens_no_cls[:, half:, :]) / 2.0

        mask_pred = self.seg_head(z_fused)
        return mask_pred


def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def seg_loss(pred, target, bce_w=0.5, dice_w=0.5):
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + dice_w * dice


def compute_metrics(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.float()

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {'precision': precision, 'recall': recall,
            'f1': f1, 'iou': iou, 'accuracy': accuracy}


if __name__ == '__main__':
    import yaml

    with open('configs/finetune.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = config['model']['pretrained_checkpoint']

    print(f"\n── Loading pretrained weights from: {ckpt} ──────────────────────")
    model = FusMAEFinetune(
        pretrained_path = ckpt,
        config          = config,
        device          = device,
        freeze_encoder  = config['head']['freeze_encoder'],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  total params     : {total:.1f}M")
    print(f"  trainable params : {trainable:.2f}M  (seg head only)")

    print("\n── Forward pass ──────────────────────────────────────────────────")
    sar     = torch.randn(2, 6,  128, 128).to(device)
    optical = torch.randn(2, 11, 128, 128).to(device)
    mask_gt = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)

    pred = model(sar, optical)
    print(f"  input sar    : {sar.shape}")
    print(f"  input optical: {optical.shape}")
    print(f"  pred mask    : {pred.shape}")
    assert pred.shape == (2, 1, 128, 128)

    print("\n── Loss computation ──────────────────────────────────────────────")
    loss = seg_loss(pred, mask_gt,
                    bce_w  = config['loss']['bce_weight'],
                    dice_w = config['loss']['dice_weight'])
    print(f"  seg loss : {loss.item():.4f}")
    assert not torch.isnan(loss)

    print("\n── Metrics ───────────────────────────────────────────────────────")
    m = compute_metrics(pred, mask_gt)
    print(f"  IoU: {m['iou']:.4f}  F1: {m['f1']:.4f}  "
          f"Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}")

    print("\n── Backward pass (seg head only) ─────────────────────────────────")
    loss.backward()
    grads         = [p.grad for p in model.seg_head.parameters() if p.grad is not None]
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    print(f"  seg head gradients: {len(grads)} tensors have gradients")
    print(f"  encoder gradients : {len(encoder_grads)}  (should be 0 — frozen)")
    assert len(encoder_grads) == 0

    