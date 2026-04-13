import os
import sys
import math
import time
import yaml
import argparse
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset  import get_dataloaders
from src.fus_mae  import FusMAE


def get_lr(epoch, config):
    warmup = config['train']['warmup_epochs']
    epochs = config['train']['epochs']
    base   = config['train']['base_lr']
    min_lr = config['train']['min_lr']

    if epoch < warmup:
        return base * (epoch + 1) / warmup

    progress = (epoch - warmup) / max(1, epochs - warmup)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base - min_lr) * cosine


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt        = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt['epoch'] + 1
    best_loss   = ckpt.get('best_loss', float('inf'))
    print(f"  resumed from epoch {ckpt['epoch']}  (best loss: {best_loss:.4f})")
    return start_epoch, best_loss


def train_one_epoch(model, loader, optimizer, scaler, device, config, epoch):
    model.train()
    fp16      = config['train']['fp16']
    clip_grad = config['train']['clip_grad']
    log_every = config['logging']['log_every']
    n_batches = len(loader)
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, ncols=90)

    for step, batch in enumerate(pbar):
        sar     = batch['sar'].to(device, non_blocking=True)
        optical = batch['optical'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=fp16):
            loss, _, _, _, _ = model(sar, optical)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        loss_val    = loss.item()
        total_loss += loss_val
        pbar.set_postfix({'loss': f'{loss_val:.4f}'})

        if step % log_every == 0:
            global_step = epoch * n_batches + step
            wandb.log({
                'train/loss_step': loss_val,
                'train/lr':        optimizer.param_groups[0]['lr'],
                'epoch':           epoch,
            }, step=global_step)

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    fp16       = config['train']['fp16']
    total_loss = 0.0

    for batch in loader:
        sar     = batch['sar'].to(device, non_blocking=True)
        optical = batch['optical'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=fp16):
            loss, _, _, _, _ = model(sar, optical)

        total_loss += loss.item()

    return total_loss / len(loader)


def build_model(config, device):
    m     = config['model']
    model = FusMAE(
        img_size            = m['img_size'],
        patch_size          = m['patch_size'],
        sar_channels        = m['sar_channels'],
        opt_channels        = m['opt_channels'],
        embed_dim           = m['embed_dim'],
        encoder_depth       = m['encoder_depth'],
        decoder_dim         = m.get('decoder_dim', 192),
        decoder_depth       = m['decoder_depth'],
        num_heads           = m['num_heads'],
        mlp_ratio           = m['mlp_ratio'],
        mask_ratio          = m['mask_ratio'],
        masking             = m['masking'],
        fusion              = m['fusion'],
        use_grad_checkpoint = config['train']['grad_checkpoint'],
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug',  action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nBuilding dataloaders...")
    loaders = get_dataloaders(
        data_root   = config['data']['root'],
        batch_size  = config['train']['batch_size'],
        num_workers = config['data']['num_workers'],
        pin_memory  = config['data']['pin_memory'],
    )

    print("\nBuilding model...")
    model = build_model(config, device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters : {total:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config['train']['base_lr'],
        weight_decay = config['train']['weight_decay'],
        betas        = (0.9, 0.95),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config['train']['fp16'])

    start_epoch = 0
    best_loss   = float('inf')

    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )

    if args.debug:
        print("\nDEBUG MODE — running 3 batches then exiting")
        model.train()
        fp16        = config['train']['fp16']
        loader_iter = iter(loaders['train'])
        for i in range(3):
            batch   = next(loader_iter)
            sar     = batch['sar'].to(device)
            optical = batch['optical'].to(device)

            with torch.amp.autocast('cuda', enabled=fp16):
                loss, _, _, _, _ = model(sar, optical)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"  batch {i+1}/3  loss: {loss.item():.4f}")

        print("\nDebug run successful — training loop works correctly\n")
        return

    wandb.init(
        project = config['logging']['project'],
        name    = config['logging']['run_name'],
        config  = config,
    )

    print(f"\nStarting pretraining — {config['train']['epochs']} epochs\n")
    ckpt_dir = 'checkpoints/pretrain'

    for epoch in range(start_epoch, config['train']['epochs']):
        t0 = time.time()

        lr = get_lr(epoch, config)
        for g in optimizer.param_groups:
            g['lr'] = lr

        train_loss = train_one_epoch(
            model, loaders['train'], optimizer, scaler, device, config, epoch
        )
        val_loss = validate(model, loaders['val'], device, config)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:03d}  "
              f"train: {train_loss:.4f}  "
              f"val: {val_loss:.4f}  "
              f"lr: {lr:.2e}  "
              f"time: {elapsed:.1f}s")

        wandb.log({
            'train/loss_epoch': train_loss,
            'val/loss':         val_loss,
            'lr':               lr,
            'epoch':            epoch,
        })

        if (epoch + 1) % config['train']['save_every'] == 0:
            save_checkpoint({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler':    scaler.state_dict(),
                'config':    config,
                'best_loss': best_loss,
            }, path=os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth'))

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler':    scaler.state_dict(),
                'config':    config,
                'best_loss': best_loss,
            }, path=os.path.join(ckpt_dir, 'best.pth'))

    wandb.finish()
    print(f"\nPretraining complete. Best val loss: {best_loss:.4f}\n")


if __name__ == '__main__':
    main()