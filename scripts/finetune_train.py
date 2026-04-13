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
from src.dataset   import get_dataloaders
from src.finetune  import FusMAEFinetune, seg_loss, compute_metrics


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


def train_one_epoch(model, loader, optimizer, scaler, device, config, epoch):
    model.train()
    fp16       = config['train']['fp16']
    clip_grad  = config['train']['clip_grad']
    log_every  = config['logging']['log_every']
    bce_w      = config['loss']['bce_weight']
    dice_w     = config['loss']['dice_weight']
    total_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, ncols=90)

    for step, batch in enumerate(pbar):
        sar     = batch['sar'].to(device, non_blocking=True)
        optical = batch['optical'].to(device, non_blocking=True)
        mask    = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=fp16):
            pred = model(sar, optical)
            loss = seg_loss(pred, mask, bce_w, dice_w)

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
            wandb.log({'finetune/loss_step': loss_val,
                       'finetune/lr': optimizer.param_groups[0]['lr'],
                       'epoch': epoch}, step=global_step)

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    fp16        = config['train']['fp16']
    bce_w       = config['loss']['bce_weight']
    dice_w      = config['loss']['dice_weight']
    total_loss  = 0.0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0}
    n_batches   = len(loader)

    for batch in loader:
        sar     = batch['sar'].to(device, non_blocking=True)
        optical = batch['optical'].to(device, non_blocking=True)
        mask    = batch['mask'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=fp16):
            pred = model(sar, optical)
            loss = seg_loss(pred, mask, bce_w, dice_w)

        total_loss += loss.item()
        metrics = compute_metrics(pred, mask)
        for k in all_metrics:
            all_metrics[k] += metrics[k]

    avg_loss    = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--debug',  action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\nBuilding dataloaders...")
    loaders = get_dataloaders(
        data_root   = config['data']['root'],
        batch_size  = config['train']['batch_size'],
        num_workers = config['data']['num_workers'],
        pin_memory  = config['data']['pin_memory'],
    )

    print("\nLoading pretrained model...")
    model = FusMAEFinetune(
        pretrained_path = config['model']['pretrained_checkpoint'],
        config          = config,
        device          = device,
        freeze_encoder  = config['head']['freeze_encoder'],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable params : {trainable:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config['train']['base_lr'],
        weight_decay = config['train']['weight_decay'],
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config['train']['fp16'])

    if args.debug:
        print("\nDEBUG MODE — running 3 batches then exiting")
        model.train()
        fp16        = config['train']['fp16']
        loader_iter = iter(loaders['train'])
        for i in range(3):
            batch   = next(loader_iter)
            sar     = batch['sar'].to(device)
            optical = batch['optical'].to(device)
            mask    = batch['mask'].to(device)
            with torch.amp.autocast('cuda', enabled=fp16):
                pred = model(sar, optical)
                loss = seg_loss(pred, mask,
                                config['loss']['bce_weight'],
                                config['loss']['dice_weight'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"  batch {i+1}/3  loss: {loss.item():.4f}")
        print("\nDebug run successful\n")
        return

    wandb.init(
        project = config['logging']['project'],
        name    = config['logging']['run_name'],
        config  = config,
    )

    print(f"\nStarting fine-tuning — {config['train']['epochs']} epochs\n")
    ckpt_dir = 'checkpoints/finetune'
    best_iou = 0.0

    for epoch in range(config['train']['epochs']):
        t0 = time.time()

        lr = get_lr(epoch, config)
        for g in optimizer.param_groups:
            g['lr'] = lr

        train_loss            = train_one_epoch(
            model, loaders['train'], optimizer, scaler, device, config, epoch
        )
        val_loss, val_metrics = validate(model, loaders['val'], device, config)
        elapsed               = time.time() - t0

        print(f"Epoch {epoch:03d}  "
              f"train: {train_loss:.4f}  "
              f"val: {val_loss:.4f}  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}  "
              f"lr: {lr:.2e}  "
              f"time: {elapsed:.1f}s")

        wandb.log({
            'finetune/train_loss': train_loss,
            'finetune/val_loss':   val_loss,
            'finetune/iou':        val_metrics['iou'],
            'finetune/f1':         val_metrics['f1'],
            'finetune/precision':  val_metrics['precision'],
            'finetune/recall':     val_metrics['recall'],
            'lr':                  lr,
            'epoch':               epoch,
        })

        if (epoch + 1) % config['train']['save_every'] == 0:
            save_checkpoint({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler':    scaler.state_dict(),
                'config':    config,
                'best_iou':  best_iou,
            }, path=os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth'))

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            save_checkpoint({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler':    scaler.state_dict(),
                'config':    config,
                'best_iou':  best_iou,
            }, path=os.path.join(ckpt_dir, 'best.pth'))

    wandb.finish()
    print(f"\nFine-tuning complete. Best val IoU: {best_iou:.4f}\n")


if __name__ == '__main__':
    main()