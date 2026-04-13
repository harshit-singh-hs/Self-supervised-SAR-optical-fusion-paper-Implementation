import os
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset  import get_dataloaders
from src.finetune import FusMAEFinetune, compute_metrics


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0}
    n_batches   = len(loader)

    for batch in tqdm(loader, desc='Evaluating', ncols=80):
        sar     = batch['sar'].to(device)
        optical = batch['optical'].to(device)
        mask    = batch['mask'].to(device)

        pred    = model(sar, optical)
        metrics = compute_metrics(pred, mask)
        for k in all_metrics:
            all_metrics[k] += metrics[k]

    return {k: v / n_batches for k, v in all_metrics.items()}


@torch.no_grad()
def visualise_predictions(model, dataset, device, n_samples=4, save_dir='logs'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 4))
    titles    = ['RGB (optical)', 'SAR (gVV)', 'Ground truth', 'Prediction']

    for i in range(n_samples):
        sample  = dataset[i]
        sar     = sample['sar'].unsqueeze(0).to(device)
        optical = sample['optical'].unsqueeze(0).to(device)
        mask_gt = sample['mask'].squeeze().cpu().numpy()

        pred      = model(sar, optical)
        pred_mask = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy()

        # contrast-stretched RGB so images are not pitch black
        rgb = optical[0, [2, 1, 0], :, :].cpu().numpy()
        rgb = rgb.transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        sar_vv = sar[0, 0, :, :].cpu().numpy()

        for j, (img, title) in enumerate(zip(
            [rgb, sar_vv, mask_gt, pred_mask], titles
        )):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(img)
            elif j == 1:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img, cmap='Reds', vmin=0, vmax=1)
            ax.set_title(title if i == 0 else '')
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  visualisation saved → {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/finetune.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/finetune/best.pth')
    parser.add_argument('--split',      type=str, default='test',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice    : {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")

    loaders = get_dataloaders(
        data_root   = config['data']['root'],
        batch_size  = config['train']['batch_size'],
        num_workers = config['data']['num_workers'],
        pin_memory  = False,
    )

    # Load fine-tuned checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Build model structure — pretrained_checkpoint only used for architecture init
    model = FusMAEFinetune(
        pretrained_path = config['model']['pretrained_checkpoint'],
        config          = config,
        device          = device,
        freeze_encoder  = False,
    ).to(device)

    # Load fine-tuned weights on top
    model.load_state_dict(ckpt['model'])
    print(f"  loaded from epoch {ckpt['epoch']}  "
          f"(best IoU: {ckpt.get('best_iou', 'N/A')})")

    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate_full(model, loaders[args.split], device)

    print(f"\n── Results on {args.split} set ───────────────────────────────────")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    print(f"\nGenerating prediction visualisations...")
    save_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_dir  = os.path.join('logs', save_name)
    visualise_predictions(model, loaders['test'].dataset, device,
                          n_samples=4, save_dir=save_dir)
    print("\nEvaluation complete\n")


if __name__ == '__main__':
    main()