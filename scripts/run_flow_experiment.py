import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src import Unet,FlowTrainer,build_dataloader

def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mnist', 'fashion_mnist', 'cifar10'], default='mnist')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--result-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-adjust-epoch', type=int, default=25)
    parser.add_argument('--eval-per-epoch', type=int, default=2)
    parser.add_argument('--save-per-epoch', type=int, default=10)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--cfg', action='store_true')
    parser.add_argument('--cfg-scale', type=float, default=5.0)
    parser.add_argument('--limit', type=int, default=None, help='Use a subset for smoke runs')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    loader, num_classes = build_dataloader(args.dataset, args.data_root, args.batch_size,
                              num_workers=args.num_workers, limit=args.limit)
    channels = 3 if args.dataset == 'cifar10' else 1
    model = Unet(input_channel=channels, base_channel=args.base_channels,
                 embed_dim=args.base_channels).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=args.lr_adjust_epoch, gamma=0.1)
    trainer = FlowTrainer(model, loader, optimizer, scheduler,
                          num_epochs=args.epochs, eval_per_epoch=args.eval_per_epoch,
                          save_per_epoch=args.save_per_epoch, use_cfg=args.cfg,
                          device=args.device, num_classes=num_classes)
    output_root = args.result_dir or Path(__file__).resolve().parents[1] / 'results' / args.dataset / ('cfg' if args.cfg else 'uncond')
    output = output_root / timestamp()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    print(f'Dataset: {args.dataset}, samples: {len(loader.dataset)}, device: {args.device}', flush=True)
    trainer.train(output, generate_num_step=args.steps, num_samples=args.num_samples,
                  cfg_scale=args.cfg_scale)
    print(f'Finished: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
