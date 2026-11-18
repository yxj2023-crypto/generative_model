"""Train the local MeanFlow implementation; defaults match MeanFlow/configs/cifar10.py."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list, set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

from src import DiT_meanflow, MeanFlow, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['cifar10', 'mnist', 'fashion_mnist'], default='cifar10')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--result-dir', type=Path, default=None, help='Parent directory of timestamped runs')
    parser.add_argument('--train-steps', type=int, default=200_000, help='Total optimizer updates per process')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU/process')
    parser.add_argument('--num-workers', type=int, default=8, help='Data workers per process')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--mlp-scale', type=float, default=4.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--warmup-steps', type=int, default=2500)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--mixed-precision', choices=['bf16', 'no'], default='bf16')
    parser.add_argument('--no-cfg', action='store_false', dest='cfg', help='Disable label/CFG conditioning')
    parser.set_defaults(cfg=True)
    parser.add_argument('--max-cfg', type=float, default=7.5)
    parser.add_argument('--cfg-ratio-sample', type=float, default=0.75)
    parser.add_argument('--cfg-ratio-drop', type=float, default=0.10)
    parser.add_argument('--time-mean', type=float, default=-0.4)
    parser.add_argument('--time-std', type=float, default=1.0)
    parser.add_argument('--adaptive-l2-gamma', type=float, default=0.0)
    parser.add_argument('--adaptive-l2-c', type=float, default=1e-2)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--sample-every', type=int, default=500, help='0 disables image generation')
    parser.add_argument('--save-every', type=int, default=0, help='Checkpoint interval; 0 saves only at the end')
    parser.add_argument('--steps', type=int, default=5, help='Integration steps for generated images')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--cfg-scale', type=float, default=None, help='Sampling guidance; defaults to (1+max_cfg)/2')
    parser.add_argument('--limit', type=int, default=None, help='Dataset subset for smoke runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    for name in ('train_steps', 'batch_size', 'hidden_dim', 'depth', 'num_heads', 'patch_size',
                 'steps', 'num_samples', 'log_every'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.hidden_dim % args.num_heads or args.hidden_dim % 4:
        parser.error('--hidden-dim must be divisible by --num-heads and by 4')
    if min(args.num_workers, args.warmup_steps, args.sample_every, args.save_every) < 0:
        parser.error('worker count and step intervals must be nonnegative')
    if args.limit is not None and args.limit < args.batch_size:
        parser.error('--limit must be at least --batch-size (drop_last=True)')
    if not 0 <= args.min_lr_ratio <= 1 or args.max_cfg < 1:
        parser.error('--min-lr-ratio must be in [0, 1] and --max-cfg must be >= 1')
    if not all(0 <= ratio <= 1 for ratio in (args.cfg_ratio_sample, args.cfg_ratio_drop)):
        parser.error('CFG sample/drop ratios must be in [0, 1]')
    if args.adaptive_l2_c <= 0 or args.time_std < 0 or args.grad_clip <= 0:
        parser.error('adaptive-l2-c and grad-clip must be positive; time-std must be nonnegative')
    if args.cfg_scale is None:
        args.cfg_scale = (1 + args.max_cfg) / 2
    return args


def lr_multiplier(step, warmup_steps, total_steps, min_lr_ratio):
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    if step >= total_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=args.cpu,
        # One scheduler advance per optimizer update, independent of world size.
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    set_seed(args.seed)
    image_size = 32 if args.dataset == 'cifar10' else 28
    channels = 3 if args.dataset == 'cifar10' else 1
    if image_size % args.patch_size:
        raise ValueError('The dataset image size must be divisible by patch_size')
    # Serialize downloads so ranks do not write the same archive concurrently.
    with accelerator.main_process_first():
        dataset, num_classes = build_dataset(args.dataset, args.data_root)
    if args.dataset == 'cifar10':
        dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=accelerator.device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    model = DiT_meanflow(
        image_size=image_size, patch_size=args.patch_size, in_channel=channels,
        hidden_dim=args.hidden_dim, depth=args.depth, num_head=args.num_heads,
        mlp_scale=args.mlp_scale, num_classes=num_classes if args.cfg else None, max_cfg=args.max_cfg,
    ).to(accelerator.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lambda step: lr_multiplier(
        step, args.warmup_steps, args.train_steps, args.min_lr_ratio))
    raw_model = model
    model, loader, optimizer, scheduler = accelerator.prepare(model, loader, optimizer, scheduler)
    trainer = MeanFlow(
        model, loader, optimizer, scheduler, num_classes=num_classes,
        image_size=(image_size, image_size), use_cfg=args.cfg, device=accelerator.device,
        raw_model=raw_model,
        adaptive_l2_gamma=args.adaptive_l2_gamma, adaptive_l2_c=args.adaptive_l2_c,
    )
    trainer.time_mean, trainer.time_std = args.time_mean, args.time_std
    trainer.cfg_scale_range = [1.0, args.max_cfg]
    trainer.cfg_ratio_sample, trainer.cfg_ratio_drop = args.cfg_ratio_sample, args.cfg_ratio_drop
    if len(loader) == 0:
        raise ValueError('No complete training batches; increase dataset size or reduce batch_size')
    # Keep initial weights identical, then draw different noise/CFG masks on each rank.
    set_seed(args.seed, device_specific=True)
    run_path = [None]
    if accelerator.is_main_process:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        parent = args.result_dir or ROOT / 'results' / args.dataset / 'meanflow'
        run_path[0] = str(parent / timestamp)
    broadcast_object_list(run_path)
    output = Path(run_path[0])
    config = dict(vars(args), image_size=image_size, channels=channels, num_classes=num_classes,
                  world_size=accelerator.num_processes,
                  global_batch_size=args.batch_size * accelerator.num_processes,
                  mode='i-meanflow', horizontal_flip=args.dataset == 'cifar10')
    if accelerator.is_main_process:
        (output / 'samples').mkdir(parents=True)
        (output / 'checkpoints').mkdir()
        (output / 'config.json').write_text(json.dumps(config, default=str, indent=2), encoding='utf-8')
    accelerator.wait_for_everyone()
    accelerator.print(f'Dataset: {args.dataset}, world_size: {accelerator.num_processes}, '
                      f'batch/GPU: {args.batch_size}, global batch: {config["global_batch_size"]}, '
                      f'precision: {accelerator.mixed_precision}, updates: {args.train_steps}\nOutput: {output}')
    log_totals = torch.zeros(2, device=accelerator.device, dtype=torch.float64)

    def on_step(step, loss):
        log_totals[0] += loss.double()
        log_totals[1] += 1
        final = step == args.train_steps
        if step % args.log_every == 0 or final:
            totals = accelerator.reduce(log_totals, reduction='sum')
            record = dict(step=step, loss=(totals[0] / totals[1]).item(),
                          lr=optimizer.param_groups[0]['lr'])
            if accelerator.is_main_process:
                with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
            accelerator.print(f'Step {step}/{args.train_steps}: loss={record["loss"]:.6f}, '
                              f'lr={record["lr"]:.8g}')
            log_totals.zero_()
        sample = args.sample_every > 0 and (step % args.sample_every == 0 or final)
        checkpoint = final or (args.save_every > 0 and step % args.save_every == 0)
        if accelerator.is_main_process:
            if sample:
                # Sampling must not consume the main rank's training RNG sequence.
                devices = [accelerator.device] if accelerator.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices), accelerator.autocast():
                    labels = torch.arange(args.num_samples, device=accelerator.device) % num_classes
                    samples = trainer.generate(args.steps, (args.num_samples, channels, image_size, image_size),
                                               y=labels if args.cfg else None, cfg_scale=args.cfg_scale)
                    save_image(samples.float(), output / 'samples' / f'step_{step:07d}.png', nrow=10)
            if checkpoint:
                accelerator.save({
                    'model': accelerator.get_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'global_step': step, 'config': config,
                }, output / 'checkpoints' / f'step_{step:07d}.pth')
        if sample or checkpoint:
            accelerator.wait_for_everyone()
            model.train()

    epoch = 0
    global_step = 0
    while global_step < args.train_steps:
        loader.set_epoch(epoch)
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # prepare() autocasts each forward; no outer autocast around the
            # no-grad CFG predictions and gradient-carrying forward together.
            loss = trainer.training_loss(*batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            on_step(global_step, loss.detach())
            if global_step >= args.train_steps:
                break
        epoch += 1
    accelerator.print(f'Finished: {output.resolve()}')
    accelerator.end_training()


if __name__ == '__main__':
    main()
