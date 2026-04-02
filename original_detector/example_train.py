#!/usr/bin/env python3
"""
Example Training Script for Original Object Detection Framework

This script demonstrates how to use the framework for training on COCO dataset.

Usage:
    Single GPU:
        python example_train.py --data_root /path/to/coco --batch_size 16

    Multi-GPU (4 GPUs):
        python -m torch.distributed.launch --nproc_per_node=4 example_train.py \\
            --data_root /path/to/coco --batch_size 64 --distributed

Author: Clean-room implementation
License: Original work
"""

import torch
import torch.multiprocessing as mp
import argparse
import json
from pathlib import Path

from model import build_model
from train import Trainer, train_distributed, train_single_gpu
from data import create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Object Detection Model')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing images')
    parser.add_argument('--train_annotation', type=str,
                       default='annotations/instances_train2017.json',
                       help='Path to training annotations (relative to data_root)')
    parser.add_argument('--val_annotation', type=str,
                       default='annotations/instances_val2017.json',
                       help='Path to validation annotations (relative to data_root)')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=80,
                       help='Number of object classes')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Number of warmup epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')

    # Validation parameters
    parser.add_argument('--validate_every', type=int, default=1,
                       help='Run validation every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')

    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--world_size', type=int, default=None,
                       help='Number of GPUs for distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Print configuration
    print("=" * 70)
    print("Original Object Detection Framework - Training")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Create configuration dictionary
    config = {
        'data_root': args.data_root,
        'train_annotation': str(Path(args.data_root) / args.train_annotation),
        'val_annotation': str(Path(args.data_root) / args.val_annotation),
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'img_size': args.img_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'accumulation_steps': args.accumulation_steps,
        'num_epochs': args.num_epochs,
        'validate_every': args.validate_every,
        'save_every': args.save_every,
        'checkpoint_dir': args.checkpoint_dir,
        'use_amp': args.use_amp
    }

    # Check if distributed training
    if args.distributed:
        # Get world size
        world_size = args.world_size or torch.cuda.device_count()
        print(f"Starting distributed training on {world_size} GPUs...")

        # Launch distributed training
        mp.spawn(
            train_distributed,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU/CPU training
        print("Starting single GPU training...")
        history = train_single_gpu(config)

        # Save training history
        history_path = Path(args.checkpoint_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
