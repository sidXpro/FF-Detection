"""
Original Object Detection Framework - Training Pipeline

This module implements a complete training pipeline with:
1. Mixed precision training (torch.cuda.amp)
2. Distributed Data Parallel (DDP) for multi-GPU training
3. Learning rate scheduling (warmup + cosine annealing)
4. Gradient accumulation
5. Model checkpointing
6. Logging and visualization

Author: Clean-room implementation
License: Original work
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm

from model import build_model, ObjectDetector
from loss import DetectionLoss
from data import create_dataloader


class Trainer:
    """
    Complete training pipeline for object detection.

    Features:
    - Mixed precision training for faster computation
    - Distributed training across multiple GPUs
    - Learning rate warmup and cosine annealing
    - Gradient accumulation for larger effective batch sizes
    - Automatic checkpointing
    - Training metrics logging
    """

    def __init__(
        self,
        model: ObjectDetector,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_classes: int = 80,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        accumulation_steps: int = 1,
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            model: Object detection model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_classes: Number of object categories
            device: Device to train on ('cuda' or 'cpu')
            lr: Initial learning rate
            weight_decay: L2 regularization weight
            warmup_epochs: Number of warmup epochs
            accumulation_steps: Gradient accumulation steps
            checkpoint_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
            rank: Process rank for DDP
            world_size: Total number of processes for DDP
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Wrap model with DDP if using distributed training
        if world_size > 1:
            self.model = DDP(model, device_ids=[rank])

        # Loss function
        self.criterion = DetectionLoss(num_classes=num_classes)

        # Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler (will be set up in train())
        self.scheduler = None
        self.warmup_epochs = warmup_epochs

        # Mixed precision training
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train(self, num_epochs: int, validate_every: int = 1, save_every: int = 5) -> Dict:
        """
        Main training loop.

        Args:
            num_epochs: Total number of epochs to train
            validate_every: Run validation every N epochs
            save_every: Save checkpoint every N epochs

        Returns:
            Training history dictionary
        """
        if self.is_main_process:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Mixed precision: {self.use_amp}")
            print(f"World size: {self.world_size}")

        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * self.warmup_epochs
        self.scheduler = self._get_scheduler(total_steps, warmup_steps)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Log training metrics
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  BBox Loss: {train_metrics['bbox_loss']:.4f}")
                print(f"  Obj Loss: {train_metrics['obj_loss']:.4f}")
                print(f"  Cls Loss: {train_metrics['cls_loss']:.4f}")
                print(f"  LR: {train_metrics['lr']:.6f}")

                self.history['train_loss'].append(train_metrics['loss'])
                self.history['learning_rate'].append(train_metrics['lr'])

            # Validation
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self._validate()
                if self.is_main_process:
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")
                    self.history['val_loss'].append(val_metrics['loss'])

                    # Save best model
                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self._save_checkpoint('best_model.pt', val_metrics)

            # Save periodic checkpoint
            if self.is_main_process and (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', train_metrics)

        # Save final model
        if self.is_main_process:
            self._save_checkpoint('final_model.pt', train_metrics)
            print("\nTraining completed!")

        return self.history

    def _train_epoch(self) -> Dict:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0

        # Progress bar (only on main process)
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        else:
            pbar = self.train_loader

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = {
                'boxes': targets['boxes'],
                'labels': targets['labels'],
                'grid_targets': [t.to(self.device) for t in targets['grid_targets']]
            }

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Accumulate metrics
            epoch_loss += loss_dict['loss']
            epoch_bbox_loss += loss_dict['bbox_loss']
            epoch_obj_loss += loss_dict['obj_loss']
            epoch_cls_loss += loss_dict['cls_loss']

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

        # Average metrics over epoch
        num_batches = len(self.train_loader)
        metrics = {
            'loss': epoch_loss / num_batches,
            'bbox_loss': epoch_bbox_loss / num_batches,
            'obj_loss': epoch_obj_loss / num_batches,
            'cls_loss': epoch_cls_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

        return metrics

    @torch.no_grad()
    def _validate(self) -> Dict:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_bbox_loss = 0.0
        val_obj_loss = 0.0
        val_cls_loss = 0.0

        for images, targets in self.val_loader:
            # Move to device
            images = images.to(self.device)
            targets = {
                'boxes': targets['boxes'],
                'labels': targets['labels'],
                'grid_targets': [t.to(self.device) for t in targets['grid_targets']]
            }

            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)

            val_loss += loss_dict['loss']
            val_bbox_loss += loss_dict['bbox_loss']
            val_obj_loss += loss_dict['obj_loss']
            val_cls_loss += loss_dict['cls_loss']

        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'loss': val_loss / num_batches,
            'bbox_loss': val_bbox_loss / num_batches,
            'obj_loss': val_obj_loss / num_batches,
            'cls_loss': val_cls_loss / num_batches
        }

        return metrics

    def _get_scheduler(self, total_steps: int, warmup_steps: int):
        """
        Create learning rate scheduler with warmup and cosine annealing.

        Schedule:
        1. Linear warmup from 0 to base_lr over warmup_steps
        2. Cosine annealing from base_lr to 0 over remaining steps
        """
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _save_checkpoint(self, filename: str, metrics: Dict):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Current training metrics
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Get model state dict (unwrap DDP if necessary)
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'history': self.history,
            'best_loss': self.best_loss
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        model = self.model.module if self.world_size > 1 else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.best_loss = checkpoint['best_loss']

        print(f"Checkpoint loaded from epoch {self.current_epoch}")


def setup_distributed(rank: int, world_size: int):
    """
    Setup distributed training environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_distributed(rank: int, world_size: int, config: Dict):
    """
    Training function for distributed training (one per GPU).

    Args:
        rank: Process rank (GPU id)
        world_size: Total number of GPUs
        config: Training configuration dictionary
    """
    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Build model
    model = build_model(num_classes=config['num_classes'])

    # Create data loaders (with distributed sampler)
    train_loader = create_dataloader(
        data_root=config['data_root'],
        annotation_file=config['train_annotation'],
        batch_size=config['batch_size'] // world_size,
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        augment=True
    )

    val_loader = None
    if config.get('val_annotation'):
        val_loader = create_dataloader(
            data_root=config['data_root'],
            annotation_file=config['val_annotation'],
            batch_size=config['batch_size'] // world_size,
            num_workers=config['num_workers'],
            img_size=config['img_size'],
            augment=False
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        device=f'cuda:{rank}',
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        accumulation_steps=config['accumulation_steps'],
        checkpoint_dir=config['checkpoint_dir'],
        use_amp=config['use_amp'],
        rank=rank,
        world_size=world_size
    )

    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        validate_every=config['validate_every'],
        save_every=config['save_every']
    )

    # Cleanup
    cleanup_distributed()


def train_single_gpu(config: Dict):
    """
    Training function for single GPU/CPU.

    Args:
        config: Training configuration dictionary
    """
    # Build model
    model = build_model(num_classes=config['num_classes'])

    # Create data loaders
    train_loader = create_dataloader(
        data_root=config['data_root'],
        annotation_file=config['train_annotation'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        augment=True
    )

    val_loader = None
    if config.get('val_annotation'):
        val_loader = create_dataloader(
            data_root=config['data_root'],
            annotation_file=config['val_annotation'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size'],
            augment=False
        )

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        device=device,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        accumulation_steps=config['accumulation_steps'],
        checkpoint_dir=config['checkpoint_dir'],
        use_amp=config['use_amp']
    )

    # Train
    history = trainer.train(
        num_epochs=config['num_epochs'],
        validate_every=config['validate_every'],
        save_every=config['save_every']
    )

    return history


if __name__ == '__main__':
    # Example training configuration
    config = {
        'data_root': 'path/to/coco',
        'train_annotation': 'path/to/annotations/instances_train.json',
        'val_annotation': 'path/to/annotations/instances_val.json',
        'num_classes': 80,
        'batch_size': 16,
        'num_workers': 4,
        'img_size': 640,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'warmup_epochs': 3,
        'accumulation_steps': 1,
        'num_epochs': 100,
        'validate_every': 1,
        'save_every': 5,
        'checkpoint_dir': './checkpoints',
        'use_amp': True
    }

    print("Training Pipeline Example")
    print("=" * 50)
    print("\nFor single GPU training:")
    print("    python train.py --config config.json")
    print("\nFor multi-GPU training:")
    print("    python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.json")
