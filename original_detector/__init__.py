"""
Original Object Detection Framework

A complete, clean-room implementation of an object detection system from first principles.

This package provides all components needed for training and deploying an object detector:
- Model architecture (Backbone + Neck + Detection Head)
- Loss functions (CIoU, Focal Loss, BCE)
- Data pipeline (Dataset, Augmentations, Encoding)
- Training pipeline (DDP, Mixed Precision, Scheduling)
- Inference pipeline (NMS, Post-processing)

Legal Notice:
-------------
This implementation is completely original and does NOT use, copy, adapt, or reference
any code from AGPL-licensed repositories (including Ultralytics YOLO, MMDetection, etc.).
All code is designed from scratch based on research papers and first principles.

Author: Clean-room implementation
License: Original work
"""

__version__ = '1.0.0'
__author__ = 'Original Implementation'

from .model import ObjectDetector, build_model
from .loss import DetectionLoss, complete_iou_loss, focal_loss
from .data import COCODetectionDataset, create_dataloader, encode_targets
from .train import Trainer, train_single_gpu, train_distributed
from .inference import ObjectDetectionInference, non_maximum_suppression, visualize_detections

__all__ = [
    # Model
    'ObjectDetector',
    'build_model',

    # Loss
    'DetectionLoss',
    'complete_iou_loss',
    'focal_loss',

    # Data
    'COCODetectionDataset',
    'create_dataloader',
    'encode_targets',

    # Training
    'Trainer',
    'train_single_gpu',
    'train_distributed',

    # Inference
    'ObjectDetectionInference',
    'non_maximum_suppression',
    'visualize_detections',
]
