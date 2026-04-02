# Original Object Detection Framework

**A clean-room implementation of an object detection system built from first principles.**

## Legal Notice

This implementation is completely original and does NOT use, copy, adapt, or reference any code from:
- Ultralytics YOLO or any AGPL-licensed repositories
- MMDetection, Detectron2, or other detection frameworks

All code is designed from scratch based on research papers and first principles.

## Architecture Overview

### 1. Model Architecture
- **Backbone**: Custom CNN with residual connections for feature extraction
- **Neck**: Feature Pyramid Network (FPN) inspired design for multi-scale feature fusion
- **Head**: Anchor-free detection head using center-based prediction

### 2. Design Rationale

#### Why Anchor-Free?
- Simpler architecture with fewer hyperparameters
- No need to tune anchor sizes/ratios for different datasets
- Better suited for hardware acceleration (regular tensor ops)
- Avoids dynamic anchor generation during inference

#### Hardware-Friendly Design
- Fixed tensor shapes throughout the pipeline
- Minimal dynamic control flow
- Regular convolution operations suitable for FPGA/systolic arrays
- Efficient matrix operations for batch processing

### 3. Loss Function

The detection loss combines three components:

1. **Bounding Box Loss**: Complete IoU (CIoU) loss for better localization
2. **Objectness Loss**: Binary cross-entropy for center-ness score
3. **Classification Loss**: Focal loss to handle class imbalance

Mathematical formulation:
```
L_total = λ_box * L_CIoU + λ_obj * L_BCE + λ_cls * L_focal

Where:
- L_CIoU = 1 - IoU + ρ²(b, b_gt)/c² + αv
- L_BCE = -y*log(p) - (1-y)*log(1-p)
- L_focal = -(1-p_t)^γ * log(p_t)
```

### 4. Data Pipeline
- Custom PyTorch Dataset for COCO-format annotations
- Augmentations: random crop, flip, color jitter, mosaic
- Grid-based label encoding for anchor-free detection

### 5. Training Pipeline
- Mixed precision training with torch.cuda.amp
- Distributed Data Parallel (DDP) for multi-GPU training
- Custom training loop with gradient accumulation
- Learning rate warmup and cosine annealing

### 6. Inference Pipeline
- Efficient Non-Maximum Suppression (NMS)
- Confidence thresholding
- Multi-scale inference support

## Usage

See `train.py` for a complete training example.

```python
from original_detector.model import ObjectDetector
from original_detector.train import Trainer

# Create model
model = ObjectDetector(num_classes=80)

# Train
trainer = Trainer(model, train_dataset, val_dataset)
trainer.train(epochs=100)
```

## Tensor Shapes

Input: `[B, 3, H, W]` where B=batch, H=W=640 (default)

Backbone outputs:
- P3: `[B, 256, H/8, W/8]`
- P4: `[B, 512, H/16, W/16]`
- P5: `[B, 1024, H/32, W/32]`

Neck outputs (FPN):
- F3: `[B, 256, H/8, W/8]`
- F4: `[B, 256, H/16, W/16]`
- F5: `[B, 256, H/32, W/32]`

Detection head outputs per scale:
- Bbox: `[B, 4, H/s, W/s]` (x, y, w, h)
- Objectness: `[B, 1, H/s, W/s]`
- Classes: `[B, num_classes, H/s, W/s]`

## License

This implementation is provided as original work for educational and research purposes.
