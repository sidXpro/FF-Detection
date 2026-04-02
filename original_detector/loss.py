"""
Original Object Detection Framework - Loss Functions

This module implements custom loss functions for object detection:
1. Complete IoU (CIoU) Loss for bounding box regression
2. Binary Cross-Entropy for objectness prediction
3. Focal Loss for classification

Mathematical formulations are provided for each loss component.

Author: Clean-room implementation
License: Original work
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.

    Boxes format: [x_center, y_center, width, height]

    Args:
        box1: [N, 4] tensor
        box2: [M, 4] tensor
        eps: Small value to avoid division by zero

    Returns:
        IoU matrix [N, M]
    """
    # Convert from center format to corner format: (x1, y1, x2, y2)
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + eps

    # IoU
    iou = inter_area / union_area
    return iou


def complete_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) Loss for bounding box regression.

    CIoU considers:
    1. IoU: Overlap between boxes
    2. Distance: Distance between box centers
    3. Aspect ratio: Consistency of aspect ratios

    Mathematical formulation:
        L_CIoU = 1 - IoU + ρ²(b, b_gt) / c² + α * v

    Where:
        - ρ²: Euclidean distance between box centers
        - c²: Diagonal length of smallest enclosing box
        - v: Aspect ratio consistency
        - α: Trade-off parameter

    Args:
        pred_boxes: Predicted boxes [N, 4] in (x, y, w, h) format
        target_boxes: Ground truth boxes [N, 4] in (x, y, w, h) format
        eps: Small value for numerical stability

    Returns:
        CIoU loss value [N]
    """
    # Convert to corner format for IoU calculation
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps

    # IoU
    iou = inter_area / union_area

    # Smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    # Diagonal of enclosing box (c²)
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Distance between centers (ρ²)
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Aspect ratio consistency (v)
    arctan_pred = torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
    arctan_target = torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps))
    v = (4 / (math.pi ** 2)) * torch.pow(arctan_pred - arctan_target, 2)

    # Trade-off parameter (α)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU loss
    ciou_loss = 1 - iou + center_dist_sq / enclose_diag_sq + alpha * v

    return ciou_loss


def focal_loss(pred_logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Focal Loss for classification to handle class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.

    Mathematical formulation:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        - p_t: Model's estimated probability for the target class
        - α_t: Weighting factor (higher for rare classes)
        - γ: Focusing parameter (higher = more focus on hard examples)

    Args:
        pred_logits: Predicted logits [N, num_classes]
        targets: Ground truth class indices [N]
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        eps: Small value for numerical stability

    Returns:
        Focal loss value [N]
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=-1)

    # Get probability for target class
    target_probs = pred_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_probs = target_probs.clamp(min=eps, max=1-eps)

    # Focal loss
    focal_weight = (1 - target_probs) ** gamma
    ce_loss = -torch.log(target_probs)
    focal_loss_val = alpha * focal_weight * ce_loss

    return focal_loss_val


class DetectionLoss(nn.Module):
    """
    Combined detection loss for object detection training.

    Total loss = λ_box * L_box + λ_obj * L_obj + λ_cls * L_cls

    Components:
    1. Bounding Box Loss (CIoU): Measures localization accuracy
    2. Objectness Loss (BCE): Measures confidence in object presence
    3. Classification Loss (Focal): Measures class prediction accuracy

    The loss is computed only for positive samples (cells containing object centers).
    """
    def __init__(self, num_classes: int, lambda_box: float = 5.0,
                 lambda_obj: float = 1.0, lambda_cls: float = 1.0):
        """
        Args:
            num_classes: Number of object categories
            lambda_box: Weight for bounding box loss
            lambda_obj: Weight for objectness loss
            lambda_cls: Weight for classification loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, predictions: List[Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total detection loss.

        Args:
            predictions: List of prediction dicts from model (one per scale)
                Each dict contains:
                    - 'bbox': [B, 4, H, W]
                    - 'obj': [B, 1, H, W]
                    - 'cls': [B, num_classes, H, W]
            targets: Dict with:
                - 'boxes': List of [N_i, 4] ground truth boxes per image
                - 'labels': List of [N_i] class labels per image
                - 'grid_targets': Pre-encoded grid targets for each scale

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        device = predictions[0]['bbox'].device
        batch_size = predictions[0]['bbox'].shape[0]

        total_bbox_loss = torch.tensor(0., device=device)
        total_obj_loss = torch.tensor(0., device=device)
        total_cls_loss = torch.tensor(0., device=device)
        num_positive = 0

        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            pred_bbox = pred['bbox']  # [B, 4, H, W]
            pred_obj = pred['obj']    # [B, 1, H, W]
            pred_cls = pred['cls']    # [B, num_classes, H, W]

            # Get target for this scale
            target_grid = targets['grid_targets'][scale_idx]  # [B, H, W, 6+num_classes]

            # Reshape predictions
            B, _, H, W = pred_bbox.shape
            pred_bbox = pred_bbox.permute(0, 2, 3, 1).reshape(B, H, W, 4)
            pred_obj = pred_obj.permute(0, 2, 3, 1).reshape(B, H, W, 1)
            pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(B, H, W, self.num_classes)

            # Extract targets
            obj_mask = target_grid[..., 0:1]  # [B, H, W, 1]
            target_bbox = target_grid[..., 1:5]  # [B, H, W, 4]
            target_cls = target_grid[..., 5:5+self.num_classes]  # [B, H, W, num_classes]

            # Find positive samples (cells with objects)
            pos_mask = obj_mask.squeeze(-1) > 0.5  # [B, H, W]

            if pos_mask.sum() > 0:
                # Bounding box loss (only on positive samples)
                pred_bbox_pos = pred_bbox[pos_mask]  # [N_pos, 4]
                target_bbox_pos = target_bbox[pos_mask]  # [N_pos, 4]
                bbox_loss = complete_iou_loss(pred_bbox_pos, target_bbox_pos).mean()
                total_bbox_loss += bbox_loss

                # Classification loss (only on positive samples)
                pred_cls_pos = pred_cls[pos_mask]  # [N_pos, num_classes]
                target_cls_pos = target_cls[pos_mask].argmax(dim=-1)  # [N_pos]
                cls_loss = focal_loss(pred_cls_pos, target_cls_pos).mean()
                total_cls_loss += cls_loss

                num_positive += pos_mask.sum().item()

            # Objectness loss (all samples)
            obj_loss = F.binary_cross_entropy(pred_obj.squeeze(-1), obj_mask.squeeze(-1), reduction='mean')
            total_obj_loss += obj_loss

        # Average losses across scales
        num_scales = len(predictions)
        total_bbox_loss = total_bbox_loss / num_scales
        total_obj_loss = total_obj_loss / num_scales
        total_cls_loss = total_cls_loss / num_scales

        # Combine losses
        total_loss = (self.lambda_box * total_bbox_loss +
                     self.lambda_obj * total_obj_loss +
                     self.lambda_cls * total_cls_loss)

        # Loss dictionary for logging
        loss_dict = {
            'loss': total_loss.item(),
            'bbox_loss': total_bbox_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'cls_loss': total_cls_loss.item(),
            'num_positive': num_positive
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")

    # Test CIoU loss
    pred_boxes = torch.tensor([[100, 100, 50, 50],
                               [200, 200, 60, 60]], dtype=torch.float32)
    target_boxes = torch.tensor([[105, 105, 48, 48],
                                 [195, 195, 65, 65]], dtype=torch.float32)

    ciou = complete_iou_loss(pred_boxes, target_boxes)
    print(f"\nCIoU Loss: {ciou}")

    # Test focal loss
    pred_logits = torch.randn(10, 80)  # 10 samples, 80 classes
    targets = torch.randint(0, 80, (10,))

    fl = focal_loss(pred_logits, targets)
    print(f"Focal Loss: {fl.mean()}")

    # Test complete detection loss
    loss_fn = DetectionLoss(num_classes=80)

    # Create dummy predictions
    predictions = []
    for scale in [80, 40, 20]:
        pred = {
            'bbox': torch.randn(2, 4, scale, scale),
            'obj': torch.rand(2, 1, scale, scale),
            'cls': torch.randn(2, 80, scale, scale)
        }
        predictions.append(pred)

    # Create dummy targets
    targets = {
        'boxes': [torch.rand(5, 4) * 640 for _ in range(2)],
        'labels': [torch.randint(0, 80, (5,)) for _ in range(2)],
        'grid_targets': [torch.rand(2, scale, scale, 86) for scale in [80, 40, 20]]
    }

    total_loss, loss_dict = loss_fn(predictions, targets)
    print(f"\nDetection Loss Test:")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss Components: {loss_dict}")
