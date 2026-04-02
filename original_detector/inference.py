"""
Original Object Detection Framework - Inference Pipeline

This module implements inference and post-processing:
1. Model inference (forward pass)
2. Non-Maximum Suppression (NMS) from scratch
3. Confidence thresholding
4. Multi-scale inference (optional)
5. Batch inference support

Author: Clean-room implementation
License: Original work
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2

from model import ObjectDetector


def non_maximum_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
    max_detections: int = 300
) -> torch.Tensor:
    """
    Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Algorithm:
    1. Sort boxes by confidence score (descending)
    2. Keep box with highest score
    3. Remove boxes with IoU > threshold with kept box
    4. Repeat until no boxes remain

    This is a completely original implementation from first principles.

    Args:
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum confidence score
        max_detections: Maximum number of detections to keep

    Returns:
        Indices of kept boxes
    """
    # Filter by score threshold
    keep_mask = scores > score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = []

    while sorted_indices.numel() > 0 and len(keep_indices) < max_detections:
        # Take box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx.item())

        if sorted_indices.numel() == 1:
            break

        # Get current box
        current_box = boxes[current_idx]

        # Get remaining boxes
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        # Calculate IoU with current box
        ious = calculate_iou_vectorized(current_box.unsqueeze(0), remaining_boxes)

        # Keep boxes with IoU < threshold
        keep_mask = ious.squeeze(0) < iou_threshold
        sorted_indices = remaining_indices[keep_mask]

    return torch.tensor(keep_indices, dtype=torch.long)


def calculate_iou_vectorized(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between one box and multiple boxes.

    Args:
        box1: Single box [1, 4] in (x1, y1, x2, y2) format
        boxes2: Multiple boxes [N, 4] in (x1, y1, x2, y2) format

    Returns:
        IoU values [1, N]
    """
    # Intersection coordinates
    inter_x1 = torch.max(box1[:, 0:1], boxes2[:, 0:1].T)
    inter_y1 = torch.max(box1[:, 1:2], boxes2[:, 1:2].T)
    inter_x2 = torch.min(box1[:, 2:3], boxes2[:, 2:3].T)
    inter_y2 = torch.min(box1[:, 3:4], boxes2[:, 3:4].T)

    # Intersection area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Box areas
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union area
    union_area = box1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)
    return iou


class ObjectDetectionInference:
    """
    Complete inference pipeline for object detection.

    Handles:
    - Image preprocessing
    - Model inference
    - Post-processing (NMS, thresholding)
    - Result formatting
    """

    def __init__(
        self,
        model: ObjectDetector,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        img_size: int = 640
    ):
        """
        Args:
            model: Trained object detection model
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            img_size: Input image size
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.img_size = img_size
        self.strides = [8, 16, 32]

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            List of detections, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'score': confidence score
                - 'class': class label
        """
        # Preprocess image
        input_tensor, scale, pad = self._preprocess(image)
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        predictions = self.model(input_tensor)

        # Post-process predictions
        detections = self._postprocess(predictions, scale, pad)

        return detections[0]  # Return first image (batch size 1)

    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of detection lists (one per image)
        """
        # Preprocess images
        batch_tensors = []
        batch_scales = []
        batch_pads = []

        for image in images:
            tensor, scale, pad = self._preprocess(image)
            batch_tensors.append(tensor)
            batch_scales.append(scale)
            batch_pads.append(pad)

        # Stack into batch
        batch_input = torch.cat(batch_tensors, dim=0).to(self.device)

        # Forward pass
        predictions = self.model(batch_input)

        # Post-process each image
        all_detections = []
        for i in range(len(images)):
            # Extract predictions for this image
            image_preds = [{k: v[i:i+1] for k, v in pred.items()} for pred in predictions]
            detections = self._postprocess(image_preds, batch_scales[i], batch_pads[i])
            all_detections.append(detections[0])

        return all_detections

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Steps:
        1. Resize with letterbox (maintain aspect ratio)
        2. Pad to square
        3. Normalize to [0, 1]
        4. Convert to tensor

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            Tuple of (preprocessed_tensor, scale, padding)
        """
        h, w = image.shape[:2]

        # Calculate scale
        scale = min(self.img_size / h, self.img_size / w)

        # Resize
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2

        # Place resized image
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert to tensor
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor, scale, (pad_h, pad_w)

    def _postprocess(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        scale: float,
        pad: Tuple[int, int]
    ) -> List[List[Dict]]:
        """
        Post-process model predictions to get final detections.

        Steps:
        1. Decode predictions from grid format
        2. Apply confidence thresholding
        3. Convert to absolute coordinates
        4. Apply NMS
        5. Adjust for image preprocessing

        Args:
            predictions: List of prediction dicts from model
            scale: Scale factor used in preprocessing
            pad: Padding (h, w) used in preprocessing

        Returns:
            List of detection lists (one per image in batch)
        """
        batch_size = predictions[0]['bbox'].shape[0]
        all_detections = [[] for _ in range(batch_size)]

        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            stride = self.strides[scale_idx]
            pred_bbox = pred['bbox']  # [B, 4, H, W]
            pred_obj = pred['obj']    # [B, 1, H, W]
            pred_cls = pred['cls']    # [B, num_classes, H, W]

            B, _, H, W = pred_bbox.shape

            # Create grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred_bbox.device),
                torch.arange(W, device=pred_bbox.device),
                indexing='ij'
            )

            # Process each image in batch
            for b in range(B):
                # Get predictions for this image
                bbox = pred_bbox[b]  # [4, H, W]
                obj = pred_obj[b]    # [1, H, W]
                cls = pred_cls[b]    # [num_classes, H, W]

                # Decode bounding boxes
                # bbox format: [dx, dy, dw, dh] relative to grid cell
                x_center = (bbox[0] + grid_x) * stride
                y_center = (bbox[1] + grid_y) * stride
                width = bbox[2] * stride
                height = bbox[3] * stride

                # Convert to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Stack boxes
                boxes = torch.stack([x1, y1, x2, y2], dim=0)  # [4, H, W]
                boxes = boxes.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]

                # Get objectness scores
                obj_scores = obj.squeeze(0).reshape(-1)  # [H*W]

                # Get class predictions
                cls_scores = F.softmax(cls, dim=0)  # [num_classes, H, W]
                cls_scores, cls_labels = cls_scores.max(dim=0)  # [H, W]
                cls_scores = cls_scores.reshape(-1)  # [H*W]
                cls_labels = cls_labels.reshape(-1)  # [H*W]

                # Combined confidence
                scores = obj_scores * cls_scores

                # Apply NMS
                keep_indices = non_maximum_suppression(
                    boxes,
                    scores,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.conf_threshold,
                    max_detections=self.max_detections
                )

                # Keep filtered detections
                if len(keep_indices) > 0:
                    kept_boxes = boxes[keep_indices]
                    kept_scores = scores[keep_indices]
                    kept_labels = cls_labels[keep_indices]

                    # Adjust for preprocessing (padding and scale)
                    pad_h, pad_w = pad
                    kept_boxes[:, [0, 2]] = (kept_boxes[:, [0, 2]] - pad_w) / scale
                    kept_boxes[:, [1, 3]] = (kept_boxes[:, [1, 3]] - pad_h) / scale

                    # Clamp to image bounds
                    kept_boxes = kept_boxes.clamp(min=0)

                    # Add to detections
                    for box, score, label in zip(kept_boxes, kept_scores, kept_labels):
                        all_detections[b].append({
                            'bbox': box.cpu().numpy(),
                            'score': score.item(),
                            'class': label.item()
                        })

        return all_detections


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: List[str] = None,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """
    Visualize detections on image.

    Args:
        image: Input image (H, W, 3) in RGB format
        detections: List of detection dictionaries
        class_names: List of class names (optional)
        conf_threshold: Minimum confidence to display

    Returns:
        Image with drawn bounding boxes
    """
    vis_image = image.copy()

    for det in detections:
        if det['score'] < conf_threshold:
            continue

        # Get box coordinates
        x1, y1, x2, y2 = det['bbox'].astype(int)

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        class_id = det['class']
        label = class_names[class_id] if class_names else f"Class {class_id}"
        label_text = f"{label}: {det['score']:.2f}"

        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)

        # Draw text
        cv2.putText(vis_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


if __name__ == '__main__':
    print("Inference Pipeline Example")
    print("=" * 50)

    print("""
    Usage:

    from model import build_model
    from inference import ObjectDetectionInference, visualize_detections
    import cv2

    # Load model
    model = build_model(num_classes=80)
    model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])

    # Create inference engine
    inference = ObjectDetectionInference(
        model=model,
        device='cuda',
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    # Run inference
    image = cv2.imread('image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = inference.predict(image)

    # Visualize
    vis_image = visualize_detections(image, detections)
    cv2.imwrite('result.jpg', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    print(f"Found {len(detections)} objects")
    for det in detections:
        print(f"  Class {det['class']}: {det['score']:.3f} at {det['bbox']}")
    """)

    # Test NMS
    print("\nTesting NMS...")
    boxes = torch.tensor([
        [100, 100, 200, 200],
        [105, 105, 205, 205],  # High overlap with first box
        [300, 300, 400, 400],
        [310, 310, 410, 410],  # High overlap with third box
    ], dtype=torch.float32)

    scores = torch.tensor([0.9, 0.8, 0.85, 0.75])

    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.1)
    print(f"Input boxes: {len(boxes)}")
    print(f"After NMS: {len(keep)} boxes kept")
    print(f"Kept indices: {keep}")
