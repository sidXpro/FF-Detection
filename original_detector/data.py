"""
Original Object Detection Framework - Data Pipeline

This module implements:
1. Custom Dataset class for COCO-format annotations
2. Data augmentations (crop, flip, color jitter, mosaic)
3. Label encoding for anchor-free detection

Author: Clean-room implementation
License: Original work
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import cv2


class COCODetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for COCO-format object detection annotations.

    Expected directory structure:
        data_root/
            images/
                train/
                    img1.jpg
                    img2.jpg
                    ...
            annotations/
                instances_train.json

    COCO JSON format:
        {
            "images": [{"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640}, ...],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}, ...],
            "categories": [{"id": 1, "name": "person"}, ...]
        }
    """

    def __init__(self, data_root: str, annotation_file: str, img_size: int = 640,
                 augment: bool = True, mosaic_prob: float = 0.5):
        """
        Args:
            data_root: Root directory containing images
            annotation_file: Path to COCO JSON annotation file
            img_size: Target image size for training
            augment: Whether to apply data augmentations
            mosaic_prob: Probability of applying mosaic augmentation
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Build mapping from image_id to annotations
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        self.num_classes = len(self.categories)

        # Group annotations by image
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a training sample.

        Returns:
            Tuple of (image, target) where:
                - image: [3, H, W] tensor
                - target: dict with 'boxes' [N, 4] and 'labels' [N]
        """
        if self.augment and random.random() < self.mosaic_prob:
            # Mosaic augmentation: combine 4 images
            image, boxes, labels = self._load_mosaic(idx)
        else:
            # Regular loading
            image, boxes, labels = self._load_image_and_labels(idx)

        # Apply augmentations
        if self.augment:
            image, boxes, labels = self._augment(image, boxes, labels)

        # Resize to target size
        image, boxes = self._resize(image, boxes, self.img_size)

        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()

        target = {
            'boxes': boxes,  # [N, 4] in (x_center, y_center, w, h) format
            'labels': labels  # [N]
        }

        return image, target

    def _load_image_and_labels(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load single image and its annotations.

        Returns:
            Tuple of (image, boxes, labels)
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_root / 'images' / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to center format
            x_center = x + w / 2
            y_center = y + h / 2

            boxes.append([x_center, y_center, w, h])
            labels.append(self.categories[ann['category_id']])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        return image, boxes, labels

    def _load_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mosaic augmentation: combine 4 images into one.

        This creates a 2x2 grid of images, which helps the model learn
        to detect objects at different scales and positions.
        """
        # Select 4 images (including current one)
        indices = [idx] + random.choices(range(len(self)), k=3)

        # Create output image
        mosaic_size = self.img_size * 2
        mosaic_img = np.full((mosaic_size, mosaic_size, 3), 114, dtype=np.uint8)

        # Center point of mosaic
        center_x = self.img_size
        center_y = self.img_size

        all_boxes = []
        all_labels = []

        # Place 4 images
        for i, index in enumerate(indices):
            image, boxes, labels = self._load_image_and_labels(index)
            h, w = image.shape[:2]

            # Determine position in mosaic
            if i == 0:  # Top-left
                x1, y1, x2, y2 = 0, 0, center_x, center_y
                crop_x1, crop_y1 = w - center_x, h - center_y
                crop_x2, crop_y2 = w, h
            elif i == 1:  # Top-right
                x1, y1, x2, y2 = center_x, 0, mosaic_size, center_y
                crop_x1, crop_y1 = 0, h - center_y
                crop_x2, crop_y2 = min(w, self.img_size), h
            elif i == 2:  # Bottom-left
                x1, y1, x2, y2 = 0, center_y, center_x, mosaic_size
                crop_x1, crop_y1 = w - center_x, 0
                crop_x2, crop_y2 = w, min(h, self.img_size)
            else:  # Bottom-right
                x1, y1, x2, y2 = center_x, center_y, mosaic_size, mosaic_size
                crop_x1, crop_y1 = 0, 0
                crop_x2, crop_y2 = min(w, self.img_size), min(h, self.img_size)

            # Resize and place image
            crop_h, crop_w = crop_y2 - crop_y1, crop_x2 - crop_x1
            if crop_h > 0 and crop_w > 0:
                image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                mosaic_img[y1:y1+crop_h, x1:x1+crop_w] = image_crop

                # Adjust box coordinates
                if len(boxes) > 0:
                    boxes_copy = boxes.copy()
                    boxes_copy[:, 0] = boxes[:, 0] - crop_x1 + x1  # x_center
                    boxes_copy[:, 1] = boxes[:, 1] - crop_y1 + y1  # y_center

                    all_boxes.append(boxes_copy)
                    all_labels.append(labels)

        # Combine all boxes and labels
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            # Filter boxes that are mostly outside the image
            valid = (boxes[:, 0] > 0) & (boxes[:, 1] > 0) & \
                   (boxes[:, 0] < mosaic_size) & (boxes[:, 1] < mosaic_size)
            boxes = boxes[valid]
            labels = labels[valid]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return mosaic_img, boxes, labels

    def _augment(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentations.

        Augmentations:
        1. Random horizontal flip
        2. Color jitter (brightness, contrast, saturation)
        3. Random scaling
        """
        h, w = image.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            if len(boxes) > 0:
                boxes[:, 0] = w - boxes[:, 0]  # Flip x coordinate

        # Color jitter
        if random.random() < 0.5:
            # Brightness
            alpha = random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            # Contrast
            alpha = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * alpha + mean, 0, 255).astype(np.uint8)

        return image, boxes, labels

    def _resize(self, image: np.ndarray, boxes: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image and adjust box coordinates.

        Uses letterbox resizing to maintain aspect ratio.
        """
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)

        # New dimensions
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        # Place resized image
        padded_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image

        # Adjust box coordinates
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * scale + pad_w  # x_center
            boxes[:, 1] = boxes[:, 1] * scale + pad_h  # y_center
            boxes[:, 2] = boxes[:, 2] * scale  # width
            boxes[:, 3] = boxes[:, 3] * scale  # height

        return padded_img, boxes


def encode_targets(boxes: torch.Tensor, labels: torch.Tensor, num_classes: int,
                   img_size: int = 640, strides: List[int] = [8, 16, 32]) -> List[torch.Tensor]:
    """
    Encode ground truth boxes and labels into grid format for anchor-free detection.

    For each scale, create a grid where each cell is responsible for detecting
    objects whose center falls within that cell.

    Args:
        boxes: Ground truth boxes [N, 4] in (x_center, y_center, w, h) format
        labels: Ground truth labels [N]
        num_classes: Number of object classes
        img_size: Input image size
        strides: List of stride values for each scale

    Returns:
        List of encoded targets for each scale [H, W, 5+num_classes]
        Format: [objectness, x, y, w, h, class_0, class_1, ..., class_N]
    """
    encoded_targets = []

    for stride in strides:
        grid_size = img_size // stride
        target = torch.zeros(grid_size, grid_size, 5 + num_classes)

        for box, label in zip(boxes, labels):
            x_center, y_center, w, h = box

            # Find grid cell containing the center
            grid_x = int(x_center / stride)
            grid_y = int(y_center / stride)

            # Check if within bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                # Set objectness
                target[grid_y, grid_x, 0] = 1.0

                # Set box coordinates (normalized to grid cell)
                target[grid_y, grid_x, 1] = x_center / stride - grid_x
                target[grid_y, grid_x, 2] = y_center / stride - grid_y
                target[grid_y, grid_x, 3] = w / stride
                target[grid_y, grid_x, 4] = h / stride

                # Set class (one-hot encoding)
                if 0 <= label < num_classes:
                    target[grid_y, grid_x, 5 + label] = 1.0

        encoded_targets.append(target)

    return encoded_targets


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    """
    Custom collate function for DataLoader.

    Handles variable number of objects per image by creating batched grid targets.

    Args:
        batch: List of (image, target) tuples from dataset

    Returns:
        Batched images and targets
    """
    images = []
    all_boxes = []
    all_labels = []

    for img, target in batch:
        images.append(img)
        all_boxes.append(target['boxes'])
        all_labels.append(target['labels'])

    # Stack images
    images = torch.stack(images, dim=0)

    # Encode targets for each scale
    batch_size = len(batch)
    num_classes = 80  # COCO classes (should be configurable)

    # Create grid targets for each scale
    strides = [8, 16, 32]
    img_size = images.shape[-1]

    batch_grid_targets = []
    for stride in strides:
        grid_size = img_size // stride
        grid_target = torch.zeros(batch_size, grid_size, grid_size, 6 + num_classes)

        for i, (boxes, labels) in enumerate(zip(all_boxes, all_labels)):
            if len(boxes) > 0:
                encoded = encode_targets(boxes, labels, num_classes, img_size, [stride])[0]
                grid_target[i] = encoded

        batch_grid_targets.append(grid_target)

    targets = {
        'boxes': all_boxes,
        'labels': all_labels,
        'grid_targets': batch_grid_targets
    }

    return images, targets


def create_dataloader(data_root: str, annotation_file: str, batch_size: int = 16,
                     num_workers: int = 4, img_size: int = 640, augment: bool = True) -> DataLoader:
    """
    Factory function to create a DataLoader for training.

    Args:
        data_root: Root directory containing images
        annotation_file: Path to COCO JSON annotation file
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        img_size: Target image size
        augment: Whether to apply augmentations

    Returns:
        DataLoader instance
    """
    dataset = COCODetectionDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        img_size=img_size,
        augment=augment
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


if __name__ == '__main__':
    # Test data pipeline
    print("Testing data pipeline...")

    # Note: This requires actual COCO dataset to test
    # For demonstration, we'll show the expected usage

    print("""
    Usage example:

    # Create dataloader
    train_loader = create_dataloader(
        data_root='path/to/coco',
        annotation_file='path/to/annotations/instances_train.json',
        batch_size=16,
        num_workers=4,
        img_size=640,
        augment=True
    )

    # Iterate through batches
    for images, targets in train_loader:
        # images: [B, 3, 640, 640]
        # targets: dict with 'boxes', 'labels', 'grid_targets'
        print(f"Batch shape: {images.shape}")
        print(f"Number of objects per image: {[len(b) for b in targets['boxes']]}")
        break
    """)
