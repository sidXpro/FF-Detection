#!/usr/bin/env python3
"""
Example Inference Script for Original Object Detection Framework

This script demonstrates how to use the framework for inference on images.

Usage:
    Single image:
        python example_inference.py --model checkpoint.pt --image image.jpg

    Directory of images:
        python example_inference.py --model checkpoint.pt --image_dir ./images --output_dir ./results

Author: Clean-room implementation
License: Original work
"""

import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from model import build_model
from inference import ObjectDetectionInference, visualize_detections


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Object Detection Inference')

    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=80,
                       help='Number of object classes')

    # Input/Output
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single input image')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')

    # Inference parameters
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--max_detections', type=int, default=300,
                       help='Maximum number of detections per image')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--class_names', type=str, default=None,
                       help='Path to class names file (one name per line)')

    return parser.parse_args()


def load_model(checkpoint_path: str, num_classes: int, device: str) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of object classes
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Build model
    model = build_model(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    return model


def load_class_names(class_names_path: str) -> list:
    """
    Load class names from file.

    Args:
        class_names_path: Path to text file with class names

    Returns:
        List of class names
    """
    with open(class_names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def process_single_image(
    inference_engine: ObjectDetectionInference,
    image_path: str,
    output_path: str,
    class_names: list,
    conf_threshold: float = 0.5
):
    """
    Process a single image.

    Args:
        inference_engine: Inference engine
        image_path: Path to input image
        output_path: Path to save output image
        class_names: List of class names
        conf_threshold: Confidence threshold for visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    detections = inference_engine.predict(image_rgb)

    # Print detections
    print(f"\nImage: {Path(image_path).name}")
    print(f"Found {len(detections)} objects:")
    for det in detections:
        class_name = class_names[det['class']] if class_names else f"Class {det['class']}"
        print(f"  {class_name}: {det['score']:.3f} at {det['bbox'].astype(int).tolist()}")

    # Visualize
    vis_image = visualize_detections(image_rgb, detections, class_names, conf_threshold)
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # Save result
    cv2.imwrite(output_path, vis_image_bgr)
    print(f"Saved result to: {output_path}")


def process_image_directory(
    inference_engine: ObjectDetectionInference,
    image_dir: str,
    output_dir: str,
    class_names: list,
    conf_threshold: float = 0.5
):
    """
    Process all images in a directory.

    Args:
        inference_engine: Inference engine
        image_dir: Input directory containing images
        output_dir: Output directory for results
        class_names: List of class names
        conf_threshold: Confidence threshold for visualization
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"\nProcessing {len(image_paths)} images...")

    # Process each image
    for image_path in tqdm(image_paths):
        output_path = output_dir / f"{image_path.stem}_result{image_path.suffix}"

        try:
            process_single_image(
                inference_engine,
                str(image_path),
                str(output_path),
                class_names,
                conf_threshold
            )
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    print(f"\nResults saved to: {output_dir}")


def main():
    """Main inference function."""
    args = parse_args()

    # Print configuration
    print("=" * 70)
    print("Original Object Detection Framework - Inference")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Load class names
    if args.class_names:
        class_names = load_class_names(args.class_names)
    else:
        class_names = COCO_CLASSES

    print(f"Using {len(class_names)} classes")

    # Load model
    print(f"\nLoading model from {args.model}...")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("Warning: CUDA not available, using CPU instead")

    model = load_model(args.model, args.num_classes, device)

    # Create inference engine
    inference_engine = ObjectDetectionInference(
        model=model,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        img_size=args.img_size
    )

    print("Model loaded successfully!")

    # Run inference
    if args.image:
        # Single image
        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_result{Path(args.image).suffix}"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        process_single_image(
            inference_engine,
            args.image,
            str(output_path),
            class_names,
            args.conf_threshold
        )

    elif args.image_dir:
        # Directory of images
        process_image_directory(
            inference_engine,
            args.image_dir,
            args.output_dir,
            class_names,
            args.conf_threshold
        )

    else:
        print("Error: Please specify either --image or --image_dir")
        return

    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
