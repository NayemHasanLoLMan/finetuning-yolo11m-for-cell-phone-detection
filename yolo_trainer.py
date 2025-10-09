"""
Train YOLO11m model on COCO cell phone dataset.

This script:
1. Converts COCO annotations to YOLO format
2. Sets up the dataset structure
3. Trains YOLO11m model
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import yaml

# Configuration
COCO_DATASET_DIR = "coco_cellphone_dataset"
YOLO_DATASET_DIR = "yolo_cellphone_dataset"
ANNOTATIONS_DIR = os.path.join(COCO_DATASET_DIR, "annotations")
IMAGES_DIR = os.path.join(COCO_DATASET_DIR, "images")

def create_yolo_structure():
    """Create YOLO dataset directory structure."""
    dirs = [
        os.path.join(YOLO_DATASET_DIR, "images", "train"),
        os.path.join(YOLO_DATASET_DIR, "images", "val"),
        os.path.join(YOLO_DATASET_DIR, "labels", "train"),
        os.path.join(YOLO_DATASET_DIR, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created YOLO dataset structure")

def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox format to YOLO format.
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, width, height = bbox
    
    # Calculate center coordinates
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def convert_coco_to_yolo(split='train'):
    """Convert COCO format annotations to YOLO format."""
    print(f"\nConverting {split} annotations to YOLO format...")
    
    # Load COCO annotations
    anno_file = os.path.join(ANNOTATIONS_DIR, f"cellphone_{split}2017.json")
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Create image_id to image info mapping
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Process each image
    converted_count = 0
    for img_id, annotations in image_annotations.items():
        img_info = images_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source and destination paths
        src_img_path = os.path.join(IMAGES_DIR, f"{split}2017", img_filename)
        dst_img_path = os.path.join(YOLO_DATASET_DIR, "images", split, img_filename)
        
        # Copy image
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found - {src_img_path}")
            continue
        
        # Create YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(YOLO_DATASET_DIR, "labels", split, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Class 0 for cell phone (single class)
                class_id = 0
                
                # Convert bbox to YOLO format
                bbox = ann['bbox']
                yolo_bbox = convert_coco_bbox_to_yolo(bbox, img_width, img_height)
                
                # Write to file: class x_center y_center width height
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} images for {split} set")
    return converted_count

def create_yaml_config():
    """Create YAML configuration file for YOLO training."""
    config = {
        'path': os.path.abspath(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # number of classes
        'names': ['cell phone']  # class names
    }
    
    yaml_path = os.path.join(YOLO_DATASET_DIR, 'cellphone.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created YAML config at {yaml_path}")
    return yaml_path

def train_yolo11m(yaml_path, epochs=100, imgsz=640, batch=16):
    """Train YOLO11m model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Please install: pip install ultralytics")
        return
    
    print("\n" + "="*60)
    print("Starting YOLO11m Training")
    print("="*60)
    
    # Load YOLO11m model
    model = YOLO('yolo11m.pt')  # Will download pretrained weights automatically
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='cellphone_yolo11m',
        project='runs/detect',
        patience=50,  # Early stopping patience
        save=True,
        device=0,  # Use GPU 0, change to 'cpu' if no GPU
        workers=8,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=True,  # Single class dataset
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        show=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        boxes=True,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved at: runs/detect/cellphone_yolo11m/weights/best.pt")
    print(f"Last model saved at: runs/detect/cellphone_yolo11m/weights/last.pt")
    print(f"Training results: runs/detect/cellphone_yolo11m/")
    
    return results

def validate_model(model_path, yaml_path):
    """Validate the trained model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    
    print("\n" + "="*60)
    print("Validating Model")
    print("="*60)
    
    model = YOLO(model_path)
    metrics = model.val(data=yaml_path)
    
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics

def main():
    print("="*60)
    print("YOLO11m Cell Phone Training Pipeline")
    print("="*60)
    
    # Step 1: Create YOLO directory structure
    print("\n[1/5] Creating YOLO dataset structure...")
    create_yolo_structure()
    
    # Step 2: Convert COCO to YOLO format
    print("\n[2/5] Converting annotations...")
    train_count = convert_coco_to_yolo('train')
    val_count = convert_coco_to_yolo('val')
    
    print(f"\nDataset Summary:")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Total images: {train_count + val_count}")
    
    # Step 3: Create YAML config
    print("\n[3/5] Creating YAML configuration...")
    yaml_path = create_yaml_config()
    
    # Step 4: Train model
    print("\n[4/5] Training YOLO11m model...")
    print("\nTraining Parameters:")
    print("  Model: YOLO11m")
    print("  Epochs: 100")
    print("  Image Size: 640")
    print("  Batch Size: 16")
    print("\nThis may take several hours depending on your hardware...")
    
    train_yolo11m(
        yaml_path=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16  # Adjust based on your GPU memory
    )
    
    # Step 5: Validate
    print("\n[5/5] Validating trained model...")
    best_model_path = "runs/detect/cellphone_yolo11m/weights/best.pt"
    if os.path.exists(best_model_path):
        validate_model(best_model_path, yaml_path)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Check training results: runs/detect/cellphone_yolo11m/")
    print("2. View training curves and metrics in the results folder")
    print("3. Use best.pt for inference on new images")
    print("\nInference example:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('runs/detect/cellphone_yolo11m/weights/best.pt')")
    print("  results = model('path/to/image.jpg')")

if __name__ == "__main__":
    # Install required packages
    print("Required packages:")
    print("  pip install ultralytics pillow pyyaml")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()