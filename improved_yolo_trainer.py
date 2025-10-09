"""
Train improved YOLO11m model with combined datasets
Combines your existing COCO data + 3 new Roboflow datasets
"""

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

# ============================================================
# STEP 1: Combine all datasets into one
# ============================================================

def combine_all_datasets():
    """Combine COCO + Roboflow datasets into one unified dataset."""
    
    print("="*60)
    print("📦 Combining All Datasets")
    print("="*60)
    
    # Create combined dataset directory
    combined_dir = "combined_phone_dataset"
    os.makedirs(f"{combined_dir}/images/train", exist_ok=True)
    os.makedirs(f"{combined_dir}/images/val", exist_ok=True)
    os.makedirs(f"{combined_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{combined_dir}/labels/val", exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    # List of all datasets to combine
    datasets = [
        {
            'name': 'COCO',
            'path': 'yolo_cellphone_dataset',
            'train_img': 'images/train',
            'val_img': 'images/val',
            'train_lbl': 'labels/train',
            'val_lbl': 'labels/val'
        },
        {
            'name': 'MobilePhone',
            'path': 'dataset_mobile_phone',
            'train_img': 'train/images',
            'val_img': 'valid/images',
            'train_lbl': 'train/labels',
            'val_lbl': 'valid/labels'
        },
        {
            'name': 'PhoneCall',
            'path': 'dataset_phone_call',
            'train_img': 'train/images',
            'val_img': 'valid/images',
            'train_lbl': 'train/labels',
            'val_lbl': 'valid/labels'
        },
        {
            'name': 'PhoneHand',
            'path': 'dataset_phone_hand',
            'train_img': 'train/images',
            'val_img': 'valid/images',
            'train_lbl': 'train/labels',
            'val_lbl': 'valid/labels'
        }
    ]
    
    # Copy all datasets
    for dataset in datasets:
        if not os.path.exists(dataset['path']):
            print(f"⚠️  {dataset['name']} not found, skipping...")
            continue
        
        print(f"\n📂 Adding {dataset['name']} dataset...")
        
        # Copy training data
        train_img_dir = os.path.join(dataset['path'], dataset['train_img'])
        train_lbl_dir = os.path.join(dataset['path'], dataset['train_lbl'])
        
        if os.path.exists(train_img_dir):
            for img_file in Path(train_img_dir).glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Create unique filename
                    new_name = f"{dataset['name']}_{img_file.name}"
                    
                    # Copy image
                    shutil.copy2(img_file, f"{combined_dir}/images/train/{new_name}")
                    
                    # Copy label
                    lbl_file = Path(train_lbl_dir) / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        shutil.copy2(lbl_file, f"{combined_dir}/labels/train/{Path(new_name).stem}.txt")
                    
                    total_train += 1
        
        # Copy validation data
        val_img_dir = os.path.join(dataset['path'], dataset['val_img'])
        val_lbl_dir = os.path.join(dataset['path'], dataset['val_lbl'])
        
        if os.path.exists(val_img_dir):
            for img_file in Path(val_img_dir).glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    new_name = f"{dataset['name']}_{img_file.name}"
                    
                    shutil.copy2(img_file, f"{combined_dir}/images/val/{new_name}")
                    
                    lbl_file = Path(val_lbl_dir) / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        shutil.copy2(lbl_file, f"{combined_dir}/labels/val/{Path(new_name).stem}.txt")
                    
                    total_val += 1
        
        print(f"  ✅ Added {dataset['name']}")
    
    print("\n" + "="*60)
    print("📊 Combined Dataset Summary")
    print("="*60)
    print(f"  Training images:   {total_train}")
    print(f"  Validation images: {total_val}")
    print(f"  Total images:      {total_train + total_val}")
    print("="*60)
    
    # Create YAML config
    config = {
        'path': os.path.abspath(combined_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['cell phone']
    }
    
    yaml_path = f"{combined_dir}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Created dataset config: {yaml_path}")
    
    return yaml_path, total_train, total_val

# ============================================================
# STEP 2: Train improved model
# ============================================================

def train_improved_model(data_yaml):
    """Train YOLO11m on combined dataset starting from your best model."""
    
    print("\n" + "="*60)
    print("🚀 Training Improved YOLO11m Model")
    print("="*60)
    
    # Load your existing best model to continue training
    print("\n📥 Loading your existing model: runs/detect/cellphone_yolo11m/weights/best.pt")
    model = YOLO('runs/detect/cellphone_yolo11m/weights/best.pt')
    
    print("\n⚙️  Training Configuration:")
    print("  - Epochs: 50 (continue training)")
    print("  - Image size: 640")
    print("  - Batch size: 16")
    print("  - Enhanced augmentation: ON")
    print("  - Early stopping: 30 epochs patience")
    print("  - Device: GPU (CUDA)")
    
    print("\n🔥 Starting training...")
    print("This will take 2-4 hours on your RTX 4060 Ti")
    print("-"*60)
    
    # Train with optimized settings
    results = model.train(
        data=data_yaml,
        epochs=50,              # Additional epochs
        imgsz=640,
        batch=16,
        name='cellphone_improved',
        project='runs/detect',
        patience=30,            # Early stopping
        save=True,
        device=0,               # GPU
        workers=8,
        optimizer='AdamW',      # Better optimizer
        lr0=0.001,             # Lower learning rate for fine-tuning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Enhanced augmentation to reduce overfitting
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation
        hsv_v=0.4,             # Value/brightness
        degrees=10,            # Rotation
        translate=0.1,         # Translation
        scale=0.5,             # Scale
        shear=0.0,             # Shear
        perspective=0.0,       # Perspective
        flipud=0.5,            # Vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # Mixup augmentation
        copy_paste=0.1,        # Copy-paste augmentation
        
        # Other settings
        single_cls=True,       # Single class (cell phone)
        amp=True,              # Mixed precision
        plots=True,            # Save plots
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    
    # Validate the model
    print("\n📊 Running validation...")
    metrics = model.val()
    
    print("\n" + "="*60)
    print("📈 Final Performance Metrics")
    print("="*60)
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  F1-Score:    {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}")
    print("="*60)
    
    print("\n✅ Improved model saved to:")
    print("   📁 runs/detect/cellphone_improved/weights/best.pt")
    print("   📁 runs/detect/cellphone_improved/weights/last.pt")
    
    return results

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*60)
    print("🎯 Cell Phone Detection - Model Improvement")
    print("="*60)
    
    print("\nThis script will:")
    print("  1. Combine all datasets (COCO + Roboflow)")
    print("  2. Continue training from your best.pt model")
    print("  3. Apply enhanced augmentation")
    print("  4. Create an improved model")
    
    input("\n⏸  Press Enter to start...")
    
    # Step 1: Combine datasets
    print("\n[STEP 1/2] Combining datasets...")
    data_yaml, train_count, val_count = combine_all_datasets()
    
    print(f"\n✅ Combined dataset ready!")
    print(f"   Training samples: {train_count}")
    print(f"   Validation samples: {val_count}")
    
    # Step 2: Train
    print("\n[STEP 2/2] Training improved model...")
    train_improved_model(data_yaml)
    
    print("\n" + "="*60)
    print("🎉 All Done!")
    print("="*60)
    print("\n📝 Next steps:")
    print("  1. Test your improved model:")
    print("     python simple_detection.py")
    print("  2. Update model path in your detection script:")
    print("     model = YOLO('runs/detect/cellphone_improved/weights/best.pt')")
    print("\n💡 Your new model should have:")
    print("  ✅ Better detection accuracy")
    print("  ✅ Fewer missed phones")
    print("  ✅ More stable/continuous detection")
    print("  ✅ Fewer false positives")
    print("="*60)

if __name__ == "__main__":
    main()