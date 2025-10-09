"""
Download COCO dataset images and annotations for cell phone category only.

This script filters the COCO dataset to download only images containing cell phones
along with their annotations.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

# Configuration
OUTPUT_DIR = "coco_cellphone_dataset"
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# COCO URLs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

# Cell phone category ID in COCO dataset
CELL_PHONE_CATEGORY_ID = 77

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def filter_annotations_for_cellphone(annotations_file, output_file):
    """Filter annotations to keep only cell phone instances."""
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Filter annotations for cell phone category
    cellphone_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['category_id'] == CELL_PHONE_CATEGORY_ID
    ]
    
    # Get unique image IDs that contain cell phones
    cellphone_image_ids = set(ann['image_id'] for ann in cellphone_annotations)
    
    # Filter images
    cellphone_images = [
        img for img in coco_data['images'] 
        if img['id'] in cellphone_image_ids
    ]
    
    # Filter categories to keep only cell phone
    cellphone_category = [
        cat for cat in coco_data['categories'] 
        if cat['id'] == CELL_PHONE_CATEGORY_ID
    ]
    
    # Create filtered dataset
    filtered_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': cellphone_images,
        'annotations': cellphone_annotations,
        'categories': cellphone_category
    }
    
    # Save filtered annotations
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered annotations saved to {output_file}")
    print(f"Found {len(cellphone_images)} images with {len(cellphone_annotations)} cell phone instances")
    
    return [img['file_name'] for img in cellphone_images]

def download_specific_images(image_list, split='train2017'):
    """Download specific images from COCO dataset."""
    base_url = f"http://images.cocodataset.org/{split}/"
    split_dir = os.path.join(IMAGES_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Downloading {len(image_list)} images from {split}...")
    for img_name in tqdm(image_list, desc=f"Downloading {split} images"):
        img_url = base_url + img_name
        img_path = os.path.join(split_dir, img_name)
        
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading {img_name}: {e}")

def main():
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    print("=" * 60)
    print("COCO Cell Phone Dataset Downloader")
    print("=" * 60)
    
    # Download and extract annotations
    annotations_zip = os.path.join(OUTPUT_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(annotations_zip):
        print("\n1. Downloading annotations...")
        download_file(COCO_ANNOTATIONS_URL, annotations_zip)
    else:
        print("\n1. Annotations already downloaded")
    
    # Extract annotations if not already extracted
    annotations_extracted = os.path.join(OUTPUT_DIR, "annotations")
    if not os.path.exists(os.path.join(annotations_extracted, "instances_train2017.json")):
        extract_zip(annotations_zip, OUTPUT_DIR)
    
    # Filter train annotations
    print("\n2. Filtering train annotations for cell phones...")
    train_cellphone_images = filter_annotations_for_cellphone(
        os.path.join(annotations_extracted, "instances_train2017.json"),
        os.path.join(ANNOTATIONS_DIR, "cellphone_train2017.json")
    )
    
    # Filter val annotations
    print("\n3. Filtering validation annotations for cell phones...")
    val_cellphone_images = filter_annotations_for_cellphone(
        os.path.join(annotations_extracted, "instances_val2017.json"),
        os.path.join(ANNOTATIONS_DIR, "cellphone_val2017.json")
    )
    
    # Download cell phone images
    print("\n4. Downloading cell phone images...")
    download_specific_images(train_cellphone_images, 'train2017')
    download_specific_images(val_cellphone_images, 'val2017')
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print(f"- Annotations: {ANNOTATIONS_DIR}")
    print(f"- Images: {IMAGES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    # Install required packages
    print("Make sure you have installed the required packages:")
    print("pip install requests tqdm")
    print()
    main()