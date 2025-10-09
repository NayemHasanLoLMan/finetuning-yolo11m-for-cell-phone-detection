"""
Simple script to download phone detection datasets from Roboflow
Just run and enter your API key!
"""

from roboflow import Roboflow

# ============================================================
# GET YOUR FREE API KEY
# ============================================================
# 1. Go to: https://app.roboflow.com/
# 2. Sign up (free)
# 3. Get API key from: https://app.roboflow.com/settings/api
# ============================================================

print("="*60)
print("📦 Roboflow Phone Detection Dataset Downloader")
print("="*60)

# Enter your API key here or when prompted
API_KEY = input("\nEnter your Roboflow API key: ").strip()

if not API_KEY:
    print("\n❌ No API key provided!")
    print("Get your free API key from: https://app.roboflow.com/settings/api")
    exit()

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)

print("\n" + "="*60)
print("Downloading Datasets...")
print("="*60)

# ============================================================
# DATASET 1: Mobile Phone Detection (1674 images)
# ============================================================
print("\n[1/3] 📱 Mobile Phone Detection Dataset (1674 images)")
print("Downloading...")
try:
    project1 = rf.workspace("realtime-mobile-phone-usage-detection-in-everyday-scenarios-using-yolo").project("mobile-phone-detection-mtsje-xhoma")
    dataset1 = project1.version(1).download("yolov8", location="dataset_mobile_phone")
    print("✅ Downloaded to: dataset_mobile_phone/")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================
# DATASET 2: Phone Call Usage (3115 images)
# ============================================================
print("\n[2/3] 📞 Phone Call Usage Dataset (3115 images)")
print("Downloading...")
try:
    project2 = rf.workspace("phoneusagedetection").project("phone-call-usage")
    dataset2 = project2.version(1).download("yolov8", location="dataset_phone_call")
    print("✅ Downloaded to: dataset_phone_call/")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================
# DATASET 3: Phone in Hand (338 images)
# ============================================================
print("\n[3/3] 🤳 Phone in Hand Dataset (338 images)")
print("Downloading...")
try:
    project3 = rf.workspace("phone-in-hand-detection").project("phone-in-hand-detection")
    dataset3 = project3.version(1).download("yolov8", location="dataset_phone_hand")
    print("✅ Downloaded to: dataset_phone_hand/")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("✅ Download Complete!")
print("="*60)
print("\nDatasets downloaded:")
print("  1. dataset_mobile_phone/  (1674 images)")
print("  2. dataset_phone_call/    (3115 images)")
print("  3. dataset_phone_hand/    (338 images)")
print("\nTotal: ~5,127 images with annotations!")
print("\nEach dataset contains:")
print("  - train/images/  (training images)")
print("  - train/labels/  (training annotations)")
print("  - valid/images/  (validation images)")
print("  - valid/labels/  (validation annotations)")
print("  - data.yaml      (dataset config)")
print("="*60)