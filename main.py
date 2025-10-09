"""
Simple Cell Phone Detection using YOLO11m
Two modes: Image detection or Webcam detection
"""

from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/detect/cellphone_yolo11m/weights/best.pt')

# ============================================================
# METHOD 1: Detect on a single image
# ============================================================
def detect_image(image_path):
    """Detect cell phones in an image."""
    print(f"\n🔍 Detecting cell phones in: {image_path}")
    
    # Run detection
    results = model(image_path, conf=0.25)
    
    # Get number of detections
    num_phones = len(results[0].boxes)
    print(f"✅ Found {num_phones} cell phone(s)!")
    
    # Save result
    results[0].save('detected_output.jpg')
    print("💾 Saved result to: detected_output.jpg")
    
    # Show result
    results[0].show()
    
    return results


# ============================================================
# METHOD 2: Detect on webcam (real-time)
# ============================================================
def detect_webcam():
    """Real-time detection using webcam."""
    print("\n📹 Starting webcam detection...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.25, verbose=False)
        
        # Display results
        annotated_frame = results[0].plot()
        cv2.imshow('Cell Phone Detection - Press Q to Quit', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam detection stopped")


# ============================================================
# MAIN FUNCTION
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🤖 YOLO11m Cell Phone Detector")
    print("="*60)
    
    print("\nChoose detection mode:")
    print("1. Single image")
    print("2. Webcam (real-time)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        path = input("Enter image path: ").strip()
        detect_image(path)
    
    elif choice == '2':
        detect_webcam()
    
    else:
        print("❌ Invalid choice! Please enter 1 or 2")


# ============================================================
# QUICK EXAMPLES - Uncomment to use directly
# ============================================================

# Example 1: Detect on image
# detect_image('phone.jpg')

# Example 2: Detect on webcam
# detect_webcam()

# Example 3: Ultra simple one-liner for image
# results = model('phone.jpg')
# results[0].show()