
import numpy as np
from detector import NanoDetDetector

detector = NanoDetDetector()

# Blank image: this tests inference, not actual vehicle detection.
image = np.zeros((720, 1280, 3), dtype=np.uint8)

detections = detector.detect(image)

print("\nInference completed!")
print("Number of detections:", len(detections))
print("Confidence threshold:", 0.55)

for item in detections:
    print(
        item["class_name"],
        "Confidence:",
        round(item["confidence"], 3)
    )
