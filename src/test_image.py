
import cv2
from detector import NanoDetDetector

detector = NanoDetDetector()

IMAGE_PATH = "videos/test_road.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

detections = detector.detect(image)

print("\n----- VEHICLE DETECTION RESULTS -----")

if not detections:
    print("No target objects detected.")

for item in detections:
    print(
        f"Class: {item['class_name']} | "
        f"Confidence: {item['confidence']:.2f}"
    )

# Display original image without bounding boxes
cv2.imshow("E-Mirror - Detection Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
