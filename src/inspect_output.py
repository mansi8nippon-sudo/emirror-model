
import cv2
import numpy as np
from detector import NanoDetDetector

detector = NanoDetDetector()

image = cv2.imread("videos/test_road.jpg")

if image is None:
    raise FileNotFoundError("test_road.jpg not found")

# Preprocess image
input_tensor, original_w, original_h = detector.preprocess(image)

# Run ONNX inference directly
output = detector.session.run(
    None,
    {detector.input_name: input_tensor}
)[0]

predictions = output[0]

print("\n----- RAW MODEL OUTPUT -----")
print("Output shape:", output.shape)
print("Minimum value:", np.min(predictions))
print("Maximum value:", np.max(predictions))
print("Mean value:", np.mean(predictions))

# Inspect class score values
class_scores = predictions[:, :80]

print("\n----- CLASS SCORE CHECK -----")
print("Minimum class score:", np.min(class_scores))
print("Maximum class score:", np.max(class_scores))

# Find the highest scores
top_indices = np.argsort(
    np.max(class_scores, axis=1)
)[-10:][::-1]

class_names = [
    "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck"
]

print("\n----- TOP 10 PREDICTIONS -----")

for index in top_indices:
    class_id = int(np.argmax(class_scores[index]))
    score = float(class_scores[index, class_id])

    name = (
        class_names[class_id]
        if class_id < len(class_names)
        else f"class_{class_id}"
    )

    print(
        f"Index: {index}, "
        f"Class: {name}, "
        f"ID: {class_id}, "
        f"Score: {score:.4f}"
    )
