
import cv2
import numpy as np
import onnxruntime as ort
import math

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "nanodet/nanodet.onnx"

INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.55
NMS_THRESHOLD = 0.40

STRIDES = [8, 16, 32, 64]
REG_MAX = 7
NUM_CLASSES = 80

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

MEAN = np.array(
    [103.53, 116.28, 123.675],
    dtype=np.float32
)

STD = np.array(
    [57.375, 57.12, 58.395],
    dtype=np.float32
)


class NanoDetDetector:

    def __init__(self):

        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        print("NanoDet-Plus-M loaded successfully!")

        # Check model output shape
        output_shape = self.session.get_outputs()[0].shape
        print("Model output shape:", output_shape)

        # Check generated prior count
        priors = self.generate_priors()
        print("Generated priors:", len(priors))

    def preprocess(self, image):

        original_h, original_w = image.shape[:2]

        resized = cv2.resize(
            image,
            (INPUT_SIZE, INPUT_SIZE)
        )

        normalized = (
            resized.astype(np.float32) - MEAN
        ) / STD

        input_tensor = normalized.transpose(2, 0, 1)

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0
        )

        return (
            input_tensor.astype(np.float32),
            original_w,
            original_h
        )

    # ---------------- GENERATE PRIORS ----------------

    def generate_priors(self):

        priors = []

        for stride in STRIDES:

            # IMPORTANT: Use ceil instead of floor division.
            # This produces 7x7 for stride 64.

            feature_h = math.ceil(INPUT_SIZE / stride)
            feature_w = math.ceil(INPUT_SIZE / stride)

            for y in range(feature_h):
                for x in range(feature_w):

                    priors.append([
                        x * stride,
                        y * stride,
                        stride
                    ])

        return np.array(
            priors,
            dtype=np.float32
        )

    # ---------------- DECODE ----------------

    def decode(self, output, original_w, original_h):

        predictions = output[0]

        priors = self.generate_priors()

        print(
            f"Predictions: {len(predictions)}, "
            f"Priors: {len(priors)}"
        )

        # Ensure prediction and prior counts match.
        if len(predictions) != len(priors):
            raise ValueError(
                "Prediction/Prior mismatch: "
                f"{len(predictions)} predictions, "
                f"{len(priors)} priors. "
                "Check model output and prior configuration."
            )

        # First 80 values are class probabilities.
        class_scores = predictions[:, :NUM_CLASSES]

        # Remaining 32 values are regression distributions.
        bbox_distributions = predictions[:, NUM_CLASSES:]

        class_ids = np.argmax(
            class_scores,
            axis=1
        )

        scores = class_scores[
            np.arange(len(class_scores)),
            class_ids
        ]

        detections = []

        for index, score in enumerate(scores):

            class_id = int(class_ids[index])

            if score < CONFIDENCE_THRESHOLD:
                continue

            if class_id not in TARGET_CLASSES:
                continue

            # Decode four distributions using DFL.
            reg = bbox_distributions[index].reshape(
                4,
                REG_MAX + 1
            )

            reg = reg - np.max(
                reg,
                axis=1,
                keepdims=True
            )

            probabilities = np.exp(reg)

            probabilities /= np.sum(
                probabilities,
                axis=1,
                keepdims=True
            )

            bins = np.arange(
                REG_MAX + 1,
                dtype=np.float32
            )

            distances = np.sum(
                probabilities * bins[None, :],
                axis=1
            )

            px, py, stride = priors[index]

            left, top, right, bottom = distances * stride

            x1 = (px - left) * original_w / INPUT_SIZE
            y1 = (py - top) * original_h / INPUT_SIZE

            x2 = (px + right) * original_w / INPUT_SIZE
            y2 = (py + bottom) * original_h / INPUT_SIZE

            x1 = np.clip(x1, 0, original_w - 1)
            y1 = np.clip(y1, 0, original_h - 1)

            x2 = np.clip(x2, 0, original_w - 1)
            y2 = np.clip(y2, 0, original_h - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "class_name": TARGET_CLASSES[class_id],
                "confidence": float(score),
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })

        return self.apply_nms(detections)

    # ---------------- NON-MAXIMUM SUPPRESSION ----------------

    def apply_nms(self, detections):

        final_detections = []

        # Apply NMS separately for each class.
        for class_id in TARGET_CLASSES:

            class_detections = [
                d for d in detections
                if d["class_id"] == class_id
            ]

            if not class_detections:
                continue

            boxes = []
            scores = []

            for d in class_detections:

                x1, y1, x2, y2 = d["box"]

                boxes.append([
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ])

                scores.append(d["confidence"])

            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD
            )

            if len(indices) > 0:

                for idx in np.array(indices).flatten():

                    final_detections.append(
                        class_detections[int(idx)]
                    )

        return final_detections

    # ---------------- DETECT ----------------

    def detect(self, image):

        input_tensor, original_w, original_h = (
            self.preprocess(image)
        )

        output = self.session.run(
            None,
            {self.input_name: input_tensor}
        )[0]

        return self.decode(
            output,
            original_w,
            original_h
        )
