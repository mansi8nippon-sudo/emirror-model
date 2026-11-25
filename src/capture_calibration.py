
import cv2
import os

# Camera settings
CAMERA_INDEX = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Checkerboard inner corners
CHECKERBOARD = (9, 6)

SAVE_DIR = "calibration_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

count = 0

print("Checkerboard capture started")
print("Show the complete checkerboard to camera")
print("Press S to save an image")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    display = frame.copy()

    if found:
        cv2.drawChessboardCorners(
            display,
            CHECKERBOARD,
            corners,
            found
        )

        cv2.putText(
            display,
            "CHECKERBOARD DETECTED - PRESS S",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            display,
            "Show complete checkerboard",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imshow("Camera Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if found:
            count += 1

            filename = os.path.join(
                SAVE_DIR,
                f"calibration_{count:02d}.jpg"
            )

            cv2.imwrite(filename, frame)

            print("Saved:", filename)

        else:
            print("Checkerboard not detected. Try again.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total images saved: {count}")
