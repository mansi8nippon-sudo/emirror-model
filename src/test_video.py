
import cv2

from detector import NanoDetDetector
from tracker import create_tracker, update_tracker
from motion_analyzer import VehicleMotionAnalyzer
from proximity_analyzer import ProximityAnalyzer
from overtaking_detector import OvertakingDetector


# Maximum video display size
MAX_DISPLAY_WIDTH = 850
MAX_DISPLAY_HEIGHT = 500


# Target object class IDs
# 0 = person
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

TARGET_CLASSES = {0, 2, 3, 5, 7}

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# Step 1: Load NanoDet model
detector = NanoDetDetector()


# Step 2: Initialize modules
tracker = create_tracker()
motion_analyzer = VehicleMotionAnalyzer()
proximity_analyzer = ProximityAnalyzer()
overtaking_detector = OvertakingDetector()


# Step 3: Open live camera
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# USB CAMERA CONFIGURATION

# CAMERA_INDEX = 1

# cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     raise RuntimeError("Could not open USB camera. Check connection.")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# print("USB Camera connected successfully!")

print("Live camera started. Press Q to exit.")
print("NanoDet + ByteTrack + Overtaking Detection started...")
print("Confidence threshold: 0.55")


frame_count = 0

try:

    while True:

        # Read camera frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera frame.")
            break

        frame_count += 1

        # Step 4: Detect objects
        detections = detector.detect(frame)

        # Step 5: Track objects
        tracked = update_tracker(tracker, detections)

        # Step 6: Overtaking / passing movement detection
        overtaking_results = overtaking_detector.update(
            tracked,
            frame.shape[1]
        )

        # Step 7: Check whether target objects are detected
        has_object = False

        if tracked.class_id is not None and len(tracked) > 0:

            has_object = any(
                int(class_id) in TARGET_CLASSES
                for class_id in tracked.class_id
            )

        # Step 8: Proximity and motion analysis
        proximity_results = proximity_analyzer.update(
            tracked,
            frame.shape[0]
        )

        motion_results = motion_analyzer.update(tracked)

        # Step 9: Generate warning
        if overtaking_results:

            warning_text = "OVERTAKING - DANGER"
            warning_active = True

        elif has_object:

            warning_text = "DANGER"



            warning_active = True

        else:

            warning_text = ""
            warning_active = False

        # Step 10: Resize frame while preserving aspect ratio
        original_height, original_width = frame.shape[:2]

        scale = min(
            MAX_DISPLAY_WIDTH / original_width,
            MAX_DISPLAY_HEIGHT / original_height
        )

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        display_frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Step 11: Display warning and full-screen red border
        if warning_active:

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            text_size = cv2.getTextSize(
                warning_text,
                font,
                font_scale,
                thickness
            )[0]

            text_x = (new_width - text_size[0]) // 2
            text_y = 40

            # Dark background behind warning
            cv2.rectangle(
                display_frame,
                (text_x - 10, 8),
                (text_x + text_size[0] + 10, 55),
                (30, 30, 30),
                -1
            )

            # Display warning text
            cv2.putText(
                display_frame,
                warning_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 255),
                thickness
            )

            # Full-screen red border
            cv2.rectangle(
                display_frame,
                (3, 3),
                (new_width - 4, new_height - 4),
                (0, 0, 255),
                7
            )

        # Step 12: Print information every 15 frames
        if frame_count % 15 == 0:

            print(f"\nFrame: {frame_count}")

            if has_object:

                for i in range(len(tracked)):

                    class_id = int(tracked.class_id[i])

                    if class_id not in TARGET_CLASSES:
                        continue

                    class_name = CLASS_NAMES.get(
                        class_id,
                        "unknown"
                    )

                    track_id = tracked.tracker_id[i]

                    print(
                        f"Object: {class_name} | "
                        f"Tracking ID: {track_id}"
                    )

            else:
                print("No tracked target objects")

            # Overtaking results
            if overtaking_results:

                for item in overtaking_results:

                    print(
                        f"OVERTAKING: {item['class_name']} | "
                        f"Direction: {item['direction']}"
                    )

            # Proximity analysis
            print("Proximity analysis:")

            for item in proximity_results:

                print(
                    f"ID: {item['track_id']} | "
                    f"Proximity: {item['motion']} | "
                    f"Size change: {item['change_percent']}%"
                )

            # Motion analysis
            print("Vehicle motion:")

            for item in motion_results:

                print(
                    f"ID: {item['track_id']} | "
                    f"Motion: {item['motion']} | "
                    f"Size change: {item['change_percent']}%"
                )

        # Step 13: Display camera output
        cv2.imshow(
            "E-Mirror - NanoDet + ByteTrack",
            display_frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("\nE-Mirror camera testing completed.")
