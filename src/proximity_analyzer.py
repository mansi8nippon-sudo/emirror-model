
from collections import defaultdict, deque


class ProximityAnalyzer:

    PERSON_ID = 0
    VEHICLE_IDS = {2, 3, 5, 7}

    CLASS_NAMES = {
        0: "PERSON",
        2: "CAR",
        3: "MOTORCYCLE",
        5: "BUS",
        7: "TRUCK"
    }

    def __init__(
        self,
        person_near_threshold=0.35,
        vehicle_near_threshold=0.30,
        history_size=8,
        approach_threshold=20
    ):
        self.person_near_threshold = person_near_threshold
        self.vehicle_near_threshold = vehicle_near_threshold
        self.history_size = history_size
        self.approach_threshold = approach_threshold

        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(self, tracked, frame_height):

        results = []

        if frame_height <= 0:
            return results

        if tracked.class_id is None:
            return results

        for i in range(len(tracked)):

            class_id = int(tracked.class_id[i])

            # Only check persons and vehicles
            if class_id not in self.VEHICLE_IDS and class_id != self.PERSON_ID:
                continue

            x1, y1, x2, y2 = tracked.xyxy[i]

            box_height = max(0, y2 - y1)

            # Object height relative to complete camera frame
            height_ratio = box_height / frame_height

            # Use track ID when available
            if tracked.tracker_id is not None:
                track_id = tracked.tracker_id[i]

                if track_id is None or track_id < 0:
                    track_id = f"untracked_{i}"
            else:
                track_id = f"untracked_{i}"

            history = self.history[track_id]
            history.append(height_ratio)

            if class_id == self.PERSON_ID:
                near_threshold = self.person_near_threshold
            else:
                near_threshold = self.vehicle_near_threshold

            # Check whether object size is increasing
            approaching = False

            if len(history) >= self.history_size:
                old_size = history[0]

                if old_size > 0:
                    change_percent = (
                        (height_ratio - old_size) / old_size
                    ) * 100

                    approaching = (
                        change_percent >= self.approach_threshold
                    )
                else:
                    change_percent = 0
            else:
                change_percent = 0

            # Determine warning condition
            is_near = height_ratio >= near_threshold

            if is_near or (
                approaching and height_ratio >= 0.12
            ):

                object_name = self.CLASS_NAMES[class_id]

                results.append({
                    "track_id": track_id,
                    "class_name": object_name,
                    "height_ratio": round(height_ratio, 3),
                    "change_percent": round(change_percent, 2),
                    "motion": "APPROACHING" if approaching else "NEAR",
                    "warning": f"DANGER: {object_name} NEAR CAMERA"
                })

        return results
