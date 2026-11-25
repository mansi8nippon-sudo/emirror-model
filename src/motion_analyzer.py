
from collections import defaultdict, deque
import numpy as np


class VehicleMotionAnalyzer:

    def __init__(self):
        # Store the recent bounding-box areas for each tracking ID
        self.history = defaultdict(lambda: deque(maxlen=12))

    def update(self, tracked):

        results = []

        if tracked.tracker_id is None:
            return results

        for i, track_id in enumerate(tracked.tracker_id):

            if track_id is None or track_id < 0:
                continue

            x1, y1, x2, y2 = tracked.xyxy[i]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            if area <= 0:
                continue

            self.history[int(track_id)].append(float(area))

            values = list(self.history[int(track_id)])

            motion = "COLLECTING DATA"
            change_percent = 0.0

            # Wait for enough observations
            if len(values) >= 8:

                earlier = np.mean(values[:4])
                recent = np.mean(values[-4:])

                if earlier > 0:
                    change_percent = (
                        (recent - earlier) / earlier
                    ) * 100

                    if change_percent > 15:
                        motion = "SIZE INCREASING"

                    elif change_percent < -15:
                        motion = "SIZE DECREASING"

                    else:
                        motion = "SIZE STABLE"

            results.append({
                "track_id": int(track_id),
                "motion": motion,
                "change_percent": round(change_percent, 2)
            })

        return results
