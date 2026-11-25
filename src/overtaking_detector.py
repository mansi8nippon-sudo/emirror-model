
from collections import defaultdict, deque


class OvertakingDetector:

    TARGET_CLASSES = {
        0: "PERSON",
        2: "CAR",
        3: "MOTORCYCLE",
        5: "BUS",
        7: "TRUCK"
    }

    def __init__(
        self,
        history_size=12,
        left_zone=0.35,
        right_zone=0.65,
        min_frames=8,
        alert_duration=45
    ):
        self.history_size = history_size
        self.left_zone = left_zone
        self.right_zone = right_zone
        self.min_frames = min_frames
        self.alert_duration = alert_duration

        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        self.alerted_ids = set()
        self.last_seen = {}
        self.active_alerts = {}
        self.frame_count = 0

    def update(self, tracked, frame_width):

        self.frame_count += 1

        if frame_width <= 0:
            return []

        if tracked.class_id is None:
            return self._get_active_alerts()

        for i in range(len(tracked)):

            class_id = int(tracked.class_id[i])

            # Ignore classes other than person and vehicles
            if class_id not in self.TARGET_CLASSES:
                continue

            track_id = tracked.tracker_id[i]

            if track_id is None or int(track_id) < 0:
                continue

            track_id = int(track_id)

            x1, y1, x2, y2 = tracked.xyxy[i]

            center_x = ((x1 + x2) / 2) / frame_width

            self.last_seen[track_id] = self.frame_count

            positions = self.history[track_id]
            positions.append(center_x)

            if len(positions) < self.min_frames:
                continue

            if track_id in self.alerted_ids:
                continue

            first_position = positions[0]
            current_position = positions[-1]

            direction = None

            # Movement from left side to right side
            if (
                first_position <= self.left_zone
                and current_position >= self.right_zone
            ):
                direction = "LEFT TO RIGHT"

            # Movement from right side to left side
            elif (
                first_position >= self.right_zone
                and current_position <= self.left_zone
            ):
                direction = "RIGHT TO LEFT"

            if direction:

                object_name = self.TARGET_CLASSES[class_id]

                self.alerted_ids.add(track_id)

                # Keep warning visible for some frames
                self.active_alerts[track_id] = {
                    "class_name": object_name,
                    "direction": direction,
                    "warning": "OVERTAKING - DANGER",
                    "expires_at": (
                        self.frame_count + self.alert_duration
                    )
                }

        # Remove old tracking histories
        expired_tracks = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if self.frame_count - last_frame > 30
        ]

        for track_id in expired_tracks:
            self.history.pop(track_id, None)
            self.last_seen.pop(track_id, None)
            self.alerted_ids.discard(track_id)

        return self._get_active_alerts()

    def _get_active_alerts(self):

        # Remove expired warnings
        expired_alerts = [
            track_id
            for track_id, alert in self.active_alerts.items()
            if self.frame_count > alert["expires_at"]
        ]

        for track_id in expired_alerts:
            self.active_alerts.pop(track_id, None)

        return list(self.active_alerts.values())
