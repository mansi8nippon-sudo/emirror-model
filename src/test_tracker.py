
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

# Initialize ByteTrack
tracker = ByteTrackTracker(
    lost_track_buffer=30,
    minimum_consecutive_frames=1
)

# Simulate one car detection in consecutive frames
for frame_no in range(1, 6):

    detections = sv.Detections(
        xyxy=np.array([
            [100 + frame_no * 5, 100, 200 + frame_no * 5, 200]
        ], dtype=np.float32),
        confidence=np.array([0.90], dtype=np.float32),
        class_id=np.array([2], dtype=int)
    )

    tracked = tracker.update(detections)

    print(
        f"Frame: {frame_no} | "
        f"Detections: {len(tracked)} | "
        f"Tracker IDs: {tracked.tracker_id}"
    )

print("\nByteTrack test completed.")
