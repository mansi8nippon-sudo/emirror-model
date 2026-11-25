
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


def create_tracker():
    """Initialize ByteTrack."""
    return ByteTrackTracker(
        lost_track_buffer=30,
        minimum_consecutive_frames=1
    )


def update_tracker(tracker, detections):
    """Convert NanoDet output into Supervision format and track it."""

    if detections:
        xyxy = np.array(
            [item["box"] for item in detections],
            dtype=np.float32
        )

        confidence = np.array(
            [item["confidence"] for item in detections],
            dtype=np.float32
        )

        class_ids = np.array(
            [item["class_id"] for item in detections],
            dtype=int
        )

    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confidence = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=int)

    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_ids
    )

    return tracker.update(sv_detections)
