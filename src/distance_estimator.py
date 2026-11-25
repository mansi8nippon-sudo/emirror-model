
class DistanceEstimator:

    def __init__(self, focal_length_pixels, reference_width_m=1.8):
        self.focal_length = focal_length_pixels
        self.reference_width = reference_width_m

    def estimate(self, bbox):
        x1, y1, x2, y2 = bbox

        pixel_width = x2 - x1

        if pixel_width <= 0:
            return None

        distance = (
            self.focal_length * self.reference_width
        ) / pixel_width

        return round(distance, 2)
