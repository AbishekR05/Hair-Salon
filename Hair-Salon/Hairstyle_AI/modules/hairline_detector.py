import numpy as np
import cv2


class HairlineDetector:
    def __init__(self, samples=30):
        self.samples = samples  # number of points along forehead

    def detect(self, hair_mask, landmarks, frame_shape):
        """
        Returns a list of (x, y) points representing the hairline.
        """
        h, w = frame_shape[:2]
        hairline_points = []

        # Forehead region landmarks (temple to temple)
        left_temple = landmarks[234]
        right_temple = landmarks[454]
        forehead = landmarks[10]

        x_start = int(left_temple.x * w)

        x_end = int(right_temple.x * w)
        y_top = int(forehead.y * h)

        # Define a limited vertical scan band
        brow = landmarks[9]  # between eyebrows
        y_start = int(forehead.y * h)
        y_end = int(brow.y * h)

        if x_end <= x_start:
            return []

        # Sample vertical scan lines across forehead
        xs = np.linspace(x_start, x_end, self.samples).astype(int)

        for x in xs:
            # Scan downward from forehead within the band
            for y in range(y_start, y_end):
                if hair_mask[y, x] > 0:
                    # Bias upward to compensate conservative segmentation
                    corrected_y = max(0, y - int(0.02 * h))
                    hairline_points.append((x, corrected_y))
                    break

        return hairline_points
