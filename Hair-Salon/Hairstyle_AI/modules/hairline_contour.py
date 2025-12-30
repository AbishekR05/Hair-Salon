import cv2
import numpy as np

class HairlineContourDetector:
    def __init__(self, smooth_kernel=7):
        self.smooth_kernel = smooth_kernel

    def detect(self, hair_mask, landmarks, frame_shape):
        h, w = frame_shape[:2]
        contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Forehead region: temples and a bit below brow
        left_temple = landmarks[234]
        right_temple = landmarks[454]
        forehead = landmarks[10]
        brow = landmarks[9]
        brow = landmarks[9]

        x_start = int(left_temple.x * w)
        x_end = int(right_temple.x * w)
        # Expand region: from top of forehead to just above the mouth
        y_top = int(forehead.y * h - 0.10 * h)
        y_bottom = int(landmarks[13].y * h)  # landmark 13 is near the upper lip

        region_points = []
        for cnt in contours:
            for pt in cnt:
                x, y = pt[0]
                if x_start <= x <= x_end and y_top <= y <= y_bottom:
                    region_points.append((x, y))

        if not region_points:
            return []

        region_points = np.array(region_points)
        hairline = []
        for x in range(x_start, x_end + 1):
            ys = region_points[region_points[:, 0] == x][:, 1]
            if len(ys) > 0:
                y = ys.max()  # use max for lowest point
                hairline.append((x, y))

        # Optional: smooth the curve
        if len(hairline) > self.smooth_kernel:
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            xs = np.convolve([pt[0] for pt in hairline], kernel, mode='same')
            ys = np.convolve([pt[1] for pt in hairline], kernel, mode='same')
            hairline = list(zip(xs.astype(int), ys.astype(int)))
        else:
            hairline = [tuple(map(int, pt)) for pt in hairline]

        return hairline