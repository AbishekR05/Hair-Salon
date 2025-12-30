import cv2
import mediapipe as mp
import numpy as np
import math


class FaceMeshDetector:
    def __init__(self, smoothing=0.85):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Generic 3D face model
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),   # Chin
            (-43.3, 32.7, -26.0),  # Left eye corner
            (43.3, 32.7, -26.0),   # Right eye corner
            (-28.9, -28.9, -24.1), # Left mouth
            (28.9, -28.9, -24.1)   # Right mouth
        ])

        # Smoothing
        self.smoothing = smoothing
        self.prev_pose = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def estimate_head_pose(self, frame, landmarks):
        h, w, _ = frame.shape

        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rvec)

        sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = 0

        pose = {
            "yaw": math.degrees(yaw),
            "pitch": math.degrees(pitch),
            "roll": math.degrees(roll)
        }

        return self._normalize_and_smooth(pose)

    # -------------------------------
    # NORMALIZATION + SMOOTHING
    # -------------------------------
    def _normalize_and_smooth(self, pose):
        # Wrap angles to [-180, 180]
        for k in pose:
            pose[k] = (pose[k] + 180) % 360 - 180

        # Clamp pitch to human-usable range
        pose["pitch"] = max(-45, min(45, pose["pitch"]))

        # First frame
        if self.prev_pose is None:
            self.prev_pose = pose
            return pose

        # Exponential Moving Average smoothing
        smoothed = {}
        for k in pose:
            smoothed[k] = (
                self.smoothing * self.prev_pose[k]
                + (1 - self.smoothing) * pose[k]
            )

        self.prev_pose = smoothed
        return smoothed
