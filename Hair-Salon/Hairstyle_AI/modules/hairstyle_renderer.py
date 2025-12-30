
import os
import cv2
import numpy as np



class HairstyleRenderer:
    def set_hairstyle(self, hairstyle_path):
        import os
        base_dir = os.path.dirname(__file__)
        root_dir = os.path.abspath(os.path.join(base_dir, ".."))
        hairstyle_path = os.path.join(root_dir, hairstyle_path)
        self.hair = cv2.imread(hairstyle_path, cv2.IMREAD_UNCHANGED)

    def __init__(self, hairstyle_path):
        base_dir = os.path.dirname(__file__)  # modules/
        root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Hairstyle_AI
        hairstyle_path = os.path.join(root_dir, hairstyle_path)

        if not os.path.exists(hairstyle_path):
            raise ValueError(f"Hairstyle image not found: {hairstyle_path}")

        self.hair = cv2.imread(hairstyle_path, cv2.IMREAD_UNCHANGED)

        # Must be RGBA
        if self.hair is None or len(self.hair.shape) != 3 or self.hair.shape[2] != 4:
            raise ValueError(
                "Hairstyle image must be a PNG with transparency (RGBA, 4 channels)"
            )
        self.hairstyle_path = hairstyle_path

        # Cache original size
        self.orig_h, self.orig_w = self.hair.shape[:2]


    def overlay(self, frame, landmarks, pose=None):
        h, w, _ = frame.shape


        # Use more landmarks for better fit
        left = landmarks[234]      # left temple
        right = landmarks[454]     # right temple
        skull = landmarks[10]      # top of head
        forehead = landmarks[9]    # forehead (for skin color and alignment)

        left_x = int(left.x * w)
        right_x = int(right.x * w)
        skull_y = int(skull.y * h)
        forehead_y = int(forehead.y * h)

        face_width = abs(right_x - left_x)
        # Height from skull to forehead, with extra padding
        face_height = int((forehead_y - skull_y) * 1.5)

        # --- Estimate average skin color from forehead region ---
        sample_size = max(5, int(face_width * 0.08))
        fx = int(forehead.x * w)
        fy = int(forehead.y * h)
        x1s = max(0, fx - sample_size // 2)
        y1s = max(0, fy - sample_size // 2)
        x2s = min(w, fx + sample_size // 2)
        y2s = min(h, fy + sample_size // 2)
        skin_patch = frame[y1s:y2s, x1s:x2s]
        if skin_patch.size > 0:
            avg_skin_color = skin_patch.mean(axis=(0, 1)).astype(np.uint8)
            avg_skin_color = tuple(int(c) for c in avg_skin_color)
        else:
            avg_skin_color = (180, 160, 140)  # fallback BGR

        # --- Hair scaling and overlay (cover real hair area) ---
        hair_width = int(face_width * 1.3)
        hair_height = max(1, face_height)


        resized_hair = cv2.resize(
            self.hair, (hair_width, hair_height),
            interpolation=cv2.INTER_AREA
        )

        # ---- YAW ROTATION ----
        if pose is not None:
            yaw = pose.get("yaw", 0.0)
            # Clamp rotation for realism
            yaw = max(-30, min(30, yaw))
            # Rotate for visual correctness
            resized_hair = self._rotate_image(resized_hair, yaw)

            # ---- YAW-BASED HORIZONTAL SHIFT ----
            # Hair shifts toward the turning direction
            shift_x = int((yaw / 30.0) * face_width * 0.15)

        # Place hair: top at skull, bottom at forehead (with padding)
        # Use anchor_x for more generality if needed, but here we keep left_x logic
        if pose is not None:
            x1 = left_x - int(0.15 * hair_width) + shift_x
        else:
            x1 = left_x - int(0.15 * hair_width)
        y1 = skull_y - int(0.3 * hair_height)

        # --- Use hair alpha as mask to fill only hair region with skin color ---
        hair_alpha = resized_hair[:, :, 3] / 255.0
        for y in range(resized_hair.shape[0]):
            fy = y + y1
            if fy < 0 or fy >= frame.shape[0]:
                continue
            for x in range(resized_hair.shape[1]):
                fx = x + x1
                if fx < 0 or fx >= frame.shape[1]:
                    continue
                alpha = hair_alpha[y, x]
                if alpha > 0.1:  # Only where hair will be drawn
                    frame[fy, fx] = avg_skin_color

        # --- Overlay hair on top ---
        self._alpha_blend(frame, resized_hair, x1, y1)
        return frame
    
    def _alpha_blend(self, frame, hair, x_offset, y_offset):
        """
        Alpha blends RGBA hair image onto BGR frame.
        """
        h, w = hair.shape[:2]

        for y in range(h):
            fy = y + y_offset
            if fy < 0 or fy >= frame.shape[0]:
                continue

            for x in range(w):
                fx = x + x_offset
                if fx < 0 or fx >= frame.shape[1]:
                    continue

                alpha = hair[y, x, 3] / 255.0
                if alpha == 0:
                    continue

                frame[fy, fx] = (
                    alpha * hair[y, x, :3]
                    + (1 - alpha) * frame[fy, fx]
                )
    
    def _rotate_image(self, image, angle):
        """
        Rotates RGBA image around its center.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # keep transparency
        )
        return rotated

