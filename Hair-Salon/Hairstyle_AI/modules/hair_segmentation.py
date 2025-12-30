import cv2
import mediapipe as mp
import numpy as np

class HairSegmenter:
    def __init__(self):
        self.mp_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_segmentation.SelfieSegmentation(model_selection=1)

    def segment(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)

        if result.segmentation_mask is None:
            return None

        mask = result.segmentation_mask
        mask = (mask > 0.6).astype(np.uint8) * 255

        return mask
