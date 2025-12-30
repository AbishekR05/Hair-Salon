import numpy as np


class HairstyleRecommender:
    def __init__(self):
        self.rules = {
            "oval": ["quiff", "side_part", "pompadour"],
            "round": ["fade", "undercut", "textured_crop"],
            "square": ["crew_cut", "buzz", "short_quiff"],
            "heart": ["side_swept", "medium_length"]
        }

    def classify_face_shape(self, landmarks):
        # Landmark indices
        jaw_left = landmarks[234]
        jaw_right = landmarks[454]
        chin = landmarks[152]
        forehead = landmarks[10]
        cheek_left = landmarks[93]
        cheek_right = landmarks[323]

        jaw_width = abs(jaw_right.x - jaw_left.x)
        face_height = abs(chin.y - forehead.y)
        cheek_width = abs(cheek_right.x - cheek_left.x)

        ratio = face_height / jaw_width

        if ratio > 1.5:
            return "oval"
        elif cheek_width > jaw_width:
            return "round"
        elif jaw_width > cheek_width:
            return "square"
        else:
            return "heart"

    def recommend(self, landmarks):
        shape = self.classify_face_shape(landmarks)
        styles = self.rules.get(shape, [])

        return {
            "face_shape": shape,
            "recommended_styles": styles
        }
