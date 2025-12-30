import cv2
from modules.camera import Camera
from modules.face_mesh import FaceMeshDetector
from modules.hairstyle_renderer import HairstyleRenderer
from modules.hair_segmentation import HairSegmenter
from modules.hairline_detector import HairlineDetector
from modules.hairline_contour import HairlineContourDetector
from modules.hairstyle_recommender import HairstyleRecommender
from modules.hairstyle_library import HairstyleLibrary

def main():
    camera = Camera()
    face_mesh = FaceMeshDetector()
    renderer = HairstyleRenderer("assets/hairstyles/short_1.png")
    hair_segmenter = HairSegmenter()
    library = HairstyleLibrary()
    current_style = library.current()
    hairline_detector = HairlineDetector()
    hairline_contour_detector = HairlineContourDetector()
    recommender = HairstyleRecommender()
    current_style = None

    while True:
        frame = camera.read()
        if frame is None:
            break

        key = cv2.waitKey(1) & 0xFF
        style_changed = False
        if key == ord('n'):
            current_style = library.next()
            style_changed = True
        elif key == ord('p'):
            current_style = library.previous()
            style_changed = True
        if style_changed and current_style:
            renderer.set_hairstyle(f"assets/hairstyles/{current_style}.png")

        # Hair segmentation mask
        mask = hair_segmenter.segment(frame)
        if mask is not None:
            cv2.imshow("Hair Mask", mask)
            # Overlay mask on frame for debugging
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)
            cv2.imshow("Overlay", overlay)
            # Hide segmentation mask and overlay

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            rec = recommender.recommend(landmarks)

            # Update library with recommended styles
            library.set_styles(rec["recommended_styles"])

            # Ensure a style is selected
            if current_style is None:
                current_style = library.current()

            # Display face shape
            cv2.putText(
                frame,
                f"Face Shape: {rec['face_shape'].upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Display recommendations
            y_offset = 75
            cv2.putText(
                frame,
                "Recommended Styles:",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            for i, style in enumerate(rec["recommended_styles"]):
                cv2.putText(
                    frame,
                    f"- {style.replace('_', ' ').title()}",
                    (20, y_offset + 30 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )

                # Only show face shape, recommendations, and live camera
            frame = renderer.overlay(frame, landmarks)

        cv2.imshow("AI Hairstyle Prototype - Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
