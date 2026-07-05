import cv2
import os

# Classifiers shipped with this project (OpenCV 5 no longer bundles them)
_CLASSIFIERS_DIR = os.path.join(os.path.dirname(__file__), "..", "classifiers")


def _cascade_path(filename: str) -> str:
    local = os.path.abspath(os.path.join(_CLASSIFIERS_DIR, filename))
    if os.path.exists(local):
        return local
    # Fallback: try cv2.data (OpenCV 4 and older)
    bundled = os.path.join(cv2.data.haarcascades, filename)
    if os.path.exists(bundled):
        return bundled
    raise FileNotFoundError(f"Classificador não encontrado: {filename}")


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_eye.xml"))

        if self.face_cascade.empty():
            raise RuntimeError("Classificador de faces inválido.")
        if self.eye_cascade.empty():
            raise RuntimeError("Classificador de olhos inválido.")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces, gray

    def detect_eyes(self, face_roi_gray):
        # Look only in the upper 65% of face region (where eyes are)
        h = face_roi_gray.shape[0]
        upper_roi = face_roi_gray[: int(h * 0.65), :]

        eyes = self.eye_cascade.detectMultiScale(
            upper_roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(20, 20),
        )
        return eyes
