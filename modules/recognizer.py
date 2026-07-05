from __future__ import annotations

import cv2
import numpy as np
import os
import json
from typing import Dict, Optional, Tuple


class FaceRecognizer:
    CONFIDENCE_THRESHOLD = 85  # LBPH: lower = better match; >85 = reject

    def __init__(self, model_dir: str):
        try:
            self._model = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8
            )
        except AttributeError:
            raise ImportError(
                "opencv-contrib-python é necessário.\n"
                "Execute: pip install opencv-contrib-python"
            )

        self._model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._model_path = os.path.join(model_dir, "face_model.yml")
        self._labels_path = os.path.join(model_dir, "labels.json")
        self._label_map: Dict[int, str] = {}  # int id -> display name

        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self._model_path) and os.path.exists(self._labels_path):
            self._model.read(self._model_path)
            with open(self._labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._label_map = {int(k): v for k, v in raw.items()}

    def has_model(self) -> bool:
        return bool(self._label_map)

    # ------------------------------------------------------------------
    def train(self, data_dir: str):
        """
        Scan data_dir for sub-folders (one per user).
        Each .jpg/.png inside is a 200×200 grayscale face crop.
        """
        faces, labels, name_to_id = [], [], {}

        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            user_id = len(name_to_id)
            name_to_id[folder_name] = user_id

            for img_file in os.listdir(folder_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(user_id)

        if not faces:
            raise RuntimeError("Nenhuma imagem de rosto encontrada para treinamento.")

        self._model.train(faces, np.array(labels, dtype=np.int32))
        self._model.save(self._model_path)

        # Build display name map: folder_slug → "Proper Name"
        self._label_map = {
            uid: folder.replace("_", " ").title()
            for folder, uid in name_to_id.items()
        }
        with open(self._labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._label_map.items()}, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    def predict(self, face_gray_200: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (display_name, confidence) or (None, 999) when no model is loaded.
        Caller decides acceptance: confidence < CONFIDENCE_THRESHOLD → granted.
        """
        if not self.has_model():
            return None, 999.0

        try:
            label, confidence = self._model.predict(face_gray_200)
            name = self._label_map.get(label, "Desconhecido")
            return name, float(confidence)
        except Exception:
            return None, 999.0
