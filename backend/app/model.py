import numpy as np
import cv2
from ultralytics import YOLO
import os

CLASS_ID_PRODUCT = 0
CLASS_ID_REFERENCE = 1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_tight_boxes.pt")

model = YOLO(MODEL_PATH)


def run_detection(image: np.ndarray):
    results = model.predict(
        image,
        conf=0.6,
        imgsz=1024,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    reference_candidates = []
    product_boxes = []

    for box, cls, conf in zip(boxes, classes, confidences):
        box = box.tolist()

        if int(cls) == CLASS_ID_REFERENCE:
            reference_candidates.append((conf, box))

        elif int(cls) == CLASS_ID_PRODUCT:
            product_boxes.append(box)

    if not reference_candidates:
        reference_box = None
    else:
        # pick highest confidence reference
        reference_box = max(reference_candidates, key=lambda x: x[0])[1]

    return reference_box, product_boxes
