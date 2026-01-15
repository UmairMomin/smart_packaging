import numpy as np
import cv2
from ultralytics import YOLO
import os

CLASS_ID_PRODUCT = 0
CLASS_ID_REFERENCE = 1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_tight_boxes.pt")

model = YOLO(MODEL_PATH)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "result.png")


def run_detection(image: np.ndarray):
    original_image = image.copy()

    results = model.predict(
        image,
        conf=0.6,
        imgsz=640,
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

    reference_box = None
    if reference_candidates:
        reference_box = max(reference_candidates, key=lambda x: x[0])[1]

    # --------------------------------------------------
    # DRAW BOUNDING BOXES
    # --------------------------------------------------

    if reference_box:
        x1, y1, x2, y2 = map(int, reference_box)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            "REFERENCE",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    for i, box in enumerate(product_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            original_image,
            f"PRODUCT {i+1}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------
    # SAVE IMAGE (OVERWRITE)
    # --------------------------------------------------

    cv2.imwrite(OUTPUT_IMAGE_PATH, original_image)

    return reference_box, product_boxes

