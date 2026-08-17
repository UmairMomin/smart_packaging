import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "./models/best_tight_boxes.pt"
IMAGE_DIR = "./dataset/images"
CONF_THRESHOLD = 0.3

SAVE_PLOTS = True
OUTPUT_DIR = "plots"

REFERENCE_CLASS_ID = 1
PRODUCT_CLASS_ID = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------


# ---------------- IOU FUNCTION ----------------
def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU)
    Boxes format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0.0

    return inter_area / union
# ----------------------------------------------


# Load model
model = YOLO(MODEL_PATH)

iou_scores = []
processed_images = 0

# ---------------- MAIN LOOP ----------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    processed_images += 1
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    result = model.predict(
        image,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )[0]

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    ref_boxes = boxes[classes == REFERENCE_CLASS_ID]
    prod_boxes = boxes[classes == PRODUCT_CLASS_ID]

    # Compute IoU between reference and each product
    for ref in ref_boxes:
        for prod in prod_boxes:
            iou = compute_iou(ref, prod)
            iou_scores.append(iou)

# ---------------- VISUALIZATION ----------------
if iou_scores:
    plt.figure(figsize=(6, 4))
    plt.hist(iou_scores, bins=20)
    plt.xlabel("IoU Score")
    plt.ylabel("Count")
    plt.title("Bounding Box Localization Accuracy (IoU Histogram)")

    if SAVE_PLOTS:
        plt.savefig(f"{OUTPUT_DIR}/iou_histogram.png")

    plt.show()
else:
    print("No IoU scores generated. Check detections.")

# ---------------- SUMMARY ----------------
print("\n===== IoU ANALYTICS SUMMARY =====")
print(f"Images processed: {processed_images}")
print(f"Total IoU samples: {len(iou_scores)}")

if iou_scores:
    print(f"Mean IoU: {np.mean(iou_scores):.3f}")
    print(f"Median IoU: {np.median(iou_scores):.3f}")
    print(f"Min IoU: {np.min(iou_scores):.3f}")
    print(f"Max IoU: {np.max(iou_scores):.3f}")
