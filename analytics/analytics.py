import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "./models/best_tight_boxes.pt"
IMAGE_DIR = "./dataset/images"
CONF_THRESHOLD = 0.25
SAVE_PLOTS = True
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------

model = YOLO(MODEL_PATH)

all_confidences = []
bbox_areas = []
detections_per_image = []

reference_detected = 0
product_detected = 0
total_images = 0

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    total_images += 1
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    results = model.predict(
        image,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )[0]

    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
    classes = results.boxes.cls.cpu().numpy() if results.boxes else []
    confidences = results.boxes.conf.cpu().numpy() if results.boxes else []

    detections_per_image.append(len(boxes))

    for box, cls, conf in zip(boxes, classes, confidences):
        all_confidences.append(conf)

        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        bbox_areas.append(area)

        if int(cls) == 1:
            reference_detected += 1
        elif int(cls) == 0:
            product_detected += 1

# ---------------- VISUAL ANALYTICS ----------------

# 1️⃣ Confidence Distribution
plt.figure(figsize=(6,4))
plt.hist(all_confidences, bins=20)
plt.xlabel("Confidence Score")
plt.ylabel("Number of Detections")
plt.title("Detection Confidence Distribution")

if SAVE_PLOTS:
    plt.savefig(f"{OUTPUT_DIR}/confidence_distribution.png")
plt.show()


# 2️⃣ Detections per Image
plt.figure(figsize=(6,4))
plt.hist(detections_per_image, bins=15)
plt.xlabel("Detections per Image")
plt.ylabel("Image Count")
plt.title("Detection Count per Image")

if SAVE_PLOTS:
    plt.savefig(f"{OUTPUT_DIR}/detections_per_image.png")
plt.show()


# 3️⃣ Bounding Box Area Distribution
plt.figure(figsize=(6,4))
plt.hist(bbox_areas, bins=20)
plt.xlabel("Bounding Box Area (pixels)")
plt.ylabel("Count")
plt.title("Bounding Box Size Distribution")

if SAVE_PLOTS:
    plt.savefig(f"{OUTPUT_DIR}/bbox_area_distribution.png")
plt.show()


# 4️⃣ Detection Summary
labels = ["Reference Detected", "Product Detected", "No Detection"]
values = [
    reference_detected,
    product_detected,
    max(0, total_images - (reference_detected + product_detected))
]

plt.figure(figsize=(5,5))
plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Detection Coverage Summary")

if SAVE_PLOTS:
    plt.savefig(f"{OUTPUT_DIR}/detection_summary.png")
plt.show()

print("\n===== ANALYTICS SUMMARY =====")
print(f"Total Images: {total_images}")
print(f"Reference Detections: {reference_detected}")
print(f"Product Detections: {product_detected}")
print(f"Avg Detections/Image: {np.mean(detections_per_image):.2f}")
