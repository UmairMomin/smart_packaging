from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np

from app.model import run_detection

app = FastAPI(title="Smart Packaging Detection API")


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded")

    reference_box, product_boxes = run_detection(image)

    if reference_box is None:
        return {
            "reference_object": None,
            "products": product_boxes,
            "message": "Reference object not detected",
        }

    return {
        "reference_object": reference_box,
        "products": product_boxes,
        "success": True,
        "message": "Reference object detected",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/")
async def root():
    return {"message": "API is running"}
