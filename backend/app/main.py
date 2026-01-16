from fastapi import FastAPI, File, UploadFile, HTTPException, Body
import cv2
import numpy as np

from app.model import run_detection
from app.calc import compute_mm_per_pixel, calculate_product_dimensions

app = FastAPI(title="Smart Packaging Detection API")


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file", success=False)

    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400, detail="Image could not be decoded", success=False
        )

    reference_box, product_boxes = run_detection(image)

    if reference_box is None:
        return {
            "reference_object": None,
            "products": product_boxes,
            "message": "Reference object not detected",
            "success": False,
        }

    return {
        "reference_object": reference_box,
        "products": product_boxes,
        "success": True,
        "message": "Reference object detected",
    }


@app.post("/calculate-dimensions")
def calculate_dimensions(payload: dict = Body(...)):
    try:
        reference_box = payload["reference_object"]
        product_boxes = payload["products"]
        reference_type = payload.get("reference_type", "credit_card")

        mm_per_pixel = compute_mm_per_pixel(reference_box, reference_type)

        dimensions = calculate_product_dimensions(product_boxes, mm_per_pixel)

        return {
            "reference_type": reference_type,
            "scale_mm_per_pixel": round(mm_per_pixel, 4),
            "dimensions_mm": dimensions,
            "success": True,
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/")
async def root():
    return {"message": "API is running"}
