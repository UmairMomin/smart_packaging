CREDIT_CARD_WIDTH_MM = 85.6
CREDIT_CARD_HEIGHT_MM = 53.98
COIN_DIAMETER_MM = 27.0

def compute_mm_per_pixel(reference_box, reference_type):
    x1, y1, x2, y2 = reference_box

    ref_w_px = x2 - x1
    ref_h_px = y2 - y1

    if reference_type == "credit_card":
        # auto orientation
        if ref_w_px >= ref_h_px:
            ref_real_mm = CREDIT_CARD_WIDTH_MM
            ref_pixel = ref_w_px
        else:
            ref_real_mm = CREDIT_CARD_HEIGHT_MM
            ref_pixel = ref_h_px

    elif reference_type == "coin":
        # coin is circular – take max dimension
        ref_real_mm = COIN_DIAMETER_MM
        ref_pixel = max(ref_w_px, ref_h_px)

    else:
        raise ValueError("Invalid reference type")

    if ref_pixel <= 0:
        raise ValueError("Invalid reference bounding box")

    return ref_real_mm / ref_pixel


def calculate_product_dimensions(product_boxes, mm_per_pixel):
    results = []

    for box in product_boxes:
        x1, y1, x2, y2 = box

        width_px = x2 - x1
        height_px = y2 - y1

        results.append({
            "width_mm": round(width_px * mm_per_pixel, 2),
            "height_mm": round(height_px * mm_per_pixel, 2)
        })

    return results
