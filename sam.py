import requests
import base64
from PIL import Image
from io import BytesIO
from typing import List

def encode_image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def refine_bounding_box(
    img: Image.Image,
    bbox: List[int],
    segmentator: str = "sam2",
    model: str = "sam2_t.pt"
) -> dict:
    """
    Sends image and bounding box to a segmentation API and returns the result.

    Parameters:
        img (PIL.Image): The input image.
        bbox (List[int]): Bounding box in [x1, y1, x2, y2] format.
        segmentator (str): Segmentator identifier.
        model (str): Model filename.

    Returns:
        dict: JSON response from the segmentation API.
    """
    url = "https://sam.autoescuelaseco.cloud/segment_image/"
    headers = {"Content-Type": "application/json"}

    img_base64 = encode_image_to_base64(img)
    payload = {
        "image": img_base64,
        "segmentator": segmentator,
        "model": model,
        "bboxes": bbox,
        "points": []
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes

    return response.json()

from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

def filter_masks_by_polygon(
    patches: List[Image.Image],
    polygon: List[List[float]],
    threshold: float = 0.3
) -> List[Image.Image]:
    """
    Filters full-image-sized binary patch masks by overlap with a normalized polygon.

    Args:
        patches (List[PIL.Image]): List of binary masks (white patch area).
        polygon (List[List[float]]): List of normalized [x, y] points.
        threshold (float): Minimum overlap ratio to keep the patch.

    Returns:
        List[PIL.Image]: Filtered masks with enough polygon overlap.
    """
    if not patches:
        return []

    width, height = patches[0][0].size

    # Convert normalized polygon to absolute coordinates
    poly_pixels = [(int(x * width), int(y * height)) for x, y in polygon]

    # Create polygon mask over full image size
    poly_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(poly_mask).polygon(poly_pixels, fill=1)
    poly_mask_np = np.array(poly_mask)

    filtered = []

    for patch in patches:
        patch_np = np.array(patch[0]) > 0  # True where patch is white
        intersection = np.logical_and(patch_np, poly_mask_np).sum()
        patch_area = patch_np.sum()

        if patch_area == 0:
            continue

        overlap_ratio = intersection / patch_area

        if overlap_ratio >= threshold:
            filtered.append(patch)

    return filtered