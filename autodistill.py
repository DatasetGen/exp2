import base64
import json
import requests
import hashlib
import os
from PIL import Image
from io import BytesIO
from typing import Any, Dict

# Directory to store cached results
CACHE_DIR = ".segmentation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def hash_image(image: Image.Image) -> str:
    """Generate a SHA256 hash for a Pillow image."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return hashlib.sha256(buffered.getvalue()).hexdigest()

def cache_path(image_hash: str) -> str:
    """Return path to cached response file for an image hash."""
    return os.path.join(CACHE_DIR, f"{image_hash}.json")

def generate_segmentation(image: Image.Image) -> Dict[str, Any]:
    # Hash the image
    image_hash = hash_image(image)
    cache_file = cache_path(image_hash)

    # Return cached result if exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # Convert Pillow Image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # JSON payload
    payload = {
        "number_of_images": 1,
        "image": encoded_image,
        "annotation_model": "grounded_sam",
        "aspect_ratio": "string",
        "prompt": "string",
        "negative_prompt": "string",
        "strength": 1,
        "labels": [
            {
                "id": 0,
                "name": "sea",
                "prompt": "sea"
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://autodistill.autoescuelaseco.cloud/generate_images/image_variants/",
        headers=headers,
        data=json.dumps(payload)
    )
    response.raise_for_status()
    result = response.json()

    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


from PIL import ImageDraw

def draw_segmentation(image: Image.Image, normalized_points: list) -> Image.Image:
    """
    Draw a polygon segmentation on the image using normalized coordinates.

    Args:
        image (Image.Image): The input image.
        normalized_points (list): List of [x, y] points normalized by width and height.

    Returns:
        Image.Image: Image with the segmentation polygon drawn.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Denormalize the points
    points = [(x * width, y * height) for x, y in normalized_points]

    # Draw the polygon
    draw.polygon(points, outline="red", fill=(255, 0, 0, 80))  # Use fill with transparency if image supports alpha

    return image
