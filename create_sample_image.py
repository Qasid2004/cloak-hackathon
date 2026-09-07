"""
create_sample_image.py
----------------------
Prepares a sample photo for the VAE round-trip test.

Strategy:
  1. Try to download a real photo from Hugging Face (good stand-in for a
     real user photo).
  2. If there is no internet connection, generate a simple synthetic image
     instead so the pipeline can still be tested offline.

Output: inputs/sample.png  (512x512, RGB)
"""

import os

from PIL import Image, ImageDraw

# Where the sample image will be saved
INPUT_DIR = "inputs"
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample.png")

# A public test image hosted by Hugging Face (used in their docs)
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/diffusers/cat.png"
)


def download_sample_image(path: str) -> bool:
    """Download a real photo and save it. Returns True on success."""
    try:
        import requests

        response = requests.get(IMAGE_URL, timeout=30)
        response.raise_for_status()  # raise an error for bad HTTP responses

        # Normalize to RGB 512x512 (some downloads have alpha channels or
        # unusual sizes, which we don't want to worry about downstream)
        import io

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((512, 512))
        img.save(path)
        return True
    except Exception as exc:
        print(f"  Download failed ({exc}); will generate a synthetic image.")
        return False


def make_synthetic_image(path: str) -> None:
    """Create a simple colorful test image with shapes and gradients."""
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)

    # Horizontal color gradient background (blue -> orange)
    for x in range(512):
        t = x / 511.0  # 0.0 at left edge, 1.0 at right edge
        r = int(40 + 200 * t)
        g = int(80 + 60 * t)
        b = int(220 - 150 * t)
        draw.line([(x, 0), (x, 511)], fill=(r, g, b))

    # A few simple shapes so the image has edges and detail
    draw.ellipse([80, 80, 280, 280], fill=(255, 220, 90))
    draw.rectangle([300, 250, 460, 430], fill=(210, 60, 80))
    draw.polygon([(150, 450), (250, 320), (350, 450)], fill=(40, 180, 120))

    img.save(path)


def main() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)

    print("Preparing sample image...")
    if not download_sample_image(SAMPLE_PATH):
        make_synthetic_image(SAMPLE_PATH)

    # Sanity check: reopen and report basic info
    img = Image.open(SAMPLE_PATH)
    print(f"Saved {SAMPLE_PATH} | size={img.size} | mode={img.mode}")


if __name__ == "__main__":
    main()
