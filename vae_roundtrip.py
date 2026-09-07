"""
vae_roundtrip.py
----------------
Step 1 of the "Cloak" project: verify that we can load Stable Diffusion's
VAE and run a full encode -> decode round trip on an image.

Background (explained simply)
-----------------------------
Stable Diffusion does not generate images directly. It works in a compressed
representation called "latent space". The VAE (Variational Autoencoder) is
the translator between the two worlds:

  * Encoder: pixel image (512x512x3)  ->  latent (64x64x4)
      It squeezes the image ~8x smaller in each spatial dimension.
  * Decoder: latent (64x64x4)         ->  pixel image (512x512x3)
      It expands the latent back into pixels.

Why this matters for Cloak: the PGD attack from PhotoGuard adds a tiny,
invisible perturbation in PIXEL space that strongly corrupts the LATENT
space. Before we can attack the encoder, we must prove this pipeline works:
encode the image, decode it back, and confirm the reconstruction looks
(almost) identical to the original.

The VAE is lossy, so the reconstruction will never be bit-for-bit identical
- small differences (~1-4 pixel values out of 255) are normal and expected.

Usage
-----
    python vae_roundtrip.py                    # uses inputs/sample.png
    python vae_roundtrip.py path/to/photo.jpg  # use your own image

Outputs (saved in ./output/)
----------------------------
    original.png       - the input image, resized to 512x512
    reconstructed.png  - the image after encode -> decode
    difference_x10.png - |original - reconstructed| amplified 10x, so the
                         tiny reconstruction error becomes visible. Later we
                         will use the same trick to verify that our
                         adversarial perturbation stays invisible.
"""

import math
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hugging Face repo id of the VAE used by Stable Diffusion v1.5.
# "sd-vae-ft-mse" is the VAE-only checkpoint fine-tuned for the best
# reconstruction quality - and it is a small (~335 MB) download, so we do
# not need the full Stable Diffusion model for this step.
VAE_ID = "stabilityai/sd-vae-ft-mse"

# Stable Diffusion is trained on 512x512 images.
IMAGE_SIZE = 512

# The VAE downsamples the image by a factor of 8. Any input size must be a
# multiple of 8 (512 is). We enforce 512x512 for simplicity.
assert IMAGE_SIZE % 8 == 0

OUTPUT_DIR = "output"
DEFAULT_INPUT = os.path.join("inputs", "sample.png")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image file and convert it to the tensor format the VAE expects.

    Steps:
      1. Open with PIL and force RGB (some images have an alpha channel).
      2. Resize to 512x512.
      3. Convert bytes [0, 255] to floats [0.0, 1.0].
      4. Rescale from [0, 1] to [-1, 1].
         The Stable Diffusion VAE was *trained* on images in [-1, 1],
         so we must match that range or the output will be garbage.
      5. Add a batch dimension: (C, H, W) -> (1, C, H, W).
         PyTorch models always expect a batch of images, even for one image.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert PIL image -> NumPy array -> PyTorch tensor, with values in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # shape (H, W, 3)
    x = torch.from_numpy(arr).permute(2, 0, 1)     # PyTorch wants (C, H, W)

    # Normalize [0, 1] -> [-1, 1]
    x = x * 2.0 - 1.0

    # Add batch dimension and move to the chosen device (CPU/GPU)
    return x.unsqueeze(0).to(device)


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    """
    Convert a VAE output tensor back into a PIL image.
    This is the exact inverse of load_image_as_tensor().
    """
    x = x.squeeze(0)                # remove batch dim: (1, C, H, W) -> (C, H, W)
    x = (x + 1.0) / 2.0             # undo normalization: [-1, 1] -> [0, 1]
    x = x.clamp(0.0, 1.0)           # the network can output slightly outside [0,1]
    x = (x * 255.0).round().byte()  # back to bytes [0, 255]
    x = x.permute(1, 2, 0)          # (C, H, W) -> (H, W, C) for PIL
    return Image.fromarray(x.cpu().numpy())


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) measures how similar two images are,
    in decibels. Higher is better:
      * ~30 dB  : visible differences if you look closely
      * ~40 dB+ : very hard to spot any difference
      * infinity: identical images
    We will use this later to prove our perturbations stay invisible.
    """
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    # Data range is 2.0 because our tensors live in [-1, 1]
    return 10 * math.log10((2.0 ** 2) / mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow overriding the input image from the command line
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input image not found: {input_path}\n"
            "Run 'python create_sample_image.py' first to create one."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pick the best available device. No GPU here -> CPU.
    # float16 is faster on GPU but unreliable on CPU, so use float32 there.
    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32
    print(f"Device: {device} (dtype={dtype})")

    # ---- 1. Load the VAE --------------------------------------------------
    # The first run downloads the weights (~335 MB) from Hugging Face and
    # caches them in your user profile folder. No API key is needed - this
    # model is public.
    print(f"Loading VAE: {VAE_ID} ...")
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()  # evaluation mode: disables training-only behaviors

    # ---- 2. Prepare the input image ----------------------------------------
    print(f"Loading image: {input_path}")
    original = load_image_as_tensor(input_path, device)

    # Save the (resized) original so we can compare it side-by-side later
    tensor_to_image(original).save(os.path.join(OUTPUT_DIR, "original.png"))

    # ---- 3. Encode -> latent space ------------------------------------------
    # torch.no_grad() tells PyTorch we are NOT training anything right now,
    # so it can skip storing intermediate values for backpropagation.
    # (Later, during the PGD attack, we WILL need gradients.)
    with torch.no_grad():
        # vae.encode() returns a distribution (mean + variance), because a
        # *variational* autoencoder models uncertainty. We take a sample from
        # it - this is what Stable Diffusion does too.
        posterior = vae.encode(original).latent_dist
        latents = posterior.sample()

        # Stable Diffusion multiplies latents by this constant (0.18215) so
        # their values have a smaller, nicer range for the diffusion model.
        latents = latents * vae.config.scaling_factor
        print(f"Latent shape: {tuple(latents.shape)} "
              f"(the image was compressed {IMAGE_SIZE // latents.shape[-1]}x "
              f"per side)")

        # ---- 4. Decode back to pixel space ---------------------------------
        # Undo the scaling, then decode the latent back into an image.
        decoded = vae.decode(latents / vae.config.scaling_factor).sample

    # ---- 5. Save results and report quality metrics -------------------------
    tensor_to_image(decoded).save(os.path.join(OUTPUT_DIR, "reconstructed.png"))

    # Amplified difference image: |original - reconstructed| x 10.
    # The raw difference is usually too subtle to see - amplifying it makes
    # the VAE's reconstruction error visible. We will reuse this technique
    # to show that our adversarial perturbations remain invisible.
    diff = (original - decoded).abs() * 10.0
    tensor_to_image(diff).save(os.path.join(OUTPUT_DIR, "difference_x10.png"))

    # Raw per-pixel difference in [-1, 1] space, converted to 0-255 scale.
    # (We clamp here because small overshoots are possible otherwise.)
    raw_diff_255 = (original - decoded).abs().clamp(0.0, 1.0) * 255.0
    max_diff_255 = raw_diff_255.max().item()
    mean_diff_255 = raw_diff_255.mean().item()
    psnr = compute_psnr(original, decoded)

    print("\n--- Round-trip results ---")
    print(f"Max pixel difference  : {max_diff_255:.1f} / 255")
    print(f"Mean pixel difference : {mean_diff_255:.2f} / 255")
    print(f"PSNR                  : {psnr:.1f} dB  (>= 30 dB is good)")
    print(f"\nSaved to '{OUTPUT_DIR}/': original.png, reconstructed.png, "
          f"difference_x10.png")

    # ---- 6. Simple pass/fail verdict ----------------------------------------
    # A healthy SD1.5 VAE reconstructs clean images with PSNR > ~25 dB.
    if psnr > 25.0:
        print("SUCCESS: encode/decode pipeline works correctly.")
    else:
        print("WARNING: reconstruction quality is unexpectedly low - "
              "check the input image and VAE weights.")


if __name__ == "__main__":
    main()
