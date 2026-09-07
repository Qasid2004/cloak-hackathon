"""
cloak_attack.py
---------------
Step 2 of the "Cloak" project: the actual protection attack.

We take the encode/decode pipeline verified in vae_roundtrip.py and add a
PGD (Projected Gradient Descent) loop on top of it. The result is a
"cloaked" image that:

  * looks (almost) identical to the original photo, but
  * is mapped to a very different point in latent space, which is what
    corrupts downstream Stable Diffusion operations (deepfakes, face swaps,
    AI edits) that must first encode the image with this same VAE.

This follows the "encoder attack" idea from the PhotoGuard paper
(Salman et al., 2023).

How PGD works (explained simply)
--------------------------------
Imagine the image as a point on a huge landscape. The "height" at each
point is our attack objective: how far the VAE latent of the image is from
the latent of the original image. We want to climb as high as possible
(gradient ASCENT, the opposite of the usual gradient descent used for
training).

Each PGD iteration does two things:
  1. STEP  : move the pixels a tiny bit in the direction that increases the
             latent distance (the sign of the gradient - the classic
             "sign trick" from the PGD paper, Madry et al. 2018).
  2. PROJECT: snap the pixels back inside an "epsilon ball" around the
             original image. The ball radius EPSILON is the maximum change
             allowed per pixel, which is what keeps the perturbation
             invisible to humans.

The name "Projected Gradient Descent" comes from this step+project
combination.

Usage
-----
    python cloak_attack.py                     # uses inputs/sample.png
    python cloak_attack.py path/to/photo.jpg   # cloak your own image

Outputs (saved in ./output/)
----------------------------
    cloaked.png        - the protected image
    difference_x10.png - |original - cloaked| amplified 10x, to visually
                         confirm the perturbation stays invisible
"""

import os
import sys

import cv2
import numpy as np
import torch

# Reuse the verified helpers and constants from step 1 instead of
# duplicating them. (Image loading is re-implemented locally below, because
# vae_roundtrip hard-codes 512x512 and we attack at a smaller size on this
# low-RAM machine - see ATTACK_IMAGE_SIZE.)
from vae_roundtrip import (
    DEFAULT_INPUT,
    OUTPUT_DIR,
    VAE_ID,
    compute_psnr,
    tensor_to_image,
)
from diffusers import AutoencoderKL
from PIL import Image

# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

# Maximum change allowed per pixel channel, on the familiar 0-255 scale.
# Applied ONLY inside the face box (see FACE_PAD_FRACTION). External A/B test:
# face eps=8 without dither broke img2img edits (edit-PSNR 23.7 dB) but showed
# a maze pattern; face eps=6 WITH dither looked like clean grain and disrupted
# edits equally (23.8 dB) yet left the edited cat still clearly recognizable.
# So we keep the dither AND go back to the stronger eps=8 budget here.
EPSILON_PIXELS = 8.0

# The perturbation is applied ONLY inside the (padded) face bounding box -
# even 8/255 noise over the whole image was noticeable, but the same budget
# confined to the face keeps the rest of the photo pixel-identical.
# FACE_PAD_FRACTION grows the detected box by this fraction on every side
# so ears/chin are covered too.
FACE_PAD_FRACTION = 0.2

# Per-step dither, in 0-255 scale. Pure PGD sign steps saturate pixels into
# a structured, maze-like +/-epsilon pattern that the eye picks up easily.
# Splashing a little uniform random noise into the update at EVERY step
# (before the epsilon clamp) breaks that structure so the final
# perturbation looks like natural photo grain. Still deterministic overall
# because torch.manual_seed(42) below fixes the noise draws.
NOISE_PIXELS = 1.0

# How many PGD iterations to run. More steps = stronger attack, but slower.
# 60 steps is plenty for the pixels to reach the epsilon boundary at the
# budgets we test (4-16 gray levels) with ALPHA_PIXELS=1.
NUM_STEPS = 60

# Resolution used for the attack. The VAE accepts any multiple of 8.
# 256 keeps the backward pass inside ~1 GB of RAM so the attack runs on a
# 4 GB laptop without swapping to disk. On a machine with >= 16 GB RAM (or
# a GPU), set this to 512 for full-resolution protection.
ATTACK_IMAGE_SIZE = 256

# Size of one PGD step, in 0-255 scale. A common rule of thumb is a step a
# few times smaller than epsilon; with the sign trick the pixels saturate
# at the epsilon boundary after a handful of steps anyway.
ALPHA_PIXELS = 1.0

# Our tensors live in [-1, 1], where the full 0-255 range maps to a width
# of 2.0. Convert the pixel-scale knobs into tensor scale.
EPSILON = EPSILON_PIXELS * 2.0 / 255.0
ALPHA = ALPHA_PIXELS * 2.0 / 255.0
NOISE = NOISE_PIXELS * 2.0 / 255.0

# Fixed seed so results are reproducible (the random start below).
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_attack_image(path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image exactly the same way vae_roundtrip.py does (RGB -> resize
    -> [0,1] -> [-1,1] -> add batch dim), but at ATTACK_IMAGE_SIZE instead
    of a hard-coded 512.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((ATTACK_IMAGE_SIZE, ATTACK_IMAGE_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0  # shape (H, W, 3)
    x = torch.from_numpy(arr).permute(2, 0, 1)     # PyTorch wants (C, H, W)
    x = x * 2.0 - 1.0                              # normalize to [-1, 1]
    return x.unsqueeze(0).to(device)


def find_face_mask(path: str, size: int, device: torch.device):
    """
    Detect the face with an OpenCV Haar cascade (bundled with opencv-python,
    no extra download) and return a 0/1 mask tensor that is 1 only inside
    the padded face box, plus the box coordinates for printing.

    We try a human-face cascade and two cat-face cascades (our sample photo
    is a cat) and keep the LARGEST box any of them reports - that is the
    dominant face in the photo.
    """
    img = np.array(Image.open(path).convert("RGB").resize((size, size)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cascades = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalcatface.xml",
        "haarcascade_frontalcatface_extended.xml",
    )
    best = None   # (area, name, x, y, w, h)
    for name in cascades:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        boxes = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(size // 8, size // 8),
        )
        for (x, y, w, h) in boxes:
            if best is None or w * h > best[0]:
                best = (w * h, name, x, y, w, h)
    if best is None:
        raise RuntimeError("No face found in the image - cannot cloak.")

    _, name, x, y, w, h = best
    print(f"Face detected with {name}: x={x}, y={y}, w={w}, h={h}")

    # Pad the box so the perturbation covers the whole head, then clip to
    # the image borders.
    px, py = int(w * FACE_PAD_FRACTION), int(h * FACE_PAD_FRACTION)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(size, x + w + px), min(size, y + h + py)

    mask = torch.zeros(1, 1, size, size, device=device)
    mask[:, :, y0:y1, x0:x1] = 1.0   # broadcasts over the 3 color channels
    return mask, (x0, y0, x1, y1)


def encode_to_latent(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    """
    Run the VAE encoder on an image tensor and return the latent.

    We use the posterior MEAN (not a random sample like vae_roundtrip.py
    did). A deterministic latent gives cleaner, more stable gradients for
    the attack, and it is exactly what the diffusion model uses at inference
    time when it wants a faithful encoding.
    """
    posterior = vae.encode(x).latent_dist
    # Same scaling constant Stable Diffusion uses (0.18215)
    return posterior.mean * vae.config.scaling_factor


def latent_distance(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute difference between two latents. This is our simple,
    readable measure of 'how different' two points in latent space are.
    """
    return (z_a - z_b).abs().mean()


def pgd_attack(
    vae: AutoencoderKL,
    x_orig: torch.Tensor,
    z_orig: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = NUM_STEPS,
) -> torch.Tensor:
    """
    Run the PGD loop and return the cloaked image tensor.

    We MAXIMIZE the distance between the cloaked latent and the original
    latent, i.e. gradient ascent on the latent distance. The `mask` confines
    every pixel change to the face region; outside it the image stays
    exactly the original.
    """
    # --- Random start ------------------------------------------------------
    # Starting from a small random perturbation (instead of the exact
    # original) helps PGD escape flat/symmetric spots where the gradient is
    # ~zero. The noise is inside the epsilon ball, so invisibility holds.
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-EPSILON, EPSILON) * mask
    x_adv = x_adv.clamp(-1.0, 1.0)
    x_adv = x_adv.detach().requires_grad_(True)

    for step in range(num_steps):
        # --- 1. Forward: encode the current candidate image ----------------
        z_adv = encode_to_latent(vae, x_adv)

        # Attack objective: squared L2 distance in latent space.
        # Bigger = latents more different = attack working better.
        distance = ((z_adv - z_orig) ** 2).mean()

        # --- 2. Backward: get the gradient w.r.t. the PIXELS ---------------
        # This is the magic of autograd: the gradient tells us, for every
        # pixel, whether increasing it would push the latent further away.
        distance.backward()

        with torch.no_grad():
            # --- 3. STEP: gradient ascent with the sign trick --------------
            # We only use the SIGN of each gradient (+1 / -1), not its size.
            # This makes every pixel move equally and is what makes PGD
            # attacks so effective.
            x_adv = x_adv + ALPHA * x_adv.grad.sign()

            # DITHER: a splash of random noise (face box only) so the pixels
            # do not lock into a geometric pattern - see NOISE_PIXELS.
            x_adv = x_adv + NOISE * torch.empty_like(x_adv).uniform_(-1.0, 1.0) * mask

            # --- 4. PROJECT back into the epsilon ball ---------------------
            # Clamp the *change* (not the image) to +/- EPSILON so we never
            # leave the invisible region around the original image...
            # ...multiply by the mask so pixels OUTSIDE the face box get
            # zero change (the rest of the photo stays pixel-identical)...
            x_adv = x_orig + (x_adv - x_orig).clamp(-EPSILON, EPSILON) * mask
            # ...and stay inside the valid image range [-1, 1].
            x_adv = x_adv.clamp(-1.0, 1.0)

        # Detach from the old computation graph and re-enable gradients for
        # the next iteration.
        x_adv = x_adv.detach().requires_grad_(True)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step + 1:3d}/{num_steps} | "
                  f"latent distance = {distance.item():.4f}", flush=True)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input image not found: {input_path}\n"
            "Run 'python create_sample_image.py' first to create one."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32
    print(f"Device: {device} (dtype={dtype})")

    # ---- 1. Load VAE and image (same as vae_roundtrip.py) ------------------
    print(f"Loading VAE: {VAE_ID} ...")
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()

    # Gradient checkpointing: recompute activations during backward instead
    # of storing them. The backward pass of the SD VAE at 512x512 otherwise
    # needs several GB of RAM - too much for a 4 GB laptop. Costs ~30% extra
    # compute time, which is a bargain here.
    vae.enable_gradient_checkpointing()

    print(f"Loading image: {input_path}")
    original = load_attack_image(input_path, device)

    # ---- 1b. Face mask: perturb ONLY the face region -------------------------
    mask, (x0, y0, x1, y1) = find_face_mask(input_path, ATTACK_IMAGE_SIZE, device)
    coverage = mask.mean().item() * 100.0
    print(f"Padded face box: x[{x0}:{x1}], y[{y0}:{y1}] -> "
          f"{coverage:.0f}% of the pixels will be perturbed")

    # ---- 2. Original latent (fixed target of the attack) -------------------
    # No gradients needed here - this is the reference point we want the
    # cloaked image to run away from.
    with torch.no_grad():
        z_orig = encode_to_latent(vae, original)

    # ---- 3. Run the PGD attack ----------------------------------------------
    print(f"Running PGD attack ({NUM_STEPS} steps, "
          f"epsilon={EPSILON_PIXELS}/255 per pixel, face region only)...")
    cloaked = pgd_attack(vae, original, z_orig, mask)

    # ---- 4. Measure attack strength and invisibility ------------------------
    with torch.no_grad():
        z_cloaked = encode_to_latent(vae, cloaked)

    # Latent distance BEFORE the attack: encode the random-start image
    # (original + tiny noise inside the face box). This shows how far the
    # noise alone gets us, so the 'after' number has a fair baseline to beat.
    noise_start = (original + torch.empty_like(original)
                   .uniform_(-EPSILON, EPSILON) * mask).clamp(-1.0, 1.0)
    with torch.no_grad():
        z_start = encode_to_latent(vae, noise_start)

    dist_before = latent_distance(z_start, z_orig).item()
    dist_after = latent_distance(z_cloaked, z_orig).item()

    # Pixel-space invisibility metrics
    psnr = compute_psnr(original, cloaked)
    # Tensor range [-1, 1] (width 2.0) maps to 255 gray levels, so the
    # tensor->255-scale conversion factor is 255 / 2 = 127.5.
    max_change_255 = ((cloaked - original).abs().max() * 127.5).item()
    # Proof that nothing outside the face box was touched (must be 0).
    outside_max_255 = ((cloaked - original).abs() * (1.0 - mask)).max().item() * 127.5

    # ---- 5. Save outputs -----------------------------------------------------
    tensor_to_image(cloaked).save(os.path.join(OUTPUT_DIR, "cloaked.png"))

    # Amplified perturbation map, same technique as vae_roundtrip.py
    diff = (original - cloaked).abs() * 10.0
    tensor_to_image(diff).save(os.path.join(OUTPUT_DIR, "difference_x10.png"))

    print("\n--- Attack results ---")
    print(f"Latent distance before attack : {dist_before:.4f}")
    print(f"Latent distance after attack  : {dist_after:.4f} "
          f"({dist_after / max(dist_before, 1e-8):.1f}x increase)")
    print(f"Max pixel change              : {max_change_255:.1f} / 255 "
          f"(budget was {EPSILON_PIXELS:.0f})")
    print(f"Max change OUTSIDE face box   : {outside_max_255:.1f} / 255 "
          f"(must be 0)")
    print(f"PSNR original vs cloaked      : {psnr:.1f} dB  (must be > 25)")
    print(f"\nSaved to '{OUTPUT_DIR}/': cloaked.png, difference_x10.png")

    # ---- 6. Pass/fail verdicts -------------------------------------------------
    ok_invisible = psnr > 25.0
    ok_attack = dist_after > dist_before * 2.0  # attack clearly beat the noise
    if ok_invisible and ok_attack:
        print("SUCCESS: cloaked image is invisible AND latent is strongly "
              "perturbed.")
    else:
        if not ok_invisible:
            print("WARNING: perturbation is too visible - lower EPSILON_PIXELS.")
        if not ok_attack:
            print("WARNING: attack too weak - try more steps or higher epsilon.")


if __name__ == "__main__":
    main()
