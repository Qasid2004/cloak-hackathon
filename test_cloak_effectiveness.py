"""
test_cloak_effectiveness.py
---------------------------
Step 3 of the "Cloak" project: the hackathon demo proof.

We feed BOTH the clean image and our cloaked image through a real Stable
Diffusion img2img pipeline with the same edit prompt and the same random
seed. Expected outcome:

  * clean image   -> a coherent, recognizable edited image
  * cloaked image -> broken / distorted / nonsensical output

because our adversarial perturbation pushed the image's latent far away
from where the diffusion model expects it to be.

Backend: OpenVINO int8-quantized SD 1.5
---------------------------------------
This laptop has no GPU and only 4 GB RAM, where the regular PyTorch SD 1.5
model (2 GB in float16, 3.4 GB in float32) swap-thrashes and never finishes.
So we use the OpenVINO port of SD 1.5 with int8-quantized weights
(~1 GB) and CPU-optimized kernels - the standard way to run Stable Diffusion
on a weak CPU machine. It is the same SD 1.5 model, just compressed.

NOTE: run this script with the OpenVINO environment, not the main one:
    .venv-ov\\Scripts\\python.exe test_cloak_effectiveness.py

Fairness: both runs use the SAME denoising seed, so any difference between
the two outputs is caused by the cloak, not by random chance.

Inputs  : output/original.png, output/cloaked.png (from earlier scripts)
Outputs : output/edit_original.png, output/edit_cloaked.png and
          output/comparison_side_by_side.png (a labeled 2x2 grid)
"""

import gc
import os
import time

import numpy as np
import torch
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OpenVINO int8-quantized port of Stable Diffusion v1.5 (public, no key).
SD_REPO = "OpenVINO/stable-diffusion-v1-5-int8-ov"

# The edit we ask the (ab)user's pipeline to perform. Our sample photo is a
# cat, so we use a cat-appropriate edit prompt.
PROMPT = "a photo of a cat wearing a red party hat"

# img2img strength: how much the output may deviate from the input.
# 0.8 = a strong edit, which makes the demo difference easy to see.
STRENGTH = 0.8

# Few denoising steps: enough for a coherent edit, fast on CPU.
NUM_STEPS = 10

# Same seed for both runs -> fair comparison (see docstring).
SEED = 1234

# The cloaked image was produced at 256x256, so we test both images at that
# size for an apples-to-apples comparison.
TEST_SIZE = 256

OUTPUT_DIR = "output"
ORIGINAL_PATH = os.path.join(OUTPUT_DIR, "original.png")
CLOAKED_PATH = os.path.join(OUTPUT_DIR, "cloaked.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_image(path: str) -> Image.Image:
    """Load an image as RGB PIL, resized to TEST_SIZE."""
    return Image.open(path).convert("RGB").resize((TEST_SIZE, TEST_SIZE))


def make_comparison_grid(
    original: Image.Image,
    cloaked: Image.Image,
    edit_original: Image.Image,
    edit_cloaked: Image.Image,
    path: str,
) -> None:
    """
    Combine the four images into one labeled 2x2 grid for the demo slide:

        [ original      | cloaked        ]
        [ edit(original)| edit(cloaked)  ]
    """
    cell = TEST_SIZE
    grid = Image.new("RGB", (2 * cell, 2 * cell), (255, 255, 255))
    grid.paste(original, (0, 0))
    grid.paste(cloaked, (cell, 0))
    grid.paste(edit_original, (0, cell))
    grid.paste(edit_cloaked, (cell, cell))

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # very old Pillow fallback
        font = ImageFont.load_default()

    labels = [
        ("original (input)", 0, 0),
        ("cloaked (input)", cell, 0),
        ("AI edit of original", 0, cell),
        ("AI edit of cloaked", cell, cell),
    ]
    for text, x, y in labels:
        draw.text((x + 5, y + 5), text, fill=(0, 0, 0), font=font)

    grid.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Load inputs -----------------------------------------------------
    original = load_test_image(ORIGINAL_PATH)
    cloaked = load_test_image(CLOAKED_PATH)

    # ---- 2. Load the int8 OpenVINO pipeline ---------------------------------
    # First run downloads ~1 GB of quantized weights from Hugging Face.
    # RAM-saving tricks for this 4 GB laptop:
    #   * compile=False: don't compile all 4 models up front (each model
    #     compiles lazily the first time it is used)
    #   * encode the prompt once, then free the text encoder (~500 MB back)
    #     before the heavy UNet compiles
    # NOTE: we deliberately do NOT call pipe.reshape() here - static shapes
    # make this OpenVINO version fail on the GroupNormalization op.
    print(f"Loading pipeline: {SD_REPO} ...", flush=True)
    pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        SD_REPO,
        safety_checker=None,      # not needed, saves RAM + download
        feature_extractor=None,
        compile=False,            # don't compile all 4 models up front
        ov_config={"INFERENCE_NUM_THREADS": "2"},  # less RAM during compile
    )

    # Encode the prompt once; afterwards the text encoder is no longer needed.
    print("Encoding prompt...", flush=True)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=PROMPT,
        device=torch.device("cpu"),   # this diffusers version requires it
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    pipe.text_encoder.clear_requests()
    pipe.text_encoder = None   # frees the text encoder (nn.Module stores None)
    gc.collect()
    print("Text encoder freed.", flush=True)

    # ---- 3. Run the same edit on both images --------------------------------
    def run_edit(init_image: Image.Image, name: str) -> Image.Image:
        # A fresh generator with the SAME seed for both runs -> fair test
        generator = torch.Generator().manual_seed(SEED)
        t0 = time.time()
        result = pipe(
            prompt_embeds=prompt_embeds,           # pre-computed above
            negative_prompt_embeds=negative_prompt_embeds,
            image=init_image,
            strength=STRENGTH,
            num_inference_steps=NUM_STEPS,
            guidance_scale=7.5,
            generator=generator,
            output_type="pil",
        ).images[0]
        print(f"  edit({name}) done in {time.time() - t0:.0f}s", flush=True)
        return result

    print(f"Editing original with prompt: '{PROMPT}' ...", flush=True)
    edit_original = run_edit(original, "original")
    print(f"Editing cloaked  with prompt: '{PROMPT}' ...", flush=True)
    edit_cloaked = run_edit(cloaked, "cloaked")

    # ---- 4. Save outputs ------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    edit_original.save(os.path.join(OUTPUT_DIR, "edit_original.png"))
    edit_cloaked.save(os.path.join(OUTPUT_DIR, "edit_cloaked.png"))
    make_comparison_grid(
        original, cloaked, edit_original, edit_cloaked,
        os.path.join(OUTPUT_DIR, "comparison_side_by_side.png"),
    )

    # ---- 5. Quantify how different the two edits are ---------------------------
    a = np.asarray(edit_original, dtype=np.float64)
    b = np.asarray(edit_cloaked, dtype=np.float64)
    diff = np.abs(a - b)
    mse = (diff ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    print("\n--- Effectiveness results ---")
    print(f"Mean |edit(original) - edit(cloaked)| : {diff.mean():.1f} / 255")
    print(f"PSNR between the two edits            : {psnr:.1f} dB")
    print("  (low PSNR / high difference = the cloak changed what the AI does)")
    print(f"\nSaved to '{OUTPUT_DIR}/': edit_original.png, edit_cloaked.png, "
          f"comparison_side_by_side.png")


if __name__ == "__main__":
    main()
