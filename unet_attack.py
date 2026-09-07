"""
Cloak — UNet-level adversarial attack
Corrupts both the VAE encoder AND the UNet noise-prediction step so the
cloak survives high-strength img2img edits (0.8-1.0), not just low strength.
Perturbation confined to the face region only.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --- Strengthened hyperparameters ---
EPSILON = 8/255
ALPHA = 1/255
DITHER_STD = 0.5/255
NUM_STEPS = 80              # increased from 65
TIMESTEP_LOW = 100          # widened range, favors early/high-noise steps
TIMESTEP_HIGH = 900         # (more relevant to how strength=0.9 sampling works)
ENCODER_LOSS_WEIGHT = 0.6   # increased from 0.5


def load_models():
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.eval(); text_encoder.eval(); unet.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in text_encoder.parameters(): p.requires_grad_(False)
    for p in unet.parameters(): p.requires_grad_(False)

    with torch.no_grad():
        empty_tokens = tokenizer([""], padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  return_tensors="pt").to(device)
        text_embeddings = text_encoder(empty_tokens.input_ids)[0]

    return vae, unet, noise_scheduler, text_embeddings


def get_face_mask(image_pil):
    """OpenCV Haar cascade face detection — falls back to a centered box."""
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    mask = np.zeros((h, w), dtype=np.float32)
    if len(faces) == 0:
        bx0, by0, bx1, by1 = int(w*0.25), int(h*0.05), int(w*0.75), int(h*0.55)
    else:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        pad_x, pad_y = int(fw*0.4), int(fh*0.4)
        bx0, by0 = max(0, fx-pad_x), max(0, fy-pad_y)
        bx1, by1 = min(w, fx+fw+pad_x), min(h, fy+fh+pad_y)

    mask[by0:by1, bx0:bx1] = 1.0
    return mask, (bx0, by0, bx1, by1)


def pil_to_tensor(image_pil, size=512):
    image_pil = image_pil.convert("RGB").resize((size, size))
    arr = np.array(image_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(tensor):
    arr = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def to_vae_input(tensor01):
    return tensor01 * 2 - 1


def cloak_image(image_pil, vae, unet, noise_scheduler, text_embeddings):
    """Runs the UNet-level PGD attack. Returns the cloaked PIL image."""
    mask_np, box = get_face_mask(image_pil)
    x = pil_to_tensor(image_pil)
    mask = torch.from_numpy(cv2.resize(mask_np, (512, 512))).to(device)
    mask = (mask > 0.5).float().unsqueeze(0).unsqueeze(0)

    delta = (torch.empty_like(x).uniform_(-EPSILON, EPSILON) * mask).clone().detach()
    vae_scale = vae.config.scaling_factor

    with torch.no_grad():
        clean_latents = vae.encode(to_vae_input(x)).latent_dist.mean * vae_scale

    for step in range(NUM_STEPS):
        delta.requires_grad_(True)
        x_adv = (x + delta).clamp(0, 1)
        x_adv = x_adv * mask + x * (1 - mask)

        latents = vae.encode(to_vae_input(x_adv)).latent_dist.mean * vae_scale

        t = torch.randint(TIMESTEP_LOW, TIMESTEP_HIGH, (1,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
        unet_loss = F.mse_loss(noise_pred, noise)
        encoder_loss = F.mse_loss(latents, clean_latents)

        loss = unet_loss + ENCODER_LOSS_WEIGHT * encoder_loss
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            delta = delta + ALPHA * grad.sign() * mask
            delta = delta + torch.randn_like(delta) * DITHER_STD * mask
            delta = delta.clamp(-EPSILON, EPSILON) * mask
            delta = ((x + delta).clamp(0, 1) - x) * mask

    x_adv_final = (x + delta.detach()).clamp(0, 1)
    return tensor_to_pil(x_adv_final), box
