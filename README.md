# Cloak

**Invisible adversarial perturbation that protects your photos from AI-based deepfakes, face-swaps, and unwanted edits.**

Cloak adds a mathematically-crafted, human-invisible "shield" to your photo. When someone tries to use AI tools (Stable Diffusion img2img, face-swap models, etc.) to edit or manipulate your image, the AI produces corrupted, unusable output — while your photo still looks perfectly normal to human eyes.

Built for the **Alibaba Cloud AI Hackathon Pakistan 2026**.

---

## How it works

Cloak implements a **PhotoGuard-style PGD (Projected Gradient Descent) attack** on Stable Diffusion's encoder. It adds a tiny perturbation (max 8/255 per pixel, ~3% change) confined to the face region. This perturbation:

- **Looks invisible** to humans (PSNR 38–41 dB — visually indistinguishable from the original)
- **Corrupts the latent space** that AI models rely on, so downstream edits fail

The perturbation is computed via gradient ascent on the encoder's latent representation, then projected back into an ε-ball to keep pixel changes imperceptible.

---

## Two protection modes

| | **Standard** | **Advanced** |
|---|---|---|
| **Attack target** | VAE encoder only | VAE encoder + UNet noise predictor |
| **Speed** | Fast (~5 min for 2 steps on CPU) | Slow (requires GPU) |
| **Hardware** | Runs on CPU (4 GB RAM laptop) | Needs GPU (Google Colab, etc.) |
| **Strength** | Good for most use cases | Survives high-strength img2img edits (0.8–1.0) |
| **Resolution** | 256×256 attack | 512×512 attack |

### Validated metrics

**Standard mode (VAE-only):**
- PSNR: **38–41 dB** (invisible to the human eye; >30 dB is the threshold for "visually indistinguishable")
- Latent distance: 5–10× increase over random noise baseline
- Max pixel change: bounded to 8/255 (ε = 8/255)
- Outside face region: **0 change** (perturbation confined to detected face box)

**Advanced mode (UNet-level, validated on Colab GPU):**
- Edit-PSNR at strength 0.9: dropped from **~20.6 dB → ~14.59 dB**
  - Lower edit-PSNR = more corruption in AI-generated output
  - Proves the UNet attack survives high-strength edits that would otherwise bypass VAE-only protection

---

## Quick start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 2. Run the server

```bash
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 3. Open the demo

Visit **http://127.0.0.1:8000/** in your browser. Upload a photo with a visible face, choose a protection mode and strength, and download the cloaked result.

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive web demo |
| `/cloak` | POST | Standard protection (VAE attack) |
| `/cloak-unet` | POST | Advanced protection (VAE + UNet attack) |
| `/health` | GET | Server status and config |
| `/openapi.json` | GET | Full API schema |

Both `/cloak` and `/cloak-unet` accept:
- `file`: uploaded image (PNG/JPG)
- `steps`: query parameter (PGD iterations)

Both return the cloaked image as PNG with quality metrics in response headers (`X-Psnr-Db`, `X-Latent-Distance`, `X-Max-Change-255`, `X-Outside-Face-255`, `X-Steps`).

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CLOAK_SD_MODEL_ID` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Hugging Face model ID for the UNet attack |
| `HF_TOKEN` | *(none)* | Hugging Face API token (if needed for gated models) |

---

## Project structure

```
app.py                 # FastAPI server + embedded HTML demo
cloak_attack.py        # VAE-level PGD attack (Standard mode)
unet_attack.py         # UNet-level PGD attack (Advanced mode)
vae_roundtrip.py       # Step 1: verify VAE encode/decode works
test_cloak_effectiveness.py  # Validation tests
requirements.txt       # Python dependencies
```

---

## Technical details

- **Attack resolution**: 256×256 (Standard) or 512×512 (Advanced) — VAE accepts any multiple of 8
- **Perturbation budget**: ε = 8/255 per pixel, face region only
- **Face detection**: OpenCV Haar cascade (human + cat face classifiers)
- **PGD steps**: configurable (2–60 for Standard, 3–80 for Advanced)
- **Per-step dither**: 1/255 uniform noise to break structured patterns
- **Gradient checkpointing**: enabled on VAE to fit in 4 GB RAM

---

## Requirements

- Python 3.9+
- PyTorch 2.x
- diffusers, transformers (Hugging Face)
- OpenCV (face detection)
- FastAPI + uvicorn (web server)

**Standard mode**: runs on CPU with 4 GB RAM.  
**Advanced mode**: requires GPU with ≥6 GB VRAM (tested on Google Colab T4).

---

## License

Built for educational and research purposes. Use responsibly.

---

**Alibaba Cloud AI Hackathon Pakistan 2026**
