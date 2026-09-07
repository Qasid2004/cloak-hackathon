"""
app.py — Cloak live-demo backend
---------------------------------
A tiny FastAPI server that protects an uploaded photo with our tested
attack, so the hackathon demo is "upload a photo, get a cloaked photo".

It does NOT re-implement the attack: every piece of math lives in
cloak_attack.py (the exact config we validated: face-region PGD on the VAE
encoder, epsilon=8/255, per-step dither). This file only adds the web
glue around it:

    POST /cloak   upload an image  ->  get the cloaked PNG back
    GET  /        a small browser page for the live demo
    GET  /health  is the server alive / is the VAE loaded yet

Why the VAE is loaded once and attacks run one-at-a-time: this laptop has
4 GB RAM, so a second parallel attack would swap to disk and die.

Run it with:
    .venv\\Scripts\\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
"""

import io
import os
import tempfile
import threading

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from diffusers import AutoencoderKL
from PIL import Image

# Our tested attack module: constants + face mask + PGD loop all come
# from here, untouched.
import cloak_attack as ca
from vae_roundtrip import VAE_ID, compute_psnr, tensor_to_image

app = FastAPI(
    title="Cloak",
    description="Adversarial perturbation that protects photos from AI edits.",
)

# Serve static assets (team photos, etc.) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Shared state (loaded once, reused by every request)
# ---------------------------------------------------------------------------

# (vae, device) tuple, filled on the first request - loading the VAE takes
# ~30 s, so we do it lazily once instead of at every import/startup.
_STATE = {"vae": None, "device": None}
_LOAD_LOCK = threading.Lock()    # guards the one-time VAE load
_ATTACK_LOCK = threading.Lock()  # only ONE attack at a time (4 GB RAM!)

# --- UNet (advanced) attack state -------------------------------------------
# Lazy-loaded on the first /cloak-unet request.  The full SD pipeline
# (VAE + UNet + CLIP) needs ~6 GB on GPU; on a 4 GB CPU laptop it will
# fail at load time, which the route turns into a friendly 503.
_UNET_STATE = {"models": None}
_UNET_MODEL_ID = os.getenv(
    "CLOAK_SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
_UNET_DEFAULT_STEPS = 3
_UNET_MAX_STEPS = 80


def get_vae():
    """Return (vae, device), loading the VAE on first use."""
    if _STATE["vae"] is None:
        with _LOAD_LOCK:
            if _STATE["vae"] is None:   # second check: another thread may have won
                if torch.cuda.is_available():
                    device, dtype = torch.device("cuda"), torch.float16
                else:
                    device, dtype = torch.device("cpu"), torch.float32
                print(f"Loading VAE: {VAE_ID} ...", flush=True)
                vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
                vae = vae.to(device)
                vae.eval()
                # Same RAM trick as cloak_attack.py: recompute activations
                # during backward instead of storing them.
                vae.enable_gradient_checkpointing()
                _STATE["vae"], _STATE["device"] = vae, device
    return _STATE["vae"], _STATE["device"]


def run_attack(image_bytes: bytes, steps: int):
    """
    Run cloak_attack.py's tested pipeline on raw uploaded image bytes.
    Returns (png_bytes, metrics_dict).
    """
    vae, device = get_vae()

    # cloak_attack's helpers read from a path, so park the upload in a
    # temporary file for the duration of the attack.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "upload.png")
        with open(path, "wb") as f:
            f.write(image_bytes)

        # --- exactly the same sequence as cloak_attack.main() -------------
        original = ca.load_attack_image(path, device)
        mask, (x0, y0, x1, y1) = ca.find_face_mask(
            path, ca.ATTACK_IMAGE_SIZE, device
        )
        with torch.no_grad():
            z_orig = ca.encode_to_latent(vae, original)

        print(f"PGD attack: {steps} steps, eps={ca.EPSILON_PIXELS}/255, "
              f"face box x[{x0}:{x1}] y[{y0}:{y1}]", flush=True)
        cloaked = ca.pgd_attack(vae, original, z_orig, mask, num_steps=steps)

        # --- metrics (same formulas as cloak_attack.main) ------------------
        with torch.no_grad():
            z_cloaked = ca.encode_to_latent(vae, cloaked)
        metrics = {
            "latent-distance": round(ca.latent_distance(z_cloaked, z_orig).item(), 4),
            "psnr-db": round(compute_psnr(original, cloaked), 1),
            # tensor [-1,1] width 2.0 == 255 gray levels -> factor 127.5
            "max-change-255": round((cloaked - original).abs().max().item() * 127.5, 1),
            "outside-face-255": round(
                ((cloaked - original).abs() * (1.0 - mask)).max().item() * 127.5, 1
            ),
            "steps": steps,
        }

        # --- encode the result as PNG bytes --------------------------------
        buf = io.BytesIO()
        tensor_to_image(cloaked).save(buf, format="PNG")
        return buf.getvalue(), metrics


def get_unet_models():
    """Lazy-load the full SD pipeline for the advanced (UNet) attack.

    Returns (vae, unet, noise_scheduler, text_embeddings) — cached after
    the first successful call.  Sets the model id on unet_attack *before*
    loading so the correct checkpoint is fetched from Hugging Face.
    """
    if _UNET_STATE["models"] is None:
        with _LOAD_LOCK:
            if _UNET_STATE["models"] is None:
                import unet_attack as ua          # lazy import (heavy deps)
                ua.MODEL_ID = _UNET_MODEL_ID      # override default before load
                models = ua.load_models()          # may raise on OOM / network
                _UNET_STATE["models"] = models
    return _UNET_STATE["models"]


def _pil_to_unit_tensor(pil_img):
    """PIL RGB -> float32 tensor in [-1, 1], shape (1, 3, H, W).

    Same normalisation the VAE was trained on: bytes [0,255] -> [0,1] -> [-1,1].
    """
    arr = np.array(pil_img, dtype=np.float32) / 255.0   # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1)          # (3, H, W)
    t = t * 2.0 - 1.0                                    # [-1, 1]
    return t.unsqueeze(0)                                # (1, 3, H, W)


def run_attack_unet(image_bytes: bytes, steps: int):
    """Run the UNet-level attack from unet_attack.py.

    Returns (png_bytes, metrics_dict).  The dict keys match /cloak so the
    frontend can parse them identically.
    """
    vae, unet, sched, emb = get_unet_models()
    dev = next(vae.parameters()).device

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "upload.png")
        with open(path, "wb") as f:
            f.write(image_bytes)
        original_pil = Image.open(path).convert("RGB")

        # Face gate — raises RuntimeError("No face found ...") if none.
        ca.find_face_mask(path, ca.ATTACK_IMAGE_SIZE, torch.device("cpu"))

        # Run the UNet attack with the caller's step count, then restore
        # the module default so other code paths are unaffected.
        import unet_attack as ua
        saved = ua.NUM_STEPS
        try:
            ua.NUM_STEPS = steps
            cloaked_pil, _ = ua.cloak_image(
                original_pil, vae, unet, sched, emb
            )
        finally:
            ua.NUM_STEPS = saved

        # --- metrics (same keys /cloak returns) -------------------------
        size = cloaked_pil.size[0]
        orig_t = _pil_to_unit_tensor(
            original_pil.resize((size, size))
        ).to(dev)
        cloak_t = _pil_to_unit_tensor(cloaked_pil).to(dev)

        # Face mask for the "outside face" metric
        mask_np, _ = ua.get_face_mask(original_pil)
        mask = torch.from_numpy(
            cv2.resize(mask_np, (size, size))
        ).to(dev)
        mask = (mask > 0.5).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            z_o = vae.encode(orig_t).latent_dist.mean * vae.config.scaling_factor
            z_c = vae.encode(cloak_t).latent_dist.mean * vae.config.scaling_factor

        metrics = {
            "latent-distance": round((z_c - z_o).abs().mean().item(), 4),
            "psnr-db": round(compute_psnr(orig_t, cloak_t), 1),
            "max-change-255": round(
                (cloak_t - orig_t).abs().max().item() * 127.5, 1
            ),
            "outside-face-255": round(
                ((cloak_t - orig_t).abs() * (1.0 - mask)).max().item() * 127.5, 1
            ),
            "steps": steps,
        }

        buf = io.BytesIO()
        cloaked_pil.save(buf, "PNG")
        return buf.getvalue(), metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/cloak")
def cloak(
    file: UploadFile = File(..., description="The photo to protect (PNG/JPG)."),
    steps: int = Query(
        ca.NUM_STEPS, ge=1, le=ca.NUM_STEPS,
        description="PGD steps. 60 = full tested quality (~1 h on this CPU "
                    "laptop); lower = faster but weaker, for live demos.",
    ),
):
    """Protect an uploaded photo. Returns the cloaked image as a PNG."""
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # One attack at a time - a second parallel PGD loop would exhaust RAM.
    with _ATTACK_LOCK:
        try:
            png, metrics = run_attack(image_bytes, steps)
        except RuntimeError as e:      # e.g. "No face found in the image"
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:         # corrupt file, unsupported format...
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    # Metrics travel in response headers so the body stays a pure PNG.
    headers = {f"X-{k}": str(v) for k, v in metrics.items()}
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/cloak-unet")
def cloak_unet(
    file: UploadFile = File(..., description="The photo to protect (PNG/JPG)."),
    steps: int = Query(
        _UNET_DEFAULT_STEPS, ge=1, le=_UNET_MAX_STEPS,
        description="PGD steps for the UNet-level attack.",
    ),
):
    """Advanced protection: attacks both the VAE and the UNet."""
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    with _ATTACK_LOCK:
        # Try loading models first — surfaces a friendly 503 on CPU laptops.
        try:
            get_unet_models()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="The UNet models could not be loaded. This attack "
                       "needs a GPU and more RAM, e.g. Google Colab.",
            )
        try:
            png, metrics = run_attack_unet(image_bytes, steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="Ran out of GPU/CPU memory during the UNet attack.",
                )
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    headers = {f"X-{k}": str(v) for k, v in metrics.items()}
    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/health")
def health():
    """Liveness probe for the demo table."""
    return {
        "status": "ok",
        "vae_loaded": _STATE["vae"] is not None,
        "device": str(_STATE["device"]) if _STATE["device"] else "not loaded yet",
        "config": {
            "epsilon_pixels": ca.EPSILON_PIXELS,
            "dither_pixels": ca.NOISE_PIXELS,
            "default_steps": ca.NUM_STEPS,
            "attack_size": ca.ATTACK_IMAGE_SIZE,
        },
        "unet": {
            "loaded": _UNET_STATE["models"] is not None,
            "model_id": _UNET_MODEL_ID,
            "default_steps": _UNET_DEFAULT_STEPS,
            "max_steps": _UNET_MAX_STEPS,
            "note": "Heavy — needs GPU / Colab for real runs.",
        },
    }


# The judge-facing demo page: upload a photo, watch the spinner, get an
# original-vs-cloaked comparison plus plain-English metrics and a download
# button. Plain HTML/CSS/JS on purpose - nothing to build, nothing to break.
# Raw string so the JS '\u....' escapes reach the browser untouched.
_DEMO_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cloak - protect your photos from AI edits</title>
<style>
  :root { --bg:#0f1317; --card:#181f26; --line:#2a3540; --ink:#e8eef4;
          --dim:#9fb0bf; --accent:#4dd08c; --accent2:#4da3ff; --err:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.5 system-ui, "Segoe UI", sans-serif; }
  .wrap { max-width:880px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font-size:46px; margin:0 0 4px; letter-spacing:1px; }
  h1 span { color:var(--accent); }
  .tag { color:var(--dim); margin:0 0 28px; font-size:17px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:14px; padding:22px; margin-bottom:22px; }
  .drop { display:block; border:2px dashed var(--line); border-radius:12px;
          padding:26px; text-align:center; color:var(--dim); cursor:pointer; }
  .drop:hover, .drop.has { border-color:var(--accent2); color:var(--ink); }
  #thumb { max-height:70px; border-radius:8px; display:none; margin-top:14px; }
  .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-top:16px; }
  label.sel { color:var(--dim); font-size:14px; }
  select, button { font:inherit; border-radius:10px; border:1px solid var(--line);
                   background:#202a33; color:var(--ink); padding:10px 16px; }
  button.go { background:var(--accent); border-color:var(--accent);
              color:#08130c; font-weight:600; cursor:pointer; }
  button.go:disabled { opacity:.45; cursor:wait; }
  .load { display:none; text-align:center; padding:26px; color:var(--dim); }
  .spin { width:42px; height:42px; margin:0 auto 14px; border-radius:50%;
          border:4px solid var(--line); border-top-color:var(--accent);
          animation:sp 1s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
  .pair { display:flex; gap:22px; justify-content:center; flex-wrap:wrap; }
  figure { margin:0; text-align:center; }
  figure img { width:min(340px, 42vw); border-radius:10px;
               border:1px solid var(--line); background:#000; }
  figcaption { color:var(--dim); font-size:14px; margin-top:8px; }
  .metrics { display:grid; gap:10px; margin-top:20px; }
  .m { display:flex; justify-content:space-between; gap:12px; align-items:center;
       background:#202a33; border-radius:10px; padding:10px 14px; font-size:15px; }
  .m b { color:var(--accent); font-weight:600; white-space:nowrap; }
  .m .why { color:var(--dim); font-size:13px; display:block; }
  #err { color:var(--err); display:none; margin-top:14px; }
  #results { display:none; }
  .note { color:#ffcf6b; font-size:13px; display:none; margin-top:10px; }
  a.dl { display:inline-block; margin-top:18px; background:var(--accent2);
         color:#04101c; font-weight:600; border-radius:10px; padding:10px 18px;
         text-decoration:none; }
  footer { color:var(--dim); font-size:13px; margin-top:26px; }
  /* --- Team section --- */
  .team-section { margin-top:32px; }
  .team-title { color:var(--dim); font-size:14px; text-transform:uppercase;
                letter-spacing:1.5px; margin:0 0 16px; text-align:center; }
  .team-grid { display:flex; gap:22px; justify-content:center; flex-wrap:wrap; }
  .team-card { background:var(--card); border:1px solid var(--line);
               border-radius:14px; padding:16px 22px;
               display:flex; align-items:center; gap:16px;
               flex:1 1 0; min-width:0; }
  .team-photo { width:56px; height:56px; border-radius:50%; object-fit:cover;
                border:2px solid var(--line); flex-shrink:0; }
  .team-info { min-width:0; }
  .team-name { color:var(--ink); font-size:15px; font-weight:600; margin:0 0 4px;
               line-height:1.3; }
  .team-badge { display:inline-block; font-size:12px; padding:3px 10px;
                border-radius:20px; font-weight:500; }
  .team-badge.lead { background:rgba(77,208,140,0.15); color:var(--accent); }
  .team-badge.member { background:rgba(77,163,255,0.15); color:var(--accent2); }
  @media (max-width:560px) {
    .team-grid { flex-direction:column; }
    .team-card { flex:1 1 auto; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>Cloak<span>.</span></h1>
  <p class="tag">Adds an invisible shield to your photo so AI editors and
     face-swaps produce garbage &mdash; while the photo still looks perfect to you.</p>

  <div class="card">
    <label class="drop" id="drop" for="file">
      <b>Choose a photo</b> (or drop it here)<br>
      <span id="fname">PNG or JPG &mdash; a face must be visible</span>
    </label>
    <input id="file" type="file" accept="image/*" hidden>
    <img id="thumb" alt="">
    <div class="row">
      <label class="sel">Protection mode
        <select id="mode">
          <option value="standard" selected>Standard &mdash; Fast (VAE)</option>
          <option value="advanced">Advanced &mdash; Stronger (UNet)</option>
        </select>
      </label>
      <label class="sel">Strength / speed
        <select id="steps">
          <option value="2" selected>Live demo &mdash; 2 steps (~5 min)</option>
          <option value="10">Standard &mdash; 10 steps (~15-25 min)</option>
          <option value="30">Balanced &mdash; 30 steps (~45-75 min)</option>
          <option value="60">Full tested &mdash; 60 steps (~1.5-2 h)</option>
        </select>
      </label>
      <button class="go" id="go" disabled>Protect my photo</button>
    </div>
    <p class="note" id="advNote">Advanced (UNet) protection attacks the VAE
       and the UNet so it survives stronger AI edits &mdash; but it is very
       heavy and needs a GPU. On this CPU-only laptop it will likely fail to
       load; it is intended for the Google Colab demo. Standard mode runs
       here.</p>
    <p id="err"></p>
  </div>

  <div class="card load" id="load">
    <div class="spin"></div>
    <div>Protecting your photo&hellip;</div>
    <div>Elapsed <span id="clock">0:00</span> &mdash; the attack runs on CPU,
         this can take a few minutes.</div>
  </div>

  <div class="card" id="results">
    <div class="pair">
      <figure><img id="imgOrig" alt="original">
        <figcaption>Original</figcaption></figure>
      <figure><img id="imgOut" alt="cloaked">
        <figcaption id="capOut">Cloaked (256&times;256 attack resolution)</figcaption></figure>
    </div>
    <div class="metrics">
      <div class="m"><span>Invisibility (PSNR)
        <span class="why" id="mPsnrWhy"></span></span><b id="mPsnr"></b></div>
      <div class="m"><span>Attack strength (latent distance)
        <span class="why" id="mLatWhy"></span></span><b id="mLat"></b></div>
      <div class="m"><span>Perturbation budget
        <span class="why">max change per pixel, face region only</span></span>
        <b id="mMax"></b></div>
      <div class="m"><span>Rest of the photo
        <span class="why" id="mOutWhy"></span></span><b id="mOut"></b></div>
    </div>
    <a class="dl" id="dl" download="cloaked.png">&#11015; Download cloaked image</a>
  </div>

  <footer>PhotoGuard-style PGD attack on the Stable Diffusion VAE encoder
    (eps = 8/255, face region only, per-step dither). Runs fully on this
    machine &mdash; your photo never leaves the laptop.</footer>

  <div class="team-section">
    <p class="team-title">Team</p>
    <div class="team-grid">
      <div class="team-card">
        <img class="team-photo" src="/static/team/qasid.jpg" alt="Khawaja Qasid Rasheed Wyne">
        <div class="team-info">
          <p class="team-name">Khawaja Qasid Rasheed Wyne</p>
          <span class="team-badge lead">Team Lead</span>
        </div>
      </div>
      <div class="team-card">
        <img class="team-photo" src="/static/team/saim.jpg" alt="Muhammad Saim Ali Khan">
        <div class="team-info">
          <p class="team-name">Muhammad Saim Ali Khan</p>
          <span class="team-badge member">Team Member</span>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const file = $('file'), go = $('go');

// --- Step options per protection mode -----------------------------------
const STEPS = {
  standard: [
    {v:'2',  l:'Live demo &mdash; 2 steps (~5 min)'},
    {v:'10', l:'Standard &mdash; 10 steps (~15-25 min)'},
    {v:'30', l:'Balanced &mdash; 30 steps (~45-75 min)'},
    {v:'60', l:'Full tested &mdash; 60 steps (~1.5-2 h)'}
  ],
  advanced: [
    {v:'3',  l:'Live demo &mdash; 3 steps (unmeasured on this machine)', s:true},
    {v:'10', l:'Standard &mdash; 10 steps (unmeasured on this machine)'},
    {v:'30', l:'Balanced &mdash; 30 steps (unmeasured on this machine)'},
    {v:'80', l:'Full &mdash; 80 steps (unmeasured on this machine)'}
  ]
};

// Swap step options + toggle the GPU caution when the mode changes.
function applyMode() {
  var m = $('mode').value;
  $('steps').innerHTML = STEPS[m].map(function(o) {
    return '<option value="'+o.v+'"'+(o.s?' selected':'')+'>'+o.l+'</option>';
  }).join('');
  $('advNote').style.display = (m==='advanced') ? 'block' : 'none';
  $('capOut').innerHTML = (m==='advanced')
      ? 'Cloaked (512&times;512, VAE + UNet)'
      : 'Cloaked (256&times;256 attack resolution)';
  $('results').style.display = 'none';
  $('err').style.display = 'none';
}
$('mode').onchange = applyMode;

// Show a thumbnail + enable the button as soon as a file is picked.
file.onchange = () => {
  if (!file.files.length) return;
  $('fname').textContent = file.files[0].name;
  $('drop').classList.add('has');
  $('thumb').src = URL.createObjectURL(file.files[0]);
  $('thumb').style.display = 'inline-block';
  go.disabled = false;
  applyMode();
};

// Drag & drop onto the dashed area.
['dragover', 'drop'].forEach(ev =>
  $('drop').addEventListener(ev, e => e.preventDefault()));
$('drop').addEventListener('drop', e => {
  if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; file.onchange(); }
});

// Turn raw metric numbers into judge-friendly sentences.
function invisibilityWhy(p) {
  if (p >= 35) return 'looks identical to the original';
  if (p >= 30) return 'visually indistinguishable at a glance';
  if (p >= 25) return 'near-invisible (faint grain up close)';
  return 'changes visible - budget too high';
}
function strengthWhy(d) {
  if (d >= 0.25) return 'strong: AI edits of this photo get corrupted';
  if (d >= 0.10) return 'moderate: good for a live demo';
  return 'light: consider more steps for full strength';
}

let timer = null;
go.onclick = async () => {
  $('err').style.display = 'none';
  $('results').style.display = 'none';
  $('load').style.display = 'block';
  go.disabled = true;
  const t0 = Date.now();
  timer = setInterval(() => {           // live elapsed-time counter
    const s = Math.floor((Date.now() - t0) / 1000);
    $('clock').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  }, 1000);
  try {
    const fd = new FormData();
    fd.append('file', file.files[0]);
    const endpoint = ($('mode').value==='advanced') ? '/cloak-unet' : '/cloak';
    const r = await fetch(endpoint + '?steps=' + $('steps').value,
                          {method: 'POST', body: fd});
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    const url = URL.createObjectURL(await r.blob());
    $('imgOut').src = url;
    $('imgOrig').src = URL.createObjectURL(file.files[0]);
    $('dl').href = url;                 // powers the download button
    const psnr = parseFloat(r.headers.get('x-psnr-db'));
    const lat  = parseFloat(r.headers.get('x-latent-distance'));
    const out  = parseFloat(r.headers.get('x-outside-face-255'));
    $('mPsnr').textContent = psnr + ' dB';
    $('mPsnrWhy').textContent = invisibilityWhy(psnr);
    $('mLat').textContent = lat;
    $('mLatWhy').textContent = strengthWhy(lat);
    $('mMax').textContent = '\u00b1' + r.headers.get('x-max-change-255') + ' / 255';
    $('mOut').textContent = out === 0 ? 'pixel-identical' : 'changed by ' + out;
    $('mOutWhy').textContent = out === 0
        ? 'everything outside the detected face is untouched'
        : 'unexpected change outside the face box';
    $('results').style.display = 'block';
  } catch (e) {
    $('err').textContent = 'Error: ' + e.message;
    $('err').style.display = 'block';
  } finally {
    clearInterval(timer);
    $('load').style.display = 'none';
    go.disabled = false;
  }
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return _DEMO_HTML
