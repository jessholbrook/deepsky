"""Gradio demo for the deepsky diffusion model — live 128px generation.

Loads the trained EMA weights from the Hugging Face Hub and samples with DDIM.
Runs on the Space's CPU by default (or GPU if the Space has one).
"""

import random
import sys

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from deepsky.config import ModelCfg
from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.samplers import ddim_sample
from deepsky.diffusion.schedule import make_schedule
from deepsky.models.unet import UNet

REPO_ID = "jessholbrook/deepsky-128px"
CKPT = "ckpt_0200000.pt"
IMAGE_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 128px architecture — matches configs/cloud-128px-full.yaml.
cfg = ModelCfg(
    base_channels=128,
    channel_mults=(1, 2, 3, 4),
    num_res_blocks=2,
    attn_resolutions=(16, 8),
    time_emb_dim=512,
)

# Lazy singletons — built on first generate() so the Gradio server binds its
# port immediately and the Space reaches "Running" instead of timing out while
# a 1.39 GB checkpoint downloads + loads on the free CPU tier.
_model = None
_diffusion = None


def _load():
    global _model, _diffusion
    if _model is not None:
        return _model, _diffusion
    print("Downloading checkpoint from the Hub…", flush=True)
    ckpt_path = hf_hub_download(REPO_ID, CKPT)
    print(f"Loading weights from {ckpt_path} onto {device}…", flush=True)
    state = torch.load(ckpt_path, map_location=device)
    m = UNet(cfg, IMAGE_SIZE).to(device)
    m.load_state_dict(state["ema"]["shadow"])  # EMA weights = best quality
    m.eval()
    _model = m
    _diffusion = GaussianDiffusion(make_schedule("cosine", 1000).to(device))
    print("Model ready.", flush=True)
    return _model, _diffusion


@torch.no_grad()
def generate(steps, seed):
    model, diffusion = _load()
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(int(seed))
    samples = ddim_sample(
        model, diffusion, (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        device, steps=int(steps), progress=False,
    )
    arr = ((samples[0].clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(arr)), int(seed)


with gr.Blocks(title="deepsky") as demo:
    gr.Markdown(
        "# 🌌 deepsky\n"
        "A **from-scratch** diffusion model generating deep-sky astronomical images — "
        "nebulae, galaxies, star clusters — trained on public ESA/Hubble, ESA/Webb, "
        "ESO, and NASA imagery. Every pixel is generated; nothing is retrieved.\n\n"
        "Model: [jessholbrook/deepsky-128px](https://huggingface.co/jessholbrook/deepsky-128px) · "
        "Code: [github.com/jessholbrook/deepsky](https://github.com/jessholbrook/deepsky)\n\n"
        "*The first generation also loads the model (one-time ~1–2 min). After that, "
        "each 128px image takes ~30–90s on the free CPU tier — lower the step count for speed.*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            steps = gr.Slider(20, 150, value=50, step=5,
                              label="DDIM steps (more = sharper, slower)")
            seed = gr.Number(value=-1, label="Seed (−1 = random)", precision=0)
            btn = gr.Button("✨ Generate", variant="primary")
            used = gr.Number(label="Seed used", interactive=False, precision=0)
        with gr.Column(scale=1):
            out = gr.Image(label="Generated 128px sample", type="pil", height=384)
    btn.click(generate, [steps, seed], [out, used])

demo.queue(max_size=8).launch()
