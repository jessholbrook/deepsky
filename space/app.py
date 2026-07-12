"""Gradio demo for the deepsky diffusion model — live 128px generation.

Loads the trained EMA weights from the Hugging Face Hub and samples with DDIM.
Runs on ZeroGPU when the Space has it, otherwise falls back to CPU.
"""

import random

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

# ZeroGPU: `spaces.GPU` requests a GPU only for the duration of each decorated
# call. Off ZeroGPU (plain CPU Space, or local), the import may be absent — fall
# back to a no-op decorator so the exact same file runs everywhere.
try:
    import spaces

    gpu = spaces.GPU
except Exception:  # not a ZeroGPU Space

    def gpu(*dargs, **dkwargs):
        def wrap(fn):
            return fn

        return dargs[0] if dargs and callable(dargs[0]) else wrap


REPO_ID = "jessholbrook/deepsky-128px"
CKPT = "ckpt_0200000.pt"
IMAGE_SIZE = 128

# 128px architecture — matches configs/cloud-128px-full.yaml.
cfg = ModelCfg(
    base_channels=128,
    channel_mults=(1, 2, 3, 4),
    num_res_blocks=2,
    attn_resolutions=(16, 8),
    time_emb_dim=512,
)

# Lazy caches. The checkpoint is downloaded + loaded once (CPU), then the model
# and a device-matched schedule are built per device on first use. On ZeroGPU,
# CUDA is only visible *inside* the @gpu call — so device selection and the move
# to cuda both happen in generate(), not at import.
_state = None
_by_device = {}


def _state_dict():
    global _state
    if _state is None:
        print("Downloading checkpoint from the Hub…", flush=True)
        path = hf_hub_download(REPO_ID, CKPT)
        print(f"Loading weights from {path}…", flush=True)
        _state = torch.load(path, map_location="cpu")
    return _state


def _prepare(device):
    key = str(device)
    if key not in _by_device:
        m = UNet(cfg, IMAGE_SIZE)
        m.load_state_dict(_state_dict()["ema"]["shadow"])  # EMA = best quality
        m = m.to(device).eval()
        # GaussianDiffusion has no .to(); move its schedule tensors instead.
        diffusion = GaussianDiffusion(make_schedule("cosine", 1000).to(device))
        _by_device[key] = (m, diffusion)
        print(f"Model ready on {key}.", flush=True)
    return _by_device[key]


@gpu(duration=120)
@torch.no_grad()
def generate(steps, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = _prepare(device)
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
        "each 128px image takes a few seconds on GPU, or ~30–90s on the free CPU tier — "
        "lower the step count for speed.*"
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
