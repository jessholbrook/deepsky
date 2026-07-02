"""The money test: a tiny model must be able to memorize 8 images.

If this fails, the model/diffusion/sampler stack has a bug — no long run is
worth starting. ~10-20 min on MPS, slower on CPU. Run explicitly:

    uv run pytest tests/test_overfit.py -m slow -s
"""

import pytest
import torch

from deepsky.config import ModelCfg
from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.samplers import ddim_sample
from deepsky.diffusion.schedule import make_schedule
from deepsky.models.unet import UNet


@pytest.mark.slow
def test_overfit_one_batch():
    torch.manual_seed(0)
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )

    # 8 fixed structured images (smooth gradients + blobs), 64px.
    xs = torch.linspace(-1, 1, 64)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    batch = []
    for i in range(8):
        img = torch.stack(
            [
                torch.sin(3 * xx + i) * torch.cos(2 * yy),
                torch.exp(-((xx - 0.3 * i % 1) ** 2 + yy**2) * 4) * 2 - 1,
                torch.sin(5 * yy - i),
            ]
        )
        batch.append(img.clamp(-1, 1))
    x0 = torch.stack(batch).to(device)

    cfg = ModelCfg(base_channels=32, channel_mults=(1, 2), num_res_blocks=1,
                   attn_resolutions=(16,), time_emb_dim=128, dropout=0.0)
    model = UNet(cfg, image_size=64).to(device)
    diffusion = GaussianDiffusion(make_schedule("cosine", 1000).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for step in range(3000):
        loss = diffusion.loss(model, x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.5f}")

    assert loss.item() < 0.03, f"failed to overfit: final loss {loss.item():.4f}"

    model.eval()
    samples = ddim_sample(model, diffusion, (8, 3, 64, 64), device, steps=100, progress=False)
    assert torch.isfinite(samples).all()
    # Each sample should be close to SOME training image (memorization).
    dists = (samples[:, None] - x0[None, :]).abs().mean(dim=(2, 3, 4))
    best = dists.min(dim=1).values
    print(f"per-sample best-match L1: {best.tolist()}")
    assert best.mean() < 0.25, f"samples don't resemble training data: {best.tolist()}"
