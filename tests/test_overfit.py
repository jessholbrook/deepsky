"""The money test: a tiny model must be able to learn and then invert the
diffusion process on a fixed batch.

If this fails, the model/diffusion/sampler stack has a bug — no long run is
worth starting. ~10-20 min on MPS, slower on CPU. Run explicitly:

    uv run pytest tests/test_overfit.py -m slow -s

Why reconstruction, not sample-from-noise: a tiny model memorizing 8 images
learns an ill-defined score field *between* those 8 sparse modes, and the top
timesteps of the cosine schedule carry ~no signal (ᾱ_999 ≈ 2e-9), so DDIM from
fresh noise lands in blurry in-between regions — a property of the degenerate
8-point setup, not a stack bug (verified: identical on CPU and MPS, unchanged by
2x training). Instead we test the thing that *must* hold for the real run: the
sampler correctly reverses the forward process on-distribution. We noise real
images to a mid timestep and require DDIM to reconstruct them. Empirically this
lands at ~0.04 L1 across seeds; the 0.1 bar leaves ~2x margin.
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

    # (1) The model must be able to fit the batch at all.
    assert loss.item() < 0.03, f"failed to overfit: final loss {loss.item():.4f}"

    # (2) The full sampler chain must reverse the forward process. Noise the real
    #     images to a mid timestep, then reconstruct with DDIM from that latent.
    model.eval()
    t_start = 500
    torch.manual_seed(1)
    noise = torch.randn_like(x0)
    t = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
    xt = diffusion.q_sample(x0, t, noise)
    recon = ddim_sample(
        model, diffusion, x0.shape, device, steps=100, progress=False,
        x_start=xt, t_start=t_start,
    )
    assert torch.isfinite(recon).all()
    recon_l1 = (recon - x0).abs().mean().item()
    print(f"reconstruction L1 from t={t_start}: {recon_l1:.4f}")
    assert recon_l1 < 0.1, f"sampler failed to reconstruct: L1 {recon_l1:.4f}"
