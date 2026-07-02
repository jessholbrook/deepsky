import pytest
import torch

from deepsky.config import ModelCfg
from deepsky.models.unet import UNet

SMALL = ModelCfg(base_channels=32, channel_mults=(1, 2), num_res_blocks=1, attn_resolutions=(8,), time_emb_dim=64)
FULL = ModelCfg()  # defaults = mac-64px preset


@pytest.mark.parametrize("image_size,batch", [(32, 1), (32, 4), (64, 2)])
def test_output_shape_matches_input(image_size, batch):
    model = UNet(SMALL, image_size=image_size)
    x = torch.randn(batch, 3, image_size, image_size)
    t = torch.randint(0, 1000, (batch,))
    out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_full_model_at_64px():
    model = UNet(FULL, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model(x, torch.tensor([0, 999]))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_zero_init_output_head():
    """Output conv is zero-initialized: fresh model predicts ~0, so the first
    training steps see loss ≈ E[eps²] = 1.0 — a useful sanity anchor."""
    model = UNet(SMALL, image_size=32)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, torch.tensor([10, 20]))
    assert out.abs().max().item() == 0.0


CLOUD = ModelCfg(base_channels=128, channel_mults=(1, 2, 3, 4), time_emb_dim=512)


def test_param_counts():
    small = sum(p.numel() for p in UNet(SMALL, image_size=32).parameters())
    full = sum(p.numel() for p in UNet(FULL, image_size=64).parameters())
    cloud = sum(p.numel() for p in UNet(CLOUD, image_size=128).parameters())
    assert small < 5e6
    assert 15e6 < full < 40e6, f"64px preset ~19M params, got {full/1e6:.1f}M"
    assert 60e6 < cloud < 150e6, f"128px preset ~87M params, got {cloud/1e6:.1f}M"
