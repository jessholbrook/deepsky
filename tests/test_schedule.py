import pytest
import torch

from deepsky.diffusion.schedule import make_schedule


@pytest.mark.parametrize("name", ["cosine", "linear"])
@pytest.mark.parametrize("timesteps", [100, 1000])
def test_schedule_invariants(name, timesteps):
    s = make_schedule(name, timesteps)

    assert s.betas.shape == (timesteps,)
    assert (s.betas > 0).all() and (s.betas < 1).all()

    acp = s.alphas_cumprod
    assert (acp > 0).all() and (acp <= 1).all()
    assert (acp[1:] < acp[:-1]).all(), "alphas_cumprod must be strictly decreasing"

    # sqrt(ᾱ)² + sqrt(1-ᾱ)² == 1
    identity = s.sqrt_alphas_cumprod**2 + s.sqrt_one_minus_alphas_cumprod**2
    assert torch.allclose(identity, torch.ones(timesteps), atol=1e-5)

    assert (s.posterior_variance >= 0).all()


def test_cosine_endpoints_sane():
    s = make_schedule("cosine", 1000)
    # Early: almost no noise. Late: almost pure noise.
    assert s.alphas_cumprod[0] > 0.99
    assert s.alphas_cumprod[-1] < 0.01


def test_schedule_to_device_roundtrip():
    s = make_schedule("cosine", 100).to(torch.device("cpu"))
    assert s.betas.device.type == "cpu"
    assert s.timesteps == 100
