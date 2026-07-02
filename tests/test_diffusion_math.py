import torch

from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.samplers import ddim_sample, ddpm_sample
from deepsky.diffusion.schedule import make_schedule


def make_diffusion(timesteps=1000):
    return GaussianDiffusion(make_schedule("cosine", timesteps))


def test_q_sample_roundtrip():
    """Noising with known eps then inverting with the true eps recovers x0 exactly."""
    torch.manual_seed(0)
    d = make_diffusion()
    x0 = torch.rand(4, 3, 16, 16) * 2 - 1
    t = torch.tensor([0, 250, 500, 999])
    eps = torch.randn_like(x0)

    xt = d.q_sample(x0, t, eps)
    recovered = d.predict_x0_from_eps(xt, t, eps)
    # Recovery divides by sqrt(alphas_cumprod), which is ~5e-5 at t=999 under the
    # cosine schedule — fp32 error gets amplified by 1/sqrt(acp), so scale tolerance.
    tol = 1e-3 / d.schedule.sqrt_alphas_cumprod[t]
    err = (recovered - x0).abs().amax(dim=(1, 2, 3))
    assert (err <= tol).all(), f"per-t error {err.tolist()} exceeds tolerance {tol.tolist()}"


def test_loss_near_zero_for_perfect_predictor():
    """A model that returns the injected noise exactly should have ~0 loss."""
    torch.manual_seed(0)
    d = make_diffusion()
    x0 = torch.rand(8, 3, 16, 16) * 2 - 1

    injected = {}

    orig_q_sample = d.q_sample

    def capturing_q_sample(x0, t, noise):
        injected["noise"] = noise
        return orig_q_sample(x0, t, noise)

    d.q_sample = capturing_q_sample
    loss = d.loss(lambda xt, t: injected["noise"], x0)
    assert loss.item() < 1e-10


def test_q_sample_variance_at_final_step():
    """At t=T-1, x_t should be nearly pure noise."""
    torch.manual_seed(0)
    d = make_diffusion()
    x0 = torch.ones(1, 3, 64, 64)
    t = torch.tensor([999])
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    assert torch.allclose(xt, eps, atol=0.15)


class ConstantEpsModel(torch.nn.Module):
    """Predicts zero noise: samplers should run and return finite tensors."""

    def forward(self, x, t):
        return torch.zeros_like(x)


def test_samplers_run_and_are_finite():
    d = GaussianDiffusion(make_schedule("cosine", 50))
    model = ConstantEpsModel()
    shape = (2, 3, 16, 16)

    out_ddpm = ddpm_sample(model, d, shape, torch.device("cpu"), progress=False)
    out_ddim = ddim_sample(model, d, shape, torch.device("cpu"), steps=10, progress=False)

    for out in (out_ddpm, out_ddim):
        assert out.shape == shape
        assert torch.isfinite(out).all()
        assert out.abs().max() <= 1.0 + 1e-4  # x0 clamping keeps outputs in range
