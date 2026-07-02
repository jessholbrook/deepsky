import torch

from deepsky.config import ModelCfg
from deepsky.models.unet import UNet
from deepsky.training import checkpoint
from deepsky.training.ema import EMA

CFG = ModelCfg(base_channels=32, channel_mults=(1, 2), num_res_blocks=1,
               attn_resolutions=(8,), time_emb_dim=64)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = UNet(CFG, image_size=32)
    ema = EMA(model, 0.999)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # A couple of steps so optimizer/EMA state is non-trivial.
    for _ in range(3):
        loss = model(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

    path = tmp_path / "ckpt.pt"
    checkpoint.save(path, model, ema, opt, step=3)

    model2 = UNet(CFG, image_size=32)
    ema2 = EMA(model2, 0.999)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    step = checkpoint.load(path, model2, ema2, opt2, torch.device("cpu"))

    assert step == 3
    assert ema2.step == ema.step
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k])
    for k, v in ema.shadow.items():
        assert torch.equal(v, ema2.shadow[k])

    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([5])
    assert torch.equal(model(x, t), model2(x, t))


def test_prune_keeps_latest(tmp_path):
    model = UNet(CFG, image_size=32)
    ema = EMA(model, 0.999)
    opt = torch.optim.Adam(model.parameters())
    for step in [100, 200, 300, 400, 500]:
        checkpoint.save(tmp_path / f"ckpt_{step:07d}.pt", model, ema, opt, step)
    checkpoint.prune_old(tmp_path, keep=3)
    remaining = sorted(p.name for p in tmp_path.glob("ckpt_*.pt"))
    assert remaining == ["ckpt_0000300.pt", "ckpt_0000400.pt", "ckpt_0000500.pt"]
