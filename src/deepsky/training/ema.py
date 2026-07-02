"""Exponential moving average of model weights with warmup ramp."""

from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        # Warmup: effective decay ramps from 0 so early EMA isn't stuck at init.
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v, 1.0 - decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.step = state["step"]
        self.shadow = state["shadow"]
