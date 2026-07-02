"""Minimal metrics logging: stdout + CSV. No external services."""

from __future__ import annotations

import csv
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "metrics.csv"
        new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if new:
            self.writer.writerow(["step", "loss", "lr", "steps_per_sec", "time"])
        self.last_time = time.time()
        self.last_step = 0

    def log(self, step: int, loss: float, lr: float) -> None:
        now = time.time()
        rate = (step - self.last_step) / max(now - self.last_time, 1e-9)
        self.last_time, self.last_step = now, step
        self.writer.writerow([step, f"{loss:.5f}", f"{lr:.2e}", f"{rate:.2f}", int(now)])
        self.file.flush()
        print(f"step {step:>7}  loss {loss:.4f}  lr {lr:.2e}  {rate:.2f} it/s")
