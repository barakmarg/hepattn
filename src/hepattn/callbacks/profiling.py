import time
from collections import defaultdict

import numpy as np
import torch
from lightning import Callback


class ProfilingCallback(Callback):
    """Profile training step breakdown: data loading, forward, loss, backward, optimizer.

    Also collects per-section GPU timings from the model's forward pass
    (requires the model to set self._section_times when self._profiling is True).

    Usage:
        Add to trainer callbacks. Runs for `active_steps` steps after `warmup_steps`,
        then prints a report and stops training.
    """

    def __init__(self, warmup_steps: int = 5, active_steps: int = 50):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.step = 0
        self.times = defaultdict(list)
        self._t = {}

    def _active(self):
        return self.warmup_steps < self.step <= self.warmup_steps + self.active_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step += 1
        if not self._active():
            return

        # Time since last batch end = data loading
        if "batch_end" in self._t:
            self.times["data_loading"].append(time.perf_counter() - self._t["batch_end"])

        # Enable model-level section timing
        pl_module.model._profiling = True

        torch.cuda.synchronize()
        self._t["fwd_start"] = time.perf_counter()

    def on_before_backward(self, trainer, pl_module, loss):
        if not self._active():
            return
        torch.cuda.synchronize()
        self.times["forward+loss"].append(time.perf_counter() - self._t["fwd_start"])
        self._t["bwd_start"] = time.perf_counter()

    def on_after_backward(self, trainer, pl_module):
        if not self._active():
            return
        torch.cuda.synchronize()
        self.times["backward"].append(time.perf_counter() - self._t["bwd_start"])
        self._t["opt_start"] = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._active():
            torch.cuda.synchronize()
            self.times["optimizer"].append(time.perf_counter() - self._t["opt_start"])

            # Collect model forward-pass section timings
            if hasattr(pl_module.model, "_section_times") and pl_module.model._section_times:
                for name, ms in pl_module.model._section_times.items():
                    self.times[f"  fwd/{name}"].append(ms)
                pl_module.model._section_times = {}

            pl_module.model._profiling = False
            self._t["batch_end"] = time.perf_counter()

        elif self.step == self.warmup_steps + self.active_steps + 1:
            self._print_report()
            trainer.should_stop = True

        # Track batch end time during warmup too (for data_loading timing of first active step)
        if self.step == self.warmup_steps:
            self._t["batch_end"] = time.perf_counter()

    def _print_report(self):
        print("\n" + "=" * 70)
        print(f"PROFILING REPORT  (averaged over {self.active_steps} steps)")
        print("=" * 70)

        step_keys = ["data_loading", "forward+loss", "backward", "optimizer"]
        total_ms = 0
        for name in step_keys:
            if name in self.times:
                arr = np.array(self.times[name]) * 1000  # s → ms
                total_ms += arr.mean()
                pct = arr.mean() / 1 * 100  # placeholder, filled below

        # Second pass with known total
        for name in step_keys:
            if name in self.times:
                arr = np.array(self.times[name]) * 1000
                pct = arr.mean() / total_ms * 100 if total_ms > 0 else 0
                print(f"  {name:<22} {arr.mean():8.1f} ± {arr.std():6.1f} ms  ({pct:4.1f}%)")

        if total_ms > 0:
            print(f"  {'total step':<22} {total_ms:8.1f} ms  ({1000/total_ms:.2f} it/s)")

        # Forward-pass breakdown
        fwd_keys = sorted(k for k in self.times if k.startswith("  fwd/"))
        if fwd_keys:
            print()
            print("Forward pass breakdown (GPU kernel time via CUDA events):")
            print("-" * 55)
            fwd_total = 0
            for name in fwd_keys:
                arr = np.array(self.times[name])  # already in ms from CUDA events
                fwd_total += arr.mean()
            for name in fwd_keys:
                arr = np.array(self.times[name])
                pct = arr.mean() / fwd_total * 100 if fwd_total > 0 else 0
                print(f"  {name:<22} {arr.mean():8.1f} ± {arr.std():6.1f} ms  ({pct:4.1f}%)")
            print(f"  {'  fwd/total':<22} {fwd_total:8.1f} ms")

        print("=" * 70 + "\n")
