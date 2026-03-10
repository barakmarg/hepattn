from pathlib import Path

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer


class PflowPredictionWriter(Callback):
    """Write pileup removal predictions to HDF5."""

    def __init__(self) -> None:
        super().__init__()
        self.all_outputs = []

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return
        self.trainer = trainer
        self.all_outputs = []

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        test_suff = getattr(self.trainer.datamodule, "test_suff", None)
        suffix = f"_{test_suff}" if test_suff else ""
        return Path(out_dir / f"{out_basename}__test_pileup{suffix}.h5")

    def on_test_batch_end(self, trainer, module, test_step_outputs, batch, batch_idx):
        _inputs, targets = batch
        outputs, preds, _losses = test_step_outputs
        final_preds = preds["final"]

        batch_data = {
            "event_number": targets["event_number"].cpu().numpy(),
            "track_prob": final_preds["mask"]["pflow_node_prob"].squeeze(-2).cpu().float().numpy(),
            "track_is_hard_scatter": final_preds["mask"]["pflow_node_valid"].squeeze(-2).cpu().numpy(),
            "calo_hs_fraction": final_preds["calo_fraction"]["calo_hs_fraction"].squeeze(-1).cpu().float().numpy(),
            "tracks_mask_truth": targets["tracks_mask"].cpu().float().numpy(),
            "calo_energy_truth": targets["calo_hard_scatter_energy"].cpu().float().numpy(),
            "node_valid": targets["node_valid"].cpu().numpy(),
        }
        self.all_outputs.append(batch_data)

    def on_test_end(self, trainer, module):
        if not self.all_outputs:
            return

        # Concatenate all batches
        combined = {}
        for key in self.all_outputs[0]:
            combined[key] = np.concatenate([b[key] for b in self.all_outputs], axis=0)

        # Write to HDF5
        with h5py.File(self.output_path, "w") as f:
            for key, val in combined.items():
                f.create_dataset(key, data=val)

        print(f"Wrote pileup removal predictions to {self.output_path}")
        self.all_outputs = []
