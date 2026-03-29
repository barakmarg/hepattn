"""Prediction writer for the merged pileup removal + reconstruction pipeline.

Writes pileup removal outputs (track mask, calo mask, calo fraction) and
reconstruction outputs (class, masks, incidence, regression) to HDF5/ROOT.
"""

from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from hepattn.utils.array_utils import join_structured_arrays, maybe_pad


class PflowPredictionWriter(Callback):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        self.writer = None
        self.trainer = trainer
        self.batch_size = trainer.datamodule.batch_size
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_suff = trainer.datamodule.test_suff
        self.num_events = len(self.ds)
        self.var_transform = self.ds.scaler.transforms

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        return Path(out_dir / f"{out_basename}__test{suffix}.h5")

    def _write_batch_outputs(self, to_write, batch_idx):
        if self.writer is None:
            from ftag.hdf5 import H5Writer
            dtypes = {k: v.dtype for k, v in to_write.items()}
            shapes = {k: (self.num_events, *v.shape[1:]) for k, v in to_write.items()}
            self.writer = H5Writer(
                jets_name="events",
                dst=self.output_path,
                dtypes=dtypes,
                shapes=shapes,
                shuffle=False,
                precision="full",
            )
        self.writer.write(to_write)

    def on_test_batch_end(self, trainer, module, test_step_outputs, batch, batch_idx):
        _inputs, targets = batch
        outputs, preds, _losses = test_step_outputs

        to_write = {}

        # Event numbers
        to_write["events"] = u2s(
            targets["event_number"].cpu().numpy().astype(np.int64).reshape(-1, 1),
            dtype=np.dtype([("event_number", "i8")]),
        )

        # --- Pileup removal outputs ---
        track_final = preds.get("track_final", {})
        if "mask" in track_final:
            to_write["track_mask"] = u2s(
                track_final["mask"]["pflow_node_prob"].cpu().float().unsqueeze(-1).numpy(),
                dtype=np.dtype([("track_prob", np.float32)]),
            )

        calo_final = preds.get("calo_final", {})
        if "calo_mask" in calo_final:
            to_write["calo_mask"] = u2s(
                calo_final["calo_mask"]["calo_node_prob"].cpu().float().unsqueeze(-1).numpy(),
                dtype=np.dtype([("calo_prob", np.float32)]),
            )

        # --- Reconstruction outputs ---
        reco_final = preds.get("reco_final", {})
        if "classification" in reco_final:
            for key, val in reco_final["classification"].items():
                if "class" in key:
                    to_write["reco_class"] = u2s(
                        val.cpu().unsqueeze(-1).numpy(),
                        dtype=np.dtype([("reco_class", "i8")]),
                    )

        if "regression" in reco_final:
            reg_arrays = []
            for field in ["e", "pt", "eta", "sinphi", "cosphi"]:
                pred_key = f"reco_pflow_{field}"
                if pred_key in reco_final["regression"]:
                    pred_data = reco_final["regression"][pred_key].cpu().float().unsqueeze(-1)
                    if field in self.var_transform:
                        pred_data = self.var_transform[field].inverse_transform(pred_data)
                    reg_arrays.append(u2s(
                        pred_data.numpy(),
                        dtype=np.dtype([(f"pred_{field}", np.float32)]),
                    ))
            if reg_arrays:
                to_write["reco_regression"] = join_structured_arrays(reg_arrays)

        self._write_batch_outputs(to_write, batch_idx)

    def on_test_end(self, trainer, module):
        if self.writer is not None:
            print(f"Wrote predictions to {self.output_path}")
            self.writer.close()
