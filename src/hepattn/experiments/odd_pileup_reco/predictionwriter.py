"""Prediction writer for the merged pileup removal + reconstruction pipeline.

Writes pileup removal outputs (track mask, calo mask) and
reconstruction outputs (class, masks, incidence, regression) to HDF5/ROOT.
"""

from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from hepattn.utils.array_utils import join_structured_arrays, maybe_pad


def load_convert_h5(filepath):
    with h5py.File(filepath, "r") as f:
        pflow_class = f["object_class"]["pflow_class"][:]

        pflow_vars = [f"pred_{el}" for el in ["e", "pt", "eta", "sinphi", "cosphi"]]
        pflow_data = s2u(f["regression"].fields(pflow_vars)[:])

        pflow_ptetaphi = np.stack(
            [
                pflow_data[..., 1],
                pflow_data[..., 2],
                np.arctan2(pflow_data[..., 3], pflow_data[..., 4]),
            ],
            axis=-1,
        )

        proxy_ptetaphi = None
        proxy_vars = [f"proxy_{el}" for el in ["e", "pt", "eta", "sinphi", "cosphi"]]
        if all(v in f["regression"].dtype.names for v in proxy_vars):
            proxy_data = s2u(f["regression"].fields(proxy_vars)[:])
            proxy_ptetaphi = np.stack(
                [
                    proxy_data[..., 1],
                    proxy_data[..., 2],
                    np.arctan2(proxy_data[..., 3], proxy_data[..., 4]),
                ],
                axis=-1,
            )

        pflow_indicator = (pflow_class < 5) & (np.abs(pflow_ptetaphi[..., 1]) < 4)

        neutral_mask = (pflow_class < 5) & (pflow_class > 2)
        pflow_ptetaphi[neutral_mask][..., 0] = pflow_data[neutral_mask][..., 0] / np.cosh(pflow_ptetaphi[neutral_mask][..., 1])

        event_number = f["events"]["event_number"][:]

        return (
            event_number,
            pflow_class,
            pflow_ptetaphi,
            proxy_ptetaphi,
            pflow_indicator,
        )


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

    def _write_batch_outputs(self, batch_outputs, pad_masks, batch_idx):
        to_write = {}
        for input_name, outputs in batch_outputs.items():
            this_outputs = []
            name = input_name
            inputs = None

            for preds in outputs.values():
                if inputs is not None:
                    this_outputs.append(maybe_pad(preds, inputs))
                else:
                    this_outputs.append(preds)

            # add mask if present
            if name in pad_masks:
                pad_mask = pad_masks[name].cpu()
                pad_mask = u2s(np.expand_dims(pad_mask, -1), dtype=np.dtype([("mask", "?")]))
                this_outputs.append(maybe_pad(pad_mask, inputs))
            to_write[name] = join_structured_arrays(this_outputs)

        # If the writer hasn't been created yet, create it now that we have the dtypes and shapes
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
        to_write["events"] = {
            "event_number": u2s(
                targets["event_number"].cpu().numpy().astype(np.int64).reshape(-1, 1),
                dtype=np.dtype([("event_number", "i8")]),
            )
        }

        # --- Pileup removal outputs (Stream A & B) ---
        track_final = preds.get("track_final", {})
        if "mask" in track_final:
            to_write["track_mask"] = {
                "preds": u2s(
                    track_final["mask"]["pflow_node_prob"].cpu().float().unsqueeze(-1).numpy(),
                    dtype=np.dtype([("track_prob", np.float32)]),
                )
            }

        calo_final = preds.get("calo_final", {})
        if "calo_mask" in calo_final:
            to_write["calo_mask"] = {
                "preds": u2s(
                    calo_final["calo_mask"]["calo_node_prob"].cpu().float().unsqueeze(-1).numpy(),
                    dtype=np.dtype([("calo_prob", np.float32)]),
                )
            }

        # --- Reco node indices (1400 → 5500 mapping) ---
        reco_node_indices = getattr(module.model.model, "_reco_node_indices", None)
        if reco_node_indices is not None:
            to_write["reco_node_indices"] = {
                "indices": u2s(
                    reco_node_indices.cpu().numpy().astype(np.int32).reshape(*reco_node_indices.shape, 1),
                    dtype=np.dtype([("index", np.int32)]),
                )
            }

        # --- Reconstruction outputs (Stream C) ---
        reco_final_preds = preds.get("reco_final", {})
        reco_final_outputs = outputs.get("reco_final", {})

        # Object class: truth + predicted
        to_write["object_class"] = {}
        if "reco_particle_class" in targets:
            to_write["object_class"]["targets"] = u2s(
                targets["reco_particle_class"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("object_class", "i8")]),
            )
        if "classification" in reco_final_preds:
            for key, val in reco_final_preds["classification"].items():
                if "class" in key:
                    to_write["object_class"]["preds"] = u2s(
                        val.cpu().unsqueeze(-1).numpy(),
                        dtype=np.dtype([("pflow_class", "i8")]),
                    )

        # Object masks: truth + raw logits
        to_write["object_masks"] = {}
        if "reco_particle_node_valid" in targets:
            to_write["object_masks"]["targets"] = u2s(
                targets["reco_particle_node_valid"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_masks", "i8")]),
            )
        if "mask" in reco_final_outputs:
            logit_key = next((k for k in reco_final_outputs["mask"] if k.endswith("_logit")), None)
            if logit_key:
                to_write["object_masks"]["preds"] = u2s(
                    reco_final_outputs["mask"][logit_key].cpu().unsqueeze(-1).float().numpy(),
                    dtype=np.dtype([("mask_logits", np.float32)]),
                )

        # Incidence: truth + predicted
        if "reco_particle_incidence" in targets and "incidence" in reco_final_preds:
            to_write["incidence"] = {}
            to_write["incidence"]["targets"] = u2s(
                targets["reco_particle_incidence"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_incidence", np.float32)]),
            )
            incidence_key = next((k for k in reco_final_preds["incidence"] if "incidence" in k), None)
            if incidence_key:
                to_write["incidence"]["preds"] = u2s(
                    reco_final_preds["incidence"][incidence_key].cpu().unsqueeze(-1).float().numpy(),
                    dtype=np.dtype([("pred_incidence", np.float32)]),
                )

        # Regression: truth + pred + proxy for each kinematic
        to_write["regression"] = {}
        for t in ["e", "pt", "eta", "sinphi", "cosphi"]:
            # Truth
            truth_key = f"reco_particle_{t}"
            if truth_key in targets:
                truth_data = targets[truth_key].cpu().float().unsqueeze(-1)
                if t in self.var_transform:
                    truth_data = self.var_transform[t].inverse_transform(truth_data)
                to_write["regression"][f"truth_{t}"] = u2s(
                    truth_data.numpy(),
                    dtype=np.dtype([(f"truth_{t}", np.float32)]),
                )

            # Predicted
            if "regression" in reco_final_preds:
                pred_key = f"reco_pflow_{t}"
                if pred_key in reco_final_preds["regression"]:
                    pred_data = reco_final_preds["regression"][pred_key].cpu().float().unsqueeze(-1)
                    if t in self.var_transform:
                        pred_data = self.var_transform[t].inverse_transform(pred_data)
                    to_write["regression"][f"pred_{t}"] = u2s(
                        pred_data.numpy(),
                        dtype=np.dtype([(f"pred_{t}", np.float32)]),
                    )

                # Proxy (if available)
                proxy_key = f"reco_pflow_proxy_{t}"
                if proxy_key in reco_final_preds["regression"]:
                    proxy_data = reco_final_preds["regression"][proxy_key].cpu().float().unsqueeze(-1)
                    if t in self.var_transform:
                        proxy_data = self.var_transform[t].inverse_transform(proxy_data)
                    to_write["regression"][f"proxy_{t}"] = u2s(
                        proxy_data.numpy(),
                        dtype=np.dtype([(f"proxy_{t}", np.float32)]),
                    )

        # Remove empty groups
        to_write = {k: v for k, v in to_write.items() if v}

        self._write_batch_outputs(to_write, {}, batch_idx)

    def on_test_end(self, trainer, module):
        if self.writer is not None:
            print(f"Wrote predictions to {self.output_path}")
            self.writer.close()

        print("Loading predictions...")
        event_number, pflow_class, pflow_ptetaphi, proxy_ptetaphi, pflow_indicator = load_convert_h5(self.output_path.as_posix())
        root_path = self.output_path.with_suffix(".root").as_posix()
        print("Writing to ROOT file")
        root_data = {
            "mpflow": {
                "pt": ak.Array(pflow_ptetaphi[..., 0]),
                "eta": ak.Array(pflow_ptetaphi[..., 1]),
                "phi": ak.Array(pflow_ptetaphi[..., 2]),
                "class": ak.Array(pflow_class),
            },
            "pred_ind": ak.Array(pflow_indicator),
            "event_number": ak.Array(event_number)[: len(pflow_indicator)],
        }
        if proxy_ptetaphi is not None:
            root_data["proxy"] = {
                "pt": ak.Array(proxy_ptetaphi[..., 0]),
                "eta": ak.Array(proxy_ptetaphi[..., 1]),
                "phi": ak.Array(proxy_ptetaphi[..., 2]),
            }
        with uproot.recreate(root_path) as f:
            f["event_tree"] = root_data
        print(f"Wrote ROOT file to {root_path}")
