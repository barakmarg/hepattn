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
    def __init__(
        self,
        events_per_file: int | None = None,
        write_root: bool = True,
        predict_only: bool = True,
    ) -> None:
        super().__init__()
        # If set, predictions are split across multiple H5 files holding at most
        # ``events_per_file`` events each (the final file holds the remainder).
        # If None, everything is written to a single file (original behaviour).
        self.events_per_file = events_per_file
        # If False, skip the per-file ROOT conversion (H5 output only).
        self.write_root = write_root
        # If True, the LightningModule skips the test-time loss (and its
        # Hungarian matching) during the forward pass — much faster, and the
        # writer never uses the loss anyway. Set False to keep loss/metrics.
        self.predict_only = predict_only

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        # Tell the LightningModule whether to skip the (unused, expensive) loss.
        module._predict_only = self.predict_only

        self.writer = None
        self.trainer = trainer
        self.batch_size = trainer.datamodule.batch_size
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_suff = trainer.datamodule.test_suff
        self.num_events = len(self.ds)
        self.var_transform = self.ds.scaler.transforms

        # Multi-file chunking state. A batch may straddle a file boundary, so
        # _write_batch splits batches across files as needed.
        self._chunked = self.events_per_file is not None
        self._chunk_size = self.events_per_file if self._chunked else self.num_events
        self._file_idx = 0            # index of the file currently being written
        self._events_in_file = 0      # events written into the current file
        self._events_total = 0        # events written across all files
        self._file_capacity = 0       # allocated rows of the current file
        self._current_path: Path | None = None
        self._output_paths: list[Path] = []

    def _output_path_for(self, part_idx: int | None) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        base = f"{out_basename}__test{suffix}"
        if part_idx is None:
            # Single-file mode: write directly next to the checkpoint.
            return Path(out_dir / f"{base}.h5")
        # Chunked mode: group the per-1k-event part files in their own folder
        # (named like the single-file output) next to the checkpoint.
        return Path(out_dir / base / f"{base}__part{part_idx:03d}.h5")

    @property
    def output_path(self) -> Path:
        """Primary output file (the only file in single-file mode)."""
        return self._output_path_for(self._file_idx if self._chunked else None)

    @property
    def output_paths(self) -> list[Path]:
        """All output files written so far (populated as files are closed)."""
        return list(self._output_paths)

    def _open_writer(self, template: dict) -> None:
        """Open an H5Writer for the next output file, sized to its capacity."""
        from ftag.hdf5 import H5Writer
        self._file_capacity = min(self._chunk_size, self.num_events - self._events_total)
        part_idx = self._file_idx if self._chunked else None
        self._current_path = self._output_path_for(part_idx)
        self._current_path.parent.mkdir(parents=True, exist_ok=True)
        dtypes = {k: v.dtype for k, v in template.items()}
        shapes = {k: (self._file_capacity, *v.shape[1:]) for k, v in template.items()}
        self.writer = H5Writer(
            jets_name="events",
            dst=self._current_path,
            dtypes=dtypes,
            shapes=shapes,
            shuffle=False,
            precision="full",
        )
        self._events_in_file = 0

    def _close_writer(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self._output_paths.append(self._current_path)
            self.writer = None
            self._file_idx += 1

    def _write_batch(self, to_write):
        n_batch = next(iter(to_write.values())).shape[0]
        offset = 0
        while offset < n_batch:
            if self.writer is None:
                self._open_writer(to_write)
            take = min(self._file_capacity - self._events_in_file, n_batch - offset)
            chunk = {k: v[offset:offset + take] for k, v in to_write.items()}
            self.writer.write(chunk)
            self._events_in_file += take
            self._events_total += take
            offset += take
            if self._events_in_file >= self._file_capacity:
                self._close_writer()

    def on_test_batch_end(self, trainer, module, test_step_outputs, batch, batch_idx):
        _inputs, targets = batch
        outputs, preds, _losses = test_step_outputs

        to_write = {}

        # --- Event numbers ---
        to_write["events"] = u2s(
            targets["event_number"].cpu().numpy().astype(np.int64).reshape(-1, 1),
            dtype=np.dtype([("event_number", "i8")]),
        )

        # --- Pileup removal outputs (Stream A & B) — full 5500-node space ---
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

        # --- Reco node indices: (B, 1400) maps filtered positions → full 5500 space ---
        reco_node_indices = getattr(module.model, "_reco_node_indices", None)
        if reco_node_indices is not None:
            to_write["reco_node_indices"] = u2s(
                reco_node_indices.cpu().numpy().astype(np.int32).reshape(*reco_node_indices.shape, 1),
                dtype=np.dtype([("index", np.int32)]),
            )

        # --- Reconstruction outputs (Stream C) ---
        reco_final_preds = preds.get("reco_final", {})
        reco_final_outputs = outputs.get("reco_final", {})

        # Object class: truth + predicted — both (B, 400), joinable
        class_arrays = []
        if "reco_particle_class" in targets:
            class_arrays.append(u2s(
                targets["reco_particle_class"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("object_class", "i8")]),
            ))
        if "classification" in reco_final_preds:
            for key, val in reco_final_preds["classification"].items():
                if "class" in key:
                    class_arrays.append(u2s(
                        val.cpu().unsqueeze(-1).numpy(),
                        dtype=np.dtype([("pflow_class", "i8")]),
                    ))
        if class_arrays:
            to_write["object_class"] = join_structured_arrays(class_arrays)

        # Truth masks — (B, 400, 5500) full node space
        if "reco_particle_node_valid" in targets:
            to_write["truth_masks"] = u2s(
                targets["reco_particle_node_valid"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_masks", "i8")]),
            )

        # Pred mask logits — (B, 400, 1400) filtered space, use reco_node_indices to map
        if "mask" in reco_final_outputs:
            logit_key = next((k for k in reco_final_outputs["mask"] if k.endswith("_logit")), None)
            if logit_key:
                to_write["pred_mask_logits"] = u2s(
                    reco_final_outputs["mask"][logit_key].cpu().unsqueeze(-1).float().numpy(),
                    dtype=np.dtype([("mask_logits", np.float32)]),
                )

        # Truth incidence — (B, 400, 5500) full node space
        if "reco_particle_incidence" in targets:
            to_write["truth_incidence"] = u2s(
                targets["reco_particle_incidence"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_incidence", np.float32)]),
            )

        # Pred incidence — (B, 400, 1400) filtered space
        if "incidence" in reco_final_preds:
            incidence_key = next((k for k in reco_final_preds["incidence"] if "incidence" in k), None)
            if incidence_key:
                to_write["pred_incidence"] = u2s(
                    reco_final_preds["incidence"][incidence_key].cpu().unsqueeze(-1).float().numpy(),
                    dtype=np.dtype([("pred_incidence", np.float32)]),
                )

        # Regression: truth + pred + proxy — all (B, 400), joinable
        reg_arrays = []
        for t in ["e", "pt", "eta", "sinphi", "cosphi"]:
            truth_key = f"reco_particle_{t}"
            if truth_key in targets:
                truth_data = targets[truth_key].cpu().float().unsqueeze(-1)
                if t in self.var_transform:
                    truth_data = self.var_transform[t].inverse_transform(truth_data)
                reg_arrays.append(u2s(truth_data.numpy(), dtype=np.dtype([(f"truth_{t}", np.float32)])))

            if "regression" in reco_final_preds:
                pred_key = f"reco_pflow_{t}"
                if pred_key in reco_final_preds["regression"]:
                    pred_data = reco_final_preds["regression"][pred_key].cpu().float().unsqueeze(-1)
                    if t in self.var_transform:
                        pred_data = self.var_transform[t].inverse_transform(pred_data)
                    reg_arrays.append(u2s(pred_data.numpy(), dtype=np.dtype([(f"pred_{t}", np.float32)])))

                proxy_key = f"reco_pflow_proxy_{t}"
                if proxy_key in reco_final_preds["regression"]:
                    proxy_data = reco_final_preds["regression"][proxy_key].cpu().float().unsqueeze(-1)
                    if t in self.var_transform:
                        proxy_data = self.var_transform[t].inverse_transform(proxy_data)
                    reg_arrays.append(u2s(proxy_data.numpy(), dtype=np.dtype([(f"proxy_{t}", np.float32)])))

        if reg_arrays:
            to_write["regression"] = join_structured_arrays(reg_arrays)

        # --- Node-level metadata for offline track/cluster evaluation ---
        node_meta_arrays = []
        for field, target_key, dtype in [
            ("node_valid", "node_valid", np.int8),
            ("node_is_track", "node_is_track", np.int8),
            ("tracks_mask", "tracks_mask", np.int8),
            ("node_pt", "node_pt", np.float32),
            ("node_eta", "node_eta", np.float32),
            ("node_phi", "node_phi", np.float32),
            ("node_z0", "node_z0", np.float32),
            ("node_e", "node_e", np.float32),
            ("calo_hs_energy", "calo_hard_scatter_energy", np.float32),
            ("calo_hs_frac", "calo_hard_scatter_energy_frac", np.float32),
            ("calo_neutral_e", "calo_hs_neutral_energy", np.float32),
            ("calo_charged_e", "calo_hs_charged_energy", np.float32),
        ]:
            if target_key in targets:
                val = targets[target_key].cpu().float().numpy()
                if val.ndim == 2:
                    val = val[..., np.newaxis]
                node_meta_arrays.append(u2s(val, dtype=np.dtype([(field, dtype)])))

        if node_meta_arrays:
            to_write["node_metadata"] = join_structured_arrays(node_meta_arrays)

        # --- Calo fraction predictions (if calo_fraction task is present) ---
        if "calo_fraction" in calo_final and "calo_hs_fraction" in calo_final.get("calo_fraction", {}):
            calo_frac_val = calo_final["calo_fraction"]["calo_hs_fraction"].cpu().float().numpy()
            if calo_frac_val.ndim == 2:
                calo_frac_val = calo_frac_val[..., np.newaxis]
            to_write["calo_fraction"] = u2s(
                calo_frac_val,
                dtype=np.dtype([("calo_frac_pred", np.float32)]),
            )

        self._write_batch(to_write)

    def on_test_end(self, trainer, module):
        # Flush any file still open (e.g. a short final/only file that never
        # reached its full capacity).
        self._close_writer()

        for h5_path in self._output_paths:
            print(f"Wrote predictions to {h5_path}")

        if not self.write_root:
            return

        print("Loading predictions...")
        for h5_path in self._output_paths:
            event_number, pflow_class, pflow_ptetaphi, proxy_ptetaphi, pflow_indicator = load_convert_h5(h5_path.as_posix())
            root_path = h5_path.with_suffix(".root").as_posix()
            print(f"Writing to ROOT file {root_path}")
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
