"""
ODD Particle Flow Prediction Writer.

Callback to save model predictions to HDF5 files during testing.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from hepattn.utils.array_utils import join_structured_arrays, maybe_pad


class ODDPredictionWriter(Callback):
    """
    Lightning Callback to write predictions to HDF5 files.

    Saves model outputs during test phase for offline analysis.
    """

    def __init__(self) -> None:
        """Initialize prediction writer."""
        super().__init__()

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        """
        Set up the prediction writer.

        Args:
            trainer: Lightning trainer
            module: Lightning module
            stage: Current stage (fit, validate, test)
        """
        if stage != "test":
            return

        self.writer = None

        # Basic properties from trainer
        self.trainer = trainer
        self.batch_size = trainer.datamodule.batch_size
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_suff = trainer.datamodule.test_suff
        self.num_events = len(self.ds)

        # Feature transforms for denormalization if needed
        self.var_transform = self.ds.scaler.transforms

    @property
    def output_path(self) -> Path:
        """Get output file path based on checkpoint path."""
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        return Path(out_dir / f"{out_basename}__test{suffix}.h5")

    def _init_writer(self, dtypes: dict, shapes: dict) -> None:
        """
        Initialize HDF5 writer with dtypes and shapes.

        TODO: Implement or use ftag H5Writer if available.

        Args:
            dtypes: Dictionary of output name -> numpy dtype
            shapes: Dictionary of output name -> shape tuple
        """
        # TODO: Use ftag.hdf5.H5Writer or implement custom writer
        # from ftag.hdf5 import H5Writer
        # self.writer = H5Writer(
        #     jets_name="events",
        #     dst=self.output_path,
        #     dtypes=dtypes,
        #     shapes=shapes,
        #     shuffle=False,
        #     precision="full",
        # )
        raise NotImplementedError("TODO: Implement HDF5 writer initialization")

    def _write_batch_outputs(
        self,
        batch_outputs: dict,
        pad_masks: dict,
        batch_idx: int
    ) -> None:
        """
        Write a batch of outputs to file.

        TODO: Customize based on your output structure.

        Args:
            batch_outputs: Dictionary of model outputs
            pad_masks: Dictionary of padding masks
            batch_idx: Current batch index
        """
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

            # Add mask if present
            if name in pad_masks:
                pad_mask = pad_masks[name].cpu()
                pad_mask = u2s(
                    np.expand_dims(pad_mask, -1),
                    dtype=np.dtype([("mask", "?")])
                )
                this_outputs.append(maybe_pad(pad_mask, inputs))

            to_write[name] = join_structured_arrays(this_outputs)

        # Initialize writer on first batch
        if self.writer is None:
            dtypes = {k: v.dtype for k, v in to_write.items()}
            shapes = {k: (self.num_events, *v.shape[1:]) for k, v in to_write.items()}
            self._init_writer(dtypes, shapes)

        # TODO: Write batch
        # self.writer.write(to_write)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        test_step_outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """
        Called at the end of each test batch.

        TODO: Customize output structure based on your model outputs.

        Args:
            trainer: Lightning trainer
            module: Lightning module
            test_step_outputs: Output from test_step
            batch: Current batch data
            batch_idx: Current batch index
        """
        _inputs, targets = batch
        outputs, preds, _losses = test_step_outputs
        outputs = outputs["final"]
        preds = preds["final"]

        to_write = {}

        # Event numbers
        to_write["events"] = {
            "event_number": u2s(
                targets["event_number"].cpu().numpy().astype(np.int64).reshape(-1, 1),
                dtype=np.dtype([("event_number", "i8")]),
            )
        }

        # Object class predictions and targets
        to_write["object_class"] = {}
        to_write["object_class"]["targets"] = u2s(
            targets["particle_class"].cpu().unsqueeze(-1).numpy(),
            dtype=np.dtype([("object_class", "i8")]),
        )

        if "classification" in preds:
            to_write["object_class"]["preds"] = u2s(
                preds["classification"]["pflow_class"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("pflow_class", "i8")]),
            )

        # TODO: Add mask predictions
        # if "mask" in preds:
        #     to_write["masks"] = {
        #         "pred_mask": preds["mask"]["pflow_node_valid"].cpu().numpy(),
        #         "truth_mask": targets["particle_node_valid"].cpu().numpy(),
        #     }

        # TODO: Add regression predictions
        # if "regression" in preds:
        #     reg_vars = ["energy", "eta", "phi", "pt"]
        #     for i, var in enumerate(reg_vars):
        #         to_write["regression"][f"pred_{var}"] = preds["regression"][..., i].cpu().numpy()
        #         to_write["regression"][f"truth_{var}"] = targets[f"particle_{var}"].cpu().numpy()

        # TODO: Write outputs
        # self._write_batch_outputs(to_write, {}, batch_idx)

    def on_test_end(self, trainer: Trainer, module: LightningModule) -> None:
        """
        Called at end of testing.

        Args:
            trainer: Lightning trainer
            module: Lightning module
        """
        # TODO: Close writer and finalize file
        # if self.writer is not None:
        #     self.writer.close()
        pass


# TODO: Add helper functions for post-processing predictions
def load_predictions(filepath: str) -> dict:
    """
    Load predictions from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary of arrays
    """
    with h5py.File(filepath, "r") as f:
        # TODO: Implement loading logic
        # event_numbers = f["events"]["event_number"][:]
        # pflow_class = f["object_class"]["pflow_class"][:]
        # ...
        pass
    raise NotImplementedError("TODO: Implement prediction loading")
