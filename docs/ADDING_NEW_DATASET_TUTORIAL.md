# Tutorial: Adding a New Dataset Type to HepaTTN

This tutorial explains how to add a new experiment/dataset type to the `hepattn` framework. The framework is designed around PyTorch Lightning and uses a modular architecture that separates data loading, model definition, and training orchestration.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Create the Experiment Directory](#step-1-create-the-experiment-directory)
   - [Step 2: Implement the Dataset Class](#step-2-implement-the-dataset-class)
   - [Step 3: Implement the DataModule Class](#step-3-implement-the-datamodule-class)
   - [Step 4: Implement the Model (LightningModule)](#step-4-implement-the-model-lightningmodule)
   - [Step 5: Create the Configuration File](#step-5-create-the-configuration-file)
   - [Step 6: Create the Main Entry Point](#step-6-create-the-main-entry-point)
4. [Key Patterns and Conventions](#key-patterns-and-conventions)
5. [Example: Minimal New Dataset](#example-minimal-new-dataset)
6. [Testing Your Implementation](#testing-your-implementation)

---

## Overview

The `hepattn` framework currently supports several experiment types located in `src/hepattn/experiments/`:

| Experiment | Description | Data Format |
|------------|-------------|-------------|
| `clic` | CLIC particle flow reconstruction | ROOT files |
| `cld` | CLD detector reconstruction | NPZ files (per-event) |
| `trackml` | TrackML tracking challenge | Parquet files |
| `tide` | TIDE b-tagging | HDF5 files |
| `pixel` | Pixel cluster splitting | HDF5 files |
| `itk` | ITk tracking | Various |

Each experiment follows a consistent pattern with the following components:

1. **Dataset class** (`data.py`) - Extends `torch.utils.data.Dataset`
2. **DataModule class** (`data.py`) - Extends `lightning.LightningDataModule`
3. **Model class** (`model.py`) - Extends `hepattn.models.wrapper.ModelWrapper`
4. **Configuration** (`configs/*.yaml`) - YAML configuration files
5. **Main script** (`main.py`) - Entry point using `hepattn.utils.cli.CLI`

---

## Directory Structure

Create the following structure for your new experiment:

```
src/hepattn/experiments/your_experiment/
├── __init__.py
├── data.py              # Dataset and DataModule classes
├── model.py             # LightningModule wrapper
├── main.py              # Training entry point
├── prep.py              # (Optional) Data preprocessing script
├── eval.py              # (Optional) Evaluation utilities
├── configs/
│   └── base.yaml        # Default configuration
└── README.md            # (Optional) Experiment-specific documentation
```

---

## Step-by-Step Guide

### Step 1: Create the Experiment Directory

```bash
mkdir -p src/hepattn/experiments/your_experiment/configs
touch src/hepattn/experiments/your_experiment/__init__.py
```

### Step 2: Implement the Dataset Class

The Dataset class is responsible for loading individual samples. Here's the pattern used across all experiments:

```python
# src/hepattn/experiments/your_experiment/data.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(
        self,
        dirpath: str,                    # Path to data directory or file
        inputs: dict,                    # Dict mapping input names to field lists
        targets: dict,                   # Dict mapping target names to field lists
        num_events: int = -1,            # Number of events to load (-1 = all)
        # Add your experiment-specific parameters here
        particle_min_pt: float = 1.0,    # Example: selection cut
        precision: str = "32",           # Data precision
    ):
        super().__init__()
        
        # Store configuration
        self.dirpath = dirpath
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.particle_min_pt = particle_min_pt
        
        # Set precision type
        self.precision_type = {
            "16": torch.float16,
            "bf16": torch.bfloat16,
            "32": torch.float32,
            "64": torch.float64,
        }[str(precision)]
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
        
        # Discover and register available samples
        self._discover_samples()
    
    def _discover_samples(self):
        """Find all available samples in the data directory."""
        # Example for per-event files (like CLD):
        self.sample_files = list(Path(self.dirpath).glob("*.npz"))
        
        # Limit to requested number of events
        if self.num_events > 0:
            self.sample_files = self.sample_files[:self.num_events]
        else:
            self.num_events = len(self.sample_files)
        
        # Create sample IDs for tracking
        self.sample_ids = np.arange(self.num_events)
        
        print(f"Found {self.num_events} samples in {self.dirpath}")
    
    def __len__(self) -> int:
        return int(self.num_events)
    
    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """Load and return a single sample.
        
        Returns:
            inputs: Dict of input tensors with batch dimension added
            targets: Dict of target tensors with batch dimension added
        """
        # Load raw event data
        event = self._load_event(idx)
        
        # Apply any preprocessing/cuts
        event = self._preprocess_event(event)
        
        # Build input tensors
        inputs = {}
        for input_name, fields in self.inputs.items():
            # Add validity mask
            inputs[f"{input_name}_valid"] = torch.from_numpy(
                event[f"{input_name}_valid"]
            ).bool().unsqueeze(0)
            
            # Add feature fields
            for field in fields:
                inputs[f"{input_name}_{field}"] = torch.from_numpy(
                    event[f"{input_name}_{field}"]
                ).to(self.precision_type).unsqueeze(0)
        
        # Build target tensors
        targets = {}
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = torch.from_numpy(
                event[f"{target_name}_valid"]
            ).bool().unsqueeze(0)
            
            for field in fields:
                targets[f"{target_name}_{field}"] = torch.from_numpy(
                    event[f"{target_name}_{field}"]
                ).to(self.precision_type).unsqueeze(0)
        
        # Add sample ID for tracking
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int64)
        
        return inputs, targets
    
    def _load_event(self, idx: int) -> dict:
        """Load raw event data from storage."""
        # Implement based on your data format
        # Example for NPZ:
        with np.load(self.sample_files[idx], allow_pickle=True) as archive:
            return {key: archive[key] for key in archive.files}
    
    def _preprocess_event(self, event: dict) -> dict:
        """Apply preprocessing and selection cuts."""
        # Add derived quantities (e.g., cylindrical coordinates)
        # Apply cuts (e.g., minimum pT)
        # Build validity masks
        return event

### Real-world Example: CLIC Dataset

The CLIC dataset implementation (`src/hepattn/experiments/clic/pflow_data.py`) demonstrates handling ROOT files with `uproot` and complex feature engineering:

```python
class CLICDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        inputs: dict,
        targets: dict,
        scale_dict_path: str,
        num_events: int = -1,
        num_objects: int = 150,
        max_nodes: int = 160,
        remove_wrong_idxs: bool = True,
        incidence_cutval: float = 1e-4,
        is_inference: bool = False,
    ):
        super().__init__()
        self.scaler = FeatureScaler(scale_dict_path)
        self.filepath = filepath
        self.inputs = inputs
        self.targets = targets
        # ... initialization ...

        with uproot.open(filepath, num_workers=6) as f:
            tree = f["events"]
            self.num_events = tree.num_entries if num_events == -1 else num_events
            self.load_data(tree)

    def load_data(self, tree):
        # Load arrays into memory for faster access
        varlist = self.track_variables + self.topo_variables + self.particle_variables
        self.full_data_array = {}
        # ... loading logic ...
        
        # Pre-calculate cumulative sums for indexing
        self.topo_cumsum = np.cumsum([0, *self.n_topos.tolist()])
        self.track_cumsum = np.cumsum([0, *self.n_tracks.tolist()])
        self.particle_cumsum = np.cumsum([0, *self.n_particles.tolist()])

    def __getitem__(self, idx):
        # Load event data using pre-calculated indices
        # ...
        
        # Apply scaling
        for key in self.inputs["hit"]:
             # ... transformation logic ...
        
        return inputs, targets
```
```

**Key conventions:**

1. **Inputs dict**: Maps input names to lists of field names. For example:
   ```python
   inputs = {
       "hit": ["x", "y", "z", "energy"],
       "track": ["pt", "eta", "phi"]
   }
   ```

2. **Targets dict**: Same structure for targets:
   ```python
   targets = {
       "particle": ["pt", "eta", "phi", "mass"],
       "particle_hit": []  # For mask targets
   }
   ```

3. **Validity masks**: Every input/target type needs a `{name}_valid` boolean mask tensor.

4. **Batch dimension**: Always add `.unsqueeze(0)` to create a batch dimension of 1.

5. **Sample IDs**: Include for tracking samples during evaluation.

### Step 3: Implement the DataModule Class

The DataModule orchestrates train/val/test dataset creation and DataLoader setup:

```python
# Continue in src/hepattn/experiments/your_experiment/data.py

from lightning import LightningDataModule
from torch.utils.data import DataLoader

class YourDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int = 1,
        test_dir: str | None = None,
        pin_memory: bool = True,
        **kwargs,  # Pass through to Dataset
    ):
        super().__init__()
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.kwargs = kwargs  # Dataset-specific arguments
    
    def setup(self, stage: str):
        """Called by Lightning to set up datasets for each stage."""
        if stage in {"fit", "test"}:
            self.train_dataset = YourDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                **self.kwargs
            )
        
        if stage == "fit":
            self.val_dataset = YourDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                **self.kwargs
            )
            print(f"Training: {len(self.train_dataset):,} events")
            print(f"Validation: {len(self.val_dataset):,} events")
        
        if stage == "test":
            assert self.test_dir is not None, "No test_dir specified"
            self.test_dataset = YourDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                **self.kwargs
            )
            print(f"Test: {len(self.test_dataset):,} events")
    
    def _get_dataloader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            collate_fn=None,  # Or custom collator for variable-size batching
        )
    
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

### Real-world Example: CLIC DataModule

The CLIC DataModule (`src/hepattn/experiments/clic/pflow_data.py`) shows how to handle train/val/test splits and pass configuration:

```python
class PflowDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        scale_dict_path: str,
        test_path: str | None = None,
        pin_memory: bool = True,
        test_suff: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        # ...
        self.kwargs = kwargs  # Config parameters for Dataset

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = CLICDataset(
                filepath=self.train_path,
                num_events=self.num_train,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )
            self.val_dset = CLICDataset(
                filepath=self.valid_path,
                num_events=self.num_val,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )

        if stage == "test":
            self.test_dset = CLICDataset(
                filepath=self.test_path,
                num_events=self.num_test,
                scale_dict_path=self.scale_dict_path,
                is_inference=True,
                **self.kwargs,
            )
```
```

**Optional: Custom Collator**

For variable-sized inputs that need padding:

```python
from hepattn.utils.tensor_utils import pad_to_size

class YourCollator:
    def __init__(self, inputs, targets, max_num_objects):
        self.inputs = inputs
        self.targets = targets
        self.max_num_objects = max_num_objects
    
    def __call__(self, batch):
        inputs_list, targets_list = zip(*batch)
        
        # Find max sizes in this batch
        max_hits = max(inp["hit_valid"].shape[-1] for inp in inputs_list)
        
        # Pad and concatenate
        batched_inputs = {}
        batched_targets = {}
        
        for key in inputs_list[0]:
            tensors = [inp[key] for inp in inputs_list]
            # Pad to max size and concatenate along batch dim
            batched_inputs[key] = torch.cat([
                pad_to_size(t, target_size, pad_value=0)
                for t in tensors
            ], dim=0)
        
        # Similar for targets...
        
        return batched_inputs, batched_targets
```

### Step 4: Implement the Model (LightningModule)

The model wrapper extends `ModelWrapper` from `hepattn.models.wrapper`:

```python
# src/hepattn/experiments/your_experiment/model.py

from torch import nn
from hepattn.models.wrapper import ModelWrapper

class YourModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,      # The actual neural network
        lrs_config: dict,       # Learning rate scheduler config
        optimizer: str = "AdamW",
        mtl: bool = False,      # Multi-task learning
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
    
    def log_custom_metrics(self, preds, targets, stage):
        """Log experiment-specific metrics.
        
        Called automatically after each validation/test step.
        
        Args:
            preds: Dict of predictions from model.predict()
            targets: Dict of target tensors
            stage: "train", "val", or "test"
        """
        # Example: Log efficiency/purity metrics
        if stage == "train":
            return  # Skip detailed metrics during training for speed
        
        # Access predictions from final layer
        final_preds = preds.get("final", {})
        
        # Calculate and log metrics
        # Example for object detection:
        pred_valid = final_preds.get("object_valid", {}).get("object_valid")
        true_valid = targets.get("particle_valid")
        
        if pred_valid is not None and true_valid is not None:
            efficiency = (pred_valid & true_valid).sum() / true_valid.sum()
            self.log(f"{stage}/efficiency", efficiency, sync_dist=True)

### Real-world Example: CLIC Model

The CLIC model wrapper (`src/hepattn/experiments/clic/lightning_module.py`) implements custom metrics for particle flow reconstruction:

```python
class MPflow(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        
        # Initialize metrics
        self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=6, average="micro")
        self.eff = tm.classification.BinaryRecall()
        self.pur = tm.classification.BinaryPrecision()

    def log_custom_metrics(self, preds, labels, stage):
        if stage == "train":
            return

        # Extract predictions and labels
        particle_class_labels = labels["particle_class"].squeeze()
        particle_class_preds = preds["classification"]["pflow_class"].squeeze()

        # Log classification metrics
        self.obj_accuracy_micro(particle_class_preds.view(-1), particle_class_labels.view(-1))
        self.log(f"{stage}/obj_class_accuracy_micro", self.obj_accuracy_micro, sync_dist=True)

        # Log efficiency/purity
        pred_valid = particle_class_preds < 5
        truth_valid = particle_class_labels < 5
        
        self.eff(pred_valid, truth_valid)
        self.pur(pred_valid, truth_valid)
        self.log(f"{stage}/eff", self.eff, sync_dist=True)
        self.log(f"{stage}/pur", self.pur, sync_dist=True)
```
```

The `ModelWrapper` base class provides:

- `training_step`, `validation_step`, `test_step` implementations
- Loss logging via `log_losses()`
- Learning rate scheduling via `configure_optimizers()`
- Multi-task learning support

### Step 5: Create the Configuration File

Configuration uses YAML and is parsed by `jsonargparse`:

```yaml
# src/hepattn/experiments/your_experiment/configs/base.yaml

name: your_experiment_v1
seed_everything: 42

# Data configuration
data:
  train_dir: /path/to/train/
  val_dir: /path/to/val/
  test_dir: /path/to/test/
  
  num_workers: 8
  num_train: -1      # -1 means all available
  num_val: 1000
  num_test: 1000
  batch_size: 32
  
  precision: "32"
  
  # Define input features
  inputs:
    hit:
      - x
      - y
      - z
      - energy
    track:
      - pt
      - eta
      - phi
  
  # Define target features
  targets:
    particle:
      - pt
      - eta
      - phi
      - mass
    particle_hit: []  # Mask assignment target
  
  # Experiment-specific parameters
  particle_min_pt: 1.0
  event_max_num_particles: &num_particles 256

# Trainer configuration
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  gradient_clip_val: 0.1
  log_every_n_steps: 50
  default_root_dir: logs
  
  logger:
    class_path: lightning.pytorch.loggers.CometLogger
    init_args:
      project_name: your-project
  
  callbacks:
    - class_path: hepattn.callbacks.Compile
    - class_path: hepattn.callbacks.SaveConfig
    - class_path: hepattn.callbacks.Checkpoint
      init_args:
        monitor: val/loss
        save_last: true
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
    - class_path: lightning.pytorch.callbacks.TQDMProgressBar

# Model configuration
model:
  optimizer: AdamW
  
  lrs_config:
    initial: 1e-6
    max: 1e-4
    end: 1e-6
    pct_start: 0.05
    weight_decay: 1e-4
    skip_scheduler: false
  
  mtl: false
  
  model:
    class_path: hepattn.models.MaskFormer
    init_args:
      dim: &dim 256
      num_queries: *num_particles
      # ... model-specific config

### Real-world Example: CLIC Configuration

The CLIC configuration (`src/hepattn/experiments/clic/configs/base.yaml`) defines the full experiment setup:

```yaml
name: clic_v6
seed_everything: 42

data:
  inputs:
    hit: ["x", "y", "z", "r", "s", "theta", "phi"]
  targets:
    particle: ["e", "pt", "eta", "sinphi", "cosphi"]

  num_objects: &num_particles 150
  train_path: /path/to/train.root
  valid_path: /path/to/val.root
  
  num_workers: 16
  batch_size: 512
  scale_dict_path: configs/clic_var_transform.yaml

trainer:
  max_epochs: 200
  accelerator: gpu
  devices: 1
  logger:
    class_path: lightning.pytorch.loggers.CometLogger
    init_args:
      project: clic-trial

model:
  optimizer: Lion
  model:
    class_path: hepattn.models.MaskFormer
    init_args:
      dim: 256
      raw_variables: ["node_e", "node_pt", "node_eta", "node_sinphi", "node_cosphi"]
```
```

### Step 6: Create the Main Entry Point

```python
# src/hepattn/experiments/your_experiment/main.py

import pathlib
from lightning.pytorch.cli import ArgsType

from hepattn.experiments.your_experiment.data import YourDataModule
from hepattn.experiments.your_experiment.model import YourModel
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"

def main(args: ArgsType = None) -> None:
    CLI(
        model_class=YourModel,
        datamodule_class=YourDataModule,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}
        },
    )

if __name__ == "__main__":
    main()

### Real-world Example: CLIC Main Script

The CLIC main script (`src/hepattn/experiments/clic/main.py`) links the specific DataModule and Model classes:

```python
from lightning.pytorch.cli import ArgsType
from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.utils.cli import CLI

def main(args: ArgsType = None) -> None:
    CLI(
        model_class=MPflow,
        datamodule_class=PflowDataModule,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": ["configs/base.yaml"]}
        },
    )

if __name__ == "__main__":
    main()
```
```

---

## Key Patterns and Conventions

### 1. Input/Target Naming Convention

```python
# For an input type "hit" with fields ["x", "y", "z"]:
inputs = {
    "hit_valid": torch.tensor([True, True, False, ...]),  # Validity mask
    "hit_x": torch.tensor([0.1, 0.2, 0.0, ...]),
    "hit_y": torch.tensor([0.3, 0.4, 0.0, ...]),
    "hit_z": torch.tensor([0.5, 0.6, 0.0, ...]),
}

# For a target type "particle_hit" (linking particles to hits):
targets = {
    "particle_hit_valid": torch.tensor([[True, True, False], ...]),  # 2D mask
}
```

### 2. Data Formats Supported

| Format | Example Usage | Loading Method |
|--------|---------------|----------------|
| ROOT | CLIC | `uproot.open()` |
| NPZ | CLD | `np.load()` |
| HDF5 | TIDE, Pixel | `h5py.File()` |
| Parquet | TrackML | `pd.read_parquet()` |

### 3. Common Preprocessing Operations

```python
# Convert mm to m
event["hit.x"] *= 0.001

# Add cylindrical coordinates
event["hit.r"] = np.sqrt(event["hit.x"]**2 + event["hit.y"]**2)
event["hit.phi"] = np.arctan2(event["hit.y"], event["hit.x"])
event["hit.theta"] = np.arccos(event["hit.z"] / event["hit.s"])
event["hit.eta"] = -np.log(np.tan(event["hit.theta"] / 2))

# Add sin/cos of angles (better for learning)
event["hit.sinphi"] = np.sin(event["hit.phi"])
event["hit.cosphi"] = np.cos(event["hit.phi"])
```

### 4. Feature Scaling

Use `hepattn.utils.scaling.FeatureScaler` for standardization:

```python
from hepattn.utils.scaling import FeatureScaler

# In Dataset.__init__:
self.scaler = FeatureScaler(scale_dict_path)

# In __getitem__:
scaled_pt = self.scaler.transforms["pt"].transform(raw_pt)
```

---

## Example: Minimal New Dataset

Here's a complete minimal example for a new "simple_tracking" experiment:

```python
# src/hepattn/experiments/simple_tracking/data.py

from pathlib import Path
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SimpleTrackingDataset(Dataset):
    def __init__(self, dirpath: str, inputs: dict, targets: dict, num_events: int = -1):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.files = list(Path(dirpath).glob("*.npz"))[:num_events if num_events > 0 else None]
        self.num_events = len(self.files)
    
    def __len__(self):
        return self.num_events
    
    def __getitem__(self, idx):
        data = dict(np.load(self.files[idx]))
        
        inputs = {"hit_valid": torch.from_numpy(data["hit_valid"]).bool().unsqueeze(0)}
        for field in self.inputs["hit"]:
            inputs[f"hit_{field}"] = torch.from_numpy(data[f"hit_{field}"]).float().unsqueeze(0)
        
        targets = {"particle_valid": torch.from_numpy(data["particle_valid"]).bool().unsqueeze(0)}
        targets["particle_hit_valid"] = torch.from_numpy(data["particle_hit_valid"]).bool().unsqueeze(0)
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int64)
        
        return inputs, targets


class SimpleTrackingDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, num_workers, num_train, num_val, num_test, 
                 test_dir=None, batch_size=1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
    
    def setup(self, stage):
        hp = self.hparams
        if stage == "fit":
            self.train_ds = SimpleTrackingDataset(hp.train_dir, num_events=hp.num_train, **self.kwargs)
            self.val_ds = SimpleTrackingDataset(hp.val_dir, num_events=hp.num_val, **self.kwargs)
        if stage == "test":
            self.test_ds = SimpleTrackingDataset(hp.test_dir, num_events=hp.num_test, **self.kwargs)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
```

---

## Testing Your Implementation

1. **Unit test the Dataset**:
```python
# Test loading a single sample
dataset = YourDataset(dirpath="path/to/data", inputs={...}, targets={...})
inputs, targets = dataset[0]
print(inputs.keys(), targets.keys())
```

2. **Test the DataModule**:
```python
from lightning import Trainer

dm = YourDataModule(train_dir="...", val_dir="...", ...)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print(batch[0].keys())  # inputs
print(batch[1].keys())  # targets
```

3. **Run training**:
```bash
cd src/hepattn/experiments/your_experiment
python main.py fit --config configs/base.yaml
```

---

## Summary Checklist

- [ ] Created experiment directory structure
- [ ] Implemented `Dataset` class with `__len__` and `__getitem__`
- [ ] Implemented `DataModule` class with `setup()` and dataloaders
- [ ] Implemented model wrapper extending `ModelWrapper`
- [ ] Created YAML configuration file
- [ ] Created `main.py` entry point
- [ ] Input tensors have `{name}_valid` masks
- [ ] All tensors have batch dimension via `.unsqueeze(0)`
- [ ] Sample IDs included in targets for tracking
- [ ] Tested data loading end-to-end

For questions or issues, refer to existing implementations in `hepattn/experiments/cld/` or `hepattn/experiments/trackml/` as reference.
