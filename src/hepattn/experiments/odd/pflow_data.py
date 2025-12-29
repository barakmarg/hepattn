"""
ODD (Open Data Detector) Particle Flow Dataset and DataModule.

Data format: Parquet files loaded via Polars, transformed to NumPy/PyTorch tensors.

Schema:
- truth_particles: event_id (u32), particle_id (list[u64]), pdg_id (list[i64]), energy (list[f32]),
                   eta (list[f32]), phi (list[f32]), px (list[f32]), py (list[f32]), pz (list[f32]),
                   charge (list[f32]), mass (list[f32]), has_track (list[bool])
- calo_clusters:   event_id (u32), cluster_id (list[i32]), total_cluster_energy (list[f64]),
                   cluster_cx (list[f32]), cluster_cy (list[f32]), cluster_cz (list[f32])
- tracks:          event_id (u32), d0 (list[f32]), z0 (list[f32]), phi (list[f32]), theta (list[f32]),
                   qop (list[f32]), majority_particle_id (list[u64]), hit_ids (list[list[u32]]),
                   track_id (list[u16])
"""

import gc
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import polars as pl
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hepattn.utils.scaling import FeatureScaler


def normalize_phi(phi: np.ndarray) -> np.ndarray:
    """Normalize phi angle to [-pi, pi]."""
    return np.arctan2(np.sin(phi), np.cos(phi))


def do_padding(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    """Pad tensor to max_len with zeros."""
    x = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
    x[: len(tensor)] = tensor
    return x


def is_valid_file(path: str | Path) -> bool:
    """Check if file exists and is non-empty."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ODDDataset(Dataset):
    """
    ODD Particle Flow Dataset.

    Loads parquet files with truth_particles, calo_clusters, and tracks data.
    Transforms to PyTorch tensors for model consumption.
    """

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
        """
        Initialize ODD Dataset.

        Args:
            filepath: Path to parquet file or directory containing parquet files
            inputs: Dictionary specifying input features
            targets: Dictionary specifying target variables
            scale_dict_path: Path to YAML file with feature scaling parameters
            num_events: Number of events to load (-1 for all)
            num_objects: Maximum number of particles (output objects)
            max_nodes: Maximum number of nodes (tracks + clusters)
            remove_wrong_idxs: Whether to remove events with invalid indices
            incidence_cutval: Threshold for incidence matrix values
            is_inference: Whether running in inference mode
        """
        super().__init__()

        self.sampling_seed = 42
        np.random.default_rng(self.sampling_seed)
        seed_everything(self.sampling_seed, workers=True)

        self.scaler = FeatureScaler(scale_dict_path)

        # Initialize class labels mapping (PDG ID -> class index)
        self.init_label_dicts()

        # Initialize variable lists
        self.init_variables_list()

        # Input file handling
        self.filepath = filepath
        assert is_valid_file(filepath), f"Invalid file: {filepath}"

        # Store configuration
        self.inputs = inputs
        self.targets = targets
        self.num_objects = num_objects
        self.max_nodes = max_nodes
        self.remove_wrong_idxs = remove_wrong_idxs
        self.incidence_cutval = incidence_cutval
        self.is_inference = is_inference

        print(f"Loading ODD dataset from {filepath} with {num_events} samples")
        print(f"Is inference: {self.is_inference}")

        # Load data from parquet
        self.load_data(num_events)
        gc.collect()

    def load_data(self, num_events: int) -> None:
        """
        Load data from parquet file(s) using Polars.

        TODO: Implement loading logic for your parquet schema:
        - truth_particles: particle_id, pdg_id, energy, eta, phi, px, py, pz, charge, mass, has_track
        - calo_clusters: cluster_id, total_cluster_energy, cluster_cx, cluster_cy, cluster_cz
        - tracks: d0, z0, phi, theta, qop, majority_particle_id, hit_ids, track_id
        """
        self.full_data_array = {}

        # TODO: Load parquet file(s)
        # df = pl.read_parquet(self.filepath)
        # or if multiple files:
        # df_particles = pl.read_parquet(f"{self.filepath}/truth_particles.parquet")
        # df_clusters = pl.read_parquet(f"{self.filepath}/calo_clusters.parquet")
        # df_tracks = pl.read_parquet(f"{self.filepath}/tracks.parquet")

        # TODO: Join dataframes on event_id if needed
        # df = df_particles.join(df_clusters, on="event_id").join(df_tracks, on="event_id")

        # TODO: Limit number of events
        # if num_events > 0:
        #     df = df.head(num_events)

        # TODO: Extract arrays and convert to numpy
        # Example for truth particles:
        # self.full_data_array["particle_energy"] = df["energy"].to_numpy()
        # self.full_data_array["particle_eta"] = df["eta"].to_numpy()
        # self.full_data_array["particle_phi"] = df["phi"].to_numpy()
        # self.full_data_array["particle_pdg_id"] = df["pdg_id"].to_numpy()
        # self.full_data_array["particle_has_track"] = df["has_track"].to_numpy()

        # TODO: Compute derived features (sinphi, cosphi, pt from px/py)
        # particle_phi = self.full_data_array["particle_phi"]
        # self.full_data_array["particle_sinphi"] = np.sin(particle_phi)
        # self.full_data_array["particle_cosphi"] = np.cos(particle_phi)
        # self.full_data_array["particle_pt"] = np.sqrt(px**2 + py**2)

        # TODO: Count objects per event
        # self.n_tracks = np.array([len(x) for x in self.full_data_array["track_d0"]])
        # self.n_clusters = np.array([len(x) for x in self.full_data_array["cluster_energy"]])
        # self.n_particles = np.array([len(x) for x in self.full_data_array["particle_energy"]])

        # TODO: Apply event filtering (too many nodes, invalid indices, etc.)
        # mask = ((self.n_tracks + self.n_clusters) < self.max_nodes) & \
        #        (self.n_particles < self.num_objects)

        # TODO: Flatten arrays and compute cumulative sums for indexing
        # self.track_cumsum = np.cumsum([0, *self.n_tracks.tolist()])
        # self.cluster_cumsum = np.cumsum([0, *self.n_clusters.tolist()])
        # self.particle_cumsum = np.cumsum([0, *self.n_particles.tolist()])

        # TODO: Store event numbers
        # self.event_number = np.arange(self.num_events)

        # TODO: Set final number of events
        # self.num_events = len(self.event_number)

        raise NotImplementedError("TODO: Implement load_data() for your parquet schema")

    def __len__(self) -> int:
        return int(self.num_events)

    def load_event(self, idx: int) -> dict[str, Any]:
        """
        Load a single event by index.

        TODO: Implement event loading logic:
        - Extract track, cluster, and particle features for this event
        - Build node features (concatenation of track + cluster features)
        - Build incidence matrix (particle-to-node assignment)
        - Apply feature scaling

        Args:
            idx: Event index

        Returns:
            Dictionary containing:
            - node_inp_features: (max_nodes, n_features) tensor
            - node_raw_features: dict of raw feature tensors
            - particle_data: dict of particle feature tensors
            - incidence_truth: (num_objects, max_nodes) tensor
            - indicator_truth: (num_objects,) tensor indicating valid particles
            - node_q_mask: (max_nodes,) boolean tensor indicating valid nodes
        """
        # TODO: Get counts for this event
        # n_clusters = self.n_clusters[idx]
        # n_tracks = self.n_tracks[idx]
        # n_particles = self.n_particles[idx]
        # n_nodes = n_clusters + n_tracks

        # TODO: Get index ranges
        # cluster_start, cluster_end = self.cluster_cumsum[idx], self.cluster_cumsum[idx + 1]
        # track_start, track_end = self.track_cumsum[idx], self.track_cumsum[idx + 1]
        # particle_start, particle_end = self.particle_cumsum[idx], self.particle_cumsum[idx + 1]

        # TODO: Extract track features
        # track_d0 = self.full_data_array["track_d0"][track_start:track_end]
        # track_z0 = self.full_data_array["track_z0"][track_start:track_end]
        # track_phi = self.full_data_array["track_phi"][track_start:track_end]
        # track_theta = self.full_data_array["track_theta"][track_start:track_end]
        # track_qop = self.full_data_array["track_qop"][track_start:track_end]

        # TODO: Extract cluster features
        # cluster_energy = self.full_data_array["cluster_energy"][cluster_start:cluster_end]
        # cluster_cx = self.full_data_array["cluster_cx"][cluster_start:cluster_end]
        # cluster_cy = self.full_data_array["cluster_cy"][cluster_start:cluster_end]
        # cluster_cz = self.full_data_array["cluster_cz"][cluster_start:cluster_end]

        # TODO: Build node features dictionary (tracks first, then clusters)
        # node_features = {
        #     "d0": torch.cat([scaled_track_d0, torch.zeros(n_clusters)], -1),
        #     "z0": torch.cat([scaled_track_z0, torch.zeros(n_clusters)], -1),
        #     "energy": torch.cat([torch.zeros(n_tracks), scaled_cluster_energy], -1),
        #     "is_track": torch.cat([torch.ones(n_tracks), torch.zeros(n_clusters)], -1),
        #     "is_cluster": torch.cat([torch.zeros(n_tracks), torch.ones(n_clusters)], -1),
        #     # ... more features
        # }

        # TODO: Build incidence matrix
        # incidence_matrix = np.zeros((self.num_objects, n_nodes))
        # ... populate based on majority_particle_id for tracks
        # ... populate based on cluster-particle association

        # TODO: Build indicator (valid particle mask)
        # indicator = torch.zeros(self.num_objects)
        # indicator[:n_particles] = 1.0

        # TODO: Build node validity mask
        # node_q_mask = torch.zeros(self.max_nodes, dtype=bool)
        # node_q_mask[:n_nodes] = True

        raise NotImplementedError("TODO: Implement load_event() for your data schema")

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """
        Get a single sample.

        Returns:
            Tuple of (inputs, labels) dictionaries
        """
        inputs = {}
        labels = {}

        # Load event data
        data_dict = self.load_event(idx)

        # TODO: Build inputs dictionary
        # inputs = {
        #     "node_features": data_dict["node_inp_features"],
        #     "node_valid": data_dict["node_q_mask"],
        #     "node_energy": data_dict["node_raw_features"]["raw_energy"],
        #     "node_eta": data_dict["node_raw_features"]["raw_eta"],
        #     "node_phi": data_dict["node_raw_features"]["raw_phi"],
        #     "node_sinphi": data_dict["node_raw_features"]["sinphi"],
        #     "node_cosphi": data_dict["node_raw_features"]["cosphi"],
        #     "node_is_track": data_dict["node_raw_features"]["is_track"],
        # }

        # TODO: Build class labels
        # class_labels = data_dict["particle_data"]["class"].long()
        # class_labels[data_dict["indicator_truth"] == 0] = self.null_class_idx

        # TODO: Build labels dictionary
        # labels["particle_class"] = class_labels.long()
        # labels["particle_valid"] = data_dict["indicator_truth"].bool()
        # labels["node_valid"] = data_dict["node_q_mask"].bool()
        # labels["particle_node_valid"] = data_dict["incidence_truth"] > self.incidence_cutval
        # labels["particle_incidence"] = data_dict["incidence_truth"]

        # TODO: Add regression targets
        # for label in self.targets["particle"]:
        #     labels[f"particle_{label}"] = data_dict["particle_data"][label]

        # labels["event_number"] = torch.tensor(self.event_number[idx], dtype=torch.int64)

        raise NotImplementedError("TODO: Implement __getitem__() for your data schema")

        return inputs, labels

    def init_label_dicts(self) -> None:
        self.class_labels = {
            -211: 0,
            211: 0,  # pi+-
            -213: 0,
            213: 0,  # rho+-
            -221: 0,
            221: 0,  # eta+-
            -223: 0,
            223: 0,  # omega(782)
            -321: 0,
            321: 0,  # kaon+-
            -323: 0,
            323: 0,  # K*+-
            -331: 3,
            331: 3,  # eta'(958)
            -333: 3,
            333: 3,  # phi(1020)
            311: 3,  # K0
            -311: 3,
            -411: 0,
            411: 0,  # D+-
            -413: 0,
            413: 0,  # D*(2010)+-
            -423: 3,
            423: 3,  # D*(2007)0
            -431: 0,
            431: 0,  # D_s+-
            -433: 0,
            433: 0,  # D_s*+-
            -511: 3,
            511: 3,  # B0
            -521: 0,
            521: 0,  # B+-
            -523: 0,
            523: 0,  # B*+-
            -531: 3,
            531: 3,  # Bs0
            -541: 0,
            541: 0,  # B_c+-
            -1114: 0,
            1114: 0,  # delta+-
            -2114: 0,
            2114: 0,  # delta0
            -2212: 0,
            2212: 0,  # proton
            -3112: 0,
            3112: 0,  # sigma-
            -3312: 0,
            3312: 0,  # xi+-
            -3222: 0,
            3222: 0,  # sigma+
            -3334: 0,
            3334: 0,  # omega
            -4122: 0,
            4122: 0,  # lambda_c+
            -4132: 3,
            4132: 3,  # xi_c0
            -4232: 0,
            4232: 0,  # xi_c+-
            -4312: 0,
            4312: 0,  # xi'_c0
            -4322: 0,
            4322: 0,  # xi'_c+-
            -4324: 0,
            4324: 0,  # xi*c+-
            -4332: 3,
            4332: 3,  # omega_c0
            -4334: 3,
            4334: 3,  # omega*_c0
            -5112: 0,
            5112: 0,  # lambdab-
            -5122: 3,
            5122: 3,  # lambdab0
            -5132: 0,
            5132: 0,  # xib-
            -5232: 3,
            5232: 3,  # xi0_b
            -5332: 0,
            5332: 0,  # omega_b-
            -11: 1,
            11: 1,  # e
            -13: 2,
            13: 2,  # mu
            -15: 0,
            15: 0,  # tau (calling it charged hadron)
            -111: 3,
            111: 3,  # pi0
            113: 3,  # rho0
            130: 3,  # K0L
            310: 3,  # K0S
            -313: 3,
            313: 3,  # K*0
            -421: 3,
            421: 3,  # D0
            -2112: 3,
            2112: 3,  # neutrons
            -3122: 3,
            3122: 3,  # lambda
            -3322: 3,
            3322: 3,  # xi0
            22: 4,  # photon
            1000010020: 0,  # deuteron
            1000010030: 0,  # triton
            1000010040: 0,  # alpha
            1000020030: 0,  # He3
            1000020040: 0,  # He4
            1000030040: 0,  # Li6
            1000030050: 0,  # Li7
            1000020060: 0,  # C6
            1000020070: 0,  # C7
            1000020080: 0,  # O8
            1000010048: 0,  # ?
            1000020032: 0,  # ?
            -999: 5,  # residual
            -12: -1,
            12: -1,  # nu_e
            -14: -1,
            14: -1,  # nu_mu
            -16: -1,
            16: -1,  # nu_tau
        }

    def init_variables_list(self) -> None:
        """
        Initialize lists of variable names.

        TODO: Adjust based on your parquet column names.
        """
        self.track_variables = [
            "d0",
            "z0",
            "phi",
            "theta",
            "qop",
            # TODO: Add derived variables like eta, pt, sinphi, cosphi
        ]

        self.cluster_variables = [
            "total_cluster_energy",
            "cluster_cx",
            "cluster_cy",
            "cluster_cz",
            # TODO: Add derived variables like eta, phi, rho
        ]

        self.particle_variables = [
            "energy",
            "eta",
            "phi",
            "px",
            "py",
            "pz",
            "charge",
            "mass",
            "pdg_id",
            "has_track",
        ]

        # Auxiliary variables for track-particle and cluster-particle associations
        self.aux_vars = [
            "majority_particle_id",   # track -> particle
            # TODO: Add cluster-particle association variable if available
        ]


class ODDDataModule(L.LightningDataModule):
    """
    Lightning DataModule for ODD Particle Flow.

    Handles train/val/test dataset creation and dataloaders.
    """

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
        """
        Initialize DataModule.

        Args:
            train_path: Path to training parquet file(s)
            valid_path: Path to validation parquet file(s)
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            num_train: Number of training events to use
            num_val: Number of validation events to use
            num_test: Number of test events to use
            scale_dict_path: Path to feature scaling YAML file
            test_path: Path to test parquet file(s)
            pin_memory: Whether to pin memory for faster GPU transfer
            test_suff: Suffix for test output files
            **kwargs: Additional arguments passed to ODDDataset
        """
        super().__init__()

        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.test_path = test_path
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.test_suff = test_suff
        self.scale_dict_path = scale_dict_path
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        """Set up datasets for the given stage."""
        if self.trainer.is_global_zero:
            print("-" * 100)

        if stage == "fit":
            self.train_dset = ODDDataset(
                filepath=self.train_path,
                num_events=self.num_train,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )
            self.val_dset = ODDDataset(
                filepath=self.valid_path,
                num_events=self.num_val,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )

        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_path is not None, "No test file specified, see --data.test_path"
            self.test_dset = ODDDataset(
                filepath=self.test_path,
                num_events=self.num_test,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: ODDDataset, shuffle: bool) -> DataLoader:
        """Create a dataloader for the given dataset."""
        print(f"Creating {stage} dataloader with {len(dataset):,} events")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        print("Instantiating train dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        print("Instantiating validation dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        print("Instantiating test dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
