"""
ODD (Open Data Detector) Particle Flow Dataset and DataModule.

Data format: Parquet files loaded via Polars, transformed to NumPy/PyTorch tensors.

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


def morton_encode(eta: torch.Tensor, phi: torch.Tensor, n_bits: int = 16, shift_phi: bool = False) -> torch.Tensor:
    """Map 2D (eta, phi) to 1D Z-order (Morton) index for locality-preserving sort.

    Tokens nearby in the eta-phi plane (small delta-R) will be close in the
    1D Morton index with high probability, enabling windowed attention as an
    approximation of delta-R local attention.

    Parameters
    ----------
    shift_phi : bool
        When True, rotate phi by π before quantisation (add 0.5 in normalised [0,1]
        space via fmod). This moves the periodic "tear" from ±π to 0, so that tokens
        near phi=±π — which are genuinely close in ΔR — end up adjacent in the sorted
        order. Used for Swin-style shifted-window attention.
    """
    eta_norm = torch.clamp((eta + 6.0) / 12.0, 0.0, 1.0)
    phi_norm = torch.clamp((phi + torch.pi) / (2 * torch.pi), 0.0, 1.0)
    if shift_phi:
        phi_norm = torch.fmod(phi_norm + 0.5, 1.0)
    scale = (1 << n_bits) - 1
    ei = (eta_norm * scale).long()
    pi = (phi_norm * scale).long()
    result = torch.zeros(len(ei), dtype=torch.long)
    for bit in range(n_bits):
        result |= ((ei >> bit & 1) << (2 * bit + 1))
        result |= ((pi >> bit & 1) << (2 * bit))
    return result.double()  # float64: 52 mantissa bits → exact up to n_bits=26


class ODDDatasetPileup(Dataset):
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
        num_objects: int = 400,
        max_nodes: int = 7000,
        remove_wrong_idxs: bool = True,
        incidence_cutval: float = 1e-4,
        is_inference: bool = False,
        files_list: list[Path] | None = None,
        hard_scatter_energy_threshold: float = 0.03,
        window_size: int = 512,
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
            files_list: Optional list of files to load (supercedes globbing in filepath)
            window_size: Window size for windowed attention (used in deltaR stats)
        """
        super().__init__()

        self.scaler = FeatureScaler(scale_dict_path)
        # Stage (e.g., 'train', 'val', 'test') - used to toggle stage-specific logic
        # Initialize class labels mapping (PDG ID -> class index)
        self.init_label_dicts()

        # Initialize variable lists
        self.init_variables_list()

        # Input file handling
        self.filedir = filepath
        self.files_list = files_list
        # Store configuration
        self.inputs = inputs
        self.targets = targets
        self.num_objects = num_objects
        self.max_nodes = max_nodes
        self.remove_wrong_idxs = remove_wrong_idxs
        self.incidence_cutval = incidence_cutval
        self.is_inference = is_inference
        self.window_size = window_size
        self.hard_scatter_energy_threshold = hard_scatter_energy_threshold

        print(f"Loading ODD dataset from {filepath} with {num_events} samples")
        print(f"Is inference: {self.is_inference}")

        # Load data from parquet
        self.load_data(filepath,num_events)
        gc.collect()

    def load_data(self, file_dir: str, num_events: int) -> None:
        """
        Load data from parquet file(s) using Polars.
        Memory efficient version: loads sequentially and accumulates tensors.
        """
        self.full_data_array = {}
        
        # 1. Identify Files
        # ---------------------------------------------------------------------
        file_dir_path = Path(file_dir)
        

        # Check for sharded mode - look for target_particles-*.parquet
        # Using glob to find all matching files
        if self.files_list is not None:
             files = self.files_list
        else:
             files = sorted(list(file_dir_path.glob("target_particles-*.parquet")))
        
        if not files:
            # If no sharded files and no single file, try the original heuristic logic
            # or raise error. Original code assumed {file_dir}/target_particles-{i:05d}.parquet
            # We will assume that if the user didn't provide files, maybe they are generated on the fly?
            # But for now let's assume valid files exist or we fail gracefully / use heuristic.
            print(f"Warning: No parquet files found in {file_dir} via glob. Using heuristic indices.")
            num_of_files = num_events // 1000 + 1 if num_events != -1 else 50
            files_indices = list(range(num_of_files))
        else:
                # Extract indices from filenames: target_particles-{i:05d}.parquet
            try:
                files_indices = [int(f.name.split('-')[-1].split('.')[0]) for f in files]
            except (ValueError, IndexError):
                print("Warning: Could not parse indices from filenames. Using sequential index.")
                files_indices = list(range(len(files)))
            
        # 2. Sequential Loading and Accumulation
        # ---------------------------------------------------------------------
        
        accumulated_data = {} # Key -> List of Tensors
        accumulated_counts = {
            "n_tracks": [],
            "n_clusters": [],
            "n_particles": [],
            "n_deps": [],
            "n_raw_deps": [],
            "event_id": [],
            "hard_scatter_vz": [],
        }
        
        total_events_loaded = 0
        stats_total_events = 0
        stats_dropped_nodes = 0
        stats_dropped_particles = 0
        stats_max_particles_seen = 0
        stats_dropped_particles = 0 # New
        
        # List of variables to extract (keeping consistent with original code)
        track_vars = ["d0", "z0", "pt","phi", "theta",'eta', "phi_int", "eta_int",
                      "track_tanlambda", "track_omega", "particle_idx", "majority_particle_vertex_primary"]
        cluster_vars = ["total_cluster_energy", "cluster_rho",
                        "cluster_eta", "cluster_phi",
                        "hcal_fraction", "sigma_eta", "sigma_phi", "sigma_rho",
                        "cluster_time", "number_of_hits", "energy_hits_std", "max_hit_energy"]
        deps_vars = ["hard_scatter_energy_deps_in_cluster", "cluster_idx",
                     "hs_neutral_energy_in_cluster", "hs_charged_energy_in_cluster"]
        # Particle variables for reconstruction targets
        particle_vars = ["energy", "eta", "phi", "pdg_id", "particle_id", "has_track", "pt"]
        # Raw deps for incidence matrix (per-particle-per-cluster energy deposits)
        raw_deps_vars = ["total_energy_deps_in_cluster", "particle_idx", "cluster_idx"]

        if num_events != -1:
            pbar = tqdm(total=num_events, desc="Loading Events", unit="evt")
        else:
            pbar = tqdm(total=len(files_indices), desc="Loading Files", unit="file")
        
        for i, idx in enumerate(files_indices):
            if num_events != -1 and total_events_loaded >= num_events:
                break
                
            print(f"Processing file index {idx} (File {i+1}/{len(files_indices)})")
            # If we're not in test stage, allow a user hook to run here.
            
            df_particles, df_clusters, df_deps, df_tracks, df_raw_deps = self.preprocess_hook(file_dir_path, idx)

            # B. Filter (Locally)
            n_tracks = df_tracks.select(pl.col("track_id").list.len()).to_series().to_numpy()
            n_clusters = df_clusters.select(pl.col("cluster_id").list.len()).to_series().to_numpy()
            n_particles = df_particles.select(pl.col("particle_id").list.len()).to_series().to_numpy()
            n_deps = df_deps.select(pl.col("cluster_idx").list.len()).to_series().to_numpy()
            n_raw_deps = df_raw_deps.select(pl.col("particle_idx").list.len()).to_series().to_numpy()

            n_nodes = n_tracks + n_clusters
            
            # Update Stats
            stats_total_events += len(n_nodes)
            mask_nodes_ok = n_nodes < self.max_nodes
            max_particles = self.num_objects - 1  # position 0 reserved for pileup token
            mask_particles_ok = n_particles <= max_particles
            stats_dropped_nodes += (~mask_nodes_ok).sum()
            stats_dropped_particles += (~mask_particles_ok & mask_nodes_ok).sum()
            if (~mask_particles_ok & mask_nodes_ok).any():
                over = n_particles[(~mask_particles_ok) & mask_nodes_ok]
                stats_max_particles_seen = max(stats_max_particles_seen, int(over.max()))

            mask = mask_nodes_ok & mask_particles_ok
            
            # Check if we need to trim the batch to meet exact num_events
            valid_count = mask.sum()
            if num_events != -1 and total_events_loaded + valid_count > num_events:
                needed = num_events - total_events_loaded
                # Find indices of the first 'needed' True values
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > needed:
                    # Set mask to False for excess events
                    mask[valid_indices[needed:]] = False
                valid_count = mask.sum()

            if valid_count == 0:
                del df_particles, df_clusters, df_deps, df_tracks, df_raw_deps, n_tracks, n_clusters, n_nodes, mask, n_particles, n_deps, n_raw_deps
                gc.collect()
                if num_events == -1:
                    pbar.update(1)
                continue

            total_events_loaded += valid_count
            if num_events != -1:
                pbar.update(valid_count)
            else:
                pbar.update(1)
            
            # Apply Filter
            df_tracks = df_tracks.filter(mask)
            df_clusters = df_clusters.filter(mask)
            df_particles = df_particles.filter(mask)
            df_deps = df_deps.filter(mask)
            df_raw_deps = df_raw_deps.filter(mask)
            
            # Counts
            accumulated_counts["n_tracks"].append(n_tracks[mask])
            accumulated_counts["n_clusters"].append(n_clusters[mask])
            accumulated_counts["n_particles"].append(n_particles[mask])
            accumulated_counts["n_deps"].append(n_deps[mask])
            accumulated_counts["n_raw_deps"].append(n_raw_deps[mask])
            accumulated_counts["event_id"].append(df_tracks["event_id"].to_numpy())

            # C. Extract Features to Tensors
            def safe_append(name, value):
                if name not in accumulated_data: accumulated_data[name] = []
                
                # Check for floating point and cast immediately to save RAM
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    if value.dtype == torch.float64:
                        value = value.float() # Downcast to float32
                accumulated_data[name].append(value)

            # Tracks
            for var in track_vars:
                # explode list column into flat array
                flat_arr = df_tracks.select(pl.col(var).explode()).to_series().to_torch()
                var2 = var.replace("track_","")
                safe_append(f"track_{var2}", flat_arr)
            
            # Clusters
            for var in cluster_vars:
                flat_arr = df_clusters.select(pl.col(var).explode()).to_series().to_torch()
                safe_append(var, flat_arr)

            for var in deps_vars:
                flat_arr = df_deps.select(pl.col(var).explode()).to_series().to_torch()
                safe_append(f"deps_{var}", flat_arr)

            # Particles (for reconstruction targets)
            for var in particle_vars:
                flat_arr = df_particles.select(pl.col(var).explode()).to_series().to_torch()
                safe_append(f"particle_{var}", flat_arr)

            # Raw deps (for incidence matrix)
            for var in raw_deps_vars:
                flat_arr = df_raw_deps.select(pl.col(var).explode()).to_series().to_torch()
                safe_append(f"raw_deps_{var}", flat_arr)

            # Hard scatter vertex z (one per event): pre-computed in preprocess_hook
            if "hard_scatter_vz" in df_tracks.columns:
                accumulated_counts["hard_scatter_vz"].append(
                    df_tracks.select(pl.col("hard_scatter_vz").list.first()).to_series().to_numpy().astype(np.float32)
                )
            else:
                # Fallback: no truth vertex info available (e.g. pu0 data or inference)
                n_events_in_batch = int(mask.sum())
                raise ValueError("No hard_scatter_vz column found in tracks data. This is required for the vertex token feature. Please ensure your data contains this column or implement an alternative method to compute it.")
                accumulated_counts["hard_scatter_vz"].append(np.zeros(n_events_in_batch, dtype=np.float32))

            # Cleanup
            del df_particles, df_clusters, df_deps, df_tracks, df_raw_deps, n_tracks, n_clusters, n_nodes, mask, n_particles, n_deps, n_raw_deps
            # Explicit garbage collection to free Polars memory
            gc.collect()
            
        pbar.close()

        # 3. Concatenate and Finalize
        # ---------------------------------------------------------------------
        print(f"--- Filtering Statistics ---")
        print(f"Total events processed: {stats_total_events}")
        print(f"Events dropped (Too many nodes > {self.max_nodes}): {stats_dropped_nodes}")
        print(f"Events dropped (Too many particles > {self.num_objects - 1}): {stats_dropped_particles}"
              + (f" (max seen: {stats_max_particles_seen})" if stats_max_particles_seen > 0 else ""))
        print(f"Total events kept: {total_events_loaded}")
        print(f"----------------------------")

        print(f"Loaded {total_events_loaded} events. Concatenating arrays...")
        if total_events_loaded == 0:
             print("Warning: No events loaded!")
             return

        self.num_events = total_events_loaded
        
        # Concatenate counts
        self.n_tracks = np.concatenate(accumulated_counts["n_tracks"])
        self.n_clusters = np.concatenate(accumulated_counts["n_clusters"])
        self.n_particles = np.concatenate(accumulated_counts["n_particles"])
        self.n_deps = np.concatenate(accumulated_counts["n_deps"])
        self.n_raw_deps = np.concatenate(accumulated_counts["n_raw_deps"])
        self.event_number = np.concatenate(accumulated_counts["event_id"])
        self.hard_scatter_vz = np.concatenate(accumulated_counts["hard_scatter_vz"]) # Original used pl.Series/DataFrame column behaviour
        print("event numbers 10:", self.event_number[:10]) # Show first 10 event numbers for debugging
        # Concatenate features
        for key, tensor_list in accumulated_data.items():
            self.full_data_array[key] = torch.cat(tensor_list)

        # 4. Computed / Derived Features
        # ---------------------------------------------------------------------
        self.full_data_array["track_sinphi"] = np.sin(self.full_data_array["track_phi"])
        self.full_data_array["track_cosphi"] = np.cos(self.full_data_array["track_phi"])
        
        # --- Derived Track Features ---
        phi_int = self.full_data_array["track_phi_int"]
        self.full_data_array["track_sinphi_int"] = np.sin(phi_int)
        self.full_data_array["track_cosphi_int"] = np.cos(phi_int)
        
        self.full_data_array["cluster_e"] = self.full_data_array["total_cluster_energy"]
        self.full_data_array["cluster_sinphi"] = np.sin(self.full_data_array["cluster_phi"])
        self.full_data_array["cluster_cosphi"] = np.cos(self.full_data_array["cluster_phi"])

        # --- Particle derived features (for reconstruction) ---
        self.full_data_array["particle_sinphi"] = torch.sin(self.full_data_array["particle_phi"].float())
        self.full_data_array["particle_cosphi"] = torch.cos(self.full_data_array["particle_phi"].float())

        # PDG ID -> class mapping
        pdg_ids = self.full_data_array["particle_pdg_id"]
        pdg_ids_np = pdg_ids.numpy() if isinstance(pdg_ids, torch.Tensor) else pdg_ids
        self.full_data_array["particle_class"] = torch.tensor(
            [self.class_labels.get(int(x), 5) for x in pdg_ids_np], dtype=torch.long
        )

        # Cast raw deps indices to int64
        self.full_data_array["raw_deps_particle_idx"] = self.full_data_array["raw_deps_particle_idx"].to(torch.int64)
        self.full_data_array["raw_deps_cluster_idx"] = self.full_data_array["raw_deps_cluster_idx"].to(torch.int64)

        # transform variables and transform to tensors
        for key, val in self.full_data_array.items():
            # 1. Ensure it is a tensor
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            
            # 2. CRITICAL FIX: Cast Float64 (Double) to Float32
            # Polars loads Parquet as Float64 by default, which crashes AMP/Inductor
            if val.is_floating_point():
                val = val.float()
            self.full_data_array[key] = val
        # 6. Build CumSums for Indexing
        # ---------------------------------------------------------------------
        self.track_cumsum = np.cumsum([0, *self.n_tracks.tolist()])
        self.cluster_cumsum = np.cumsum([0, *self.n_clusters.tolist()])
        self.particle_cumsum = np.cumsum([0, *self.n_particles.tolist()])
        self.deps_cumsum = np.cumsum([0, *self.n_deps.tolist()])
        self.raw_deps_cumsum = np.cumsum([0, *self.n_raw_deps.tolist()])
        
        self.n_nodes = self.n_tracks + self.n_clusters
        print(f"Number of events after filtering: {self.num_events}")

        # Print pileup vs hard scatter track statistics
        vertex_primary = self.full_data_array["track_majority_particle_vertex_primary"]
        n_hard_scatter = (vertex_primary == 1).sum().item()
        n_pileup = (vertex_primary != 1).sum().item()
        n_total_tracks = len(vertex_primary)
        pct_hard_scatter = 100 * n_hard_scatter / n_total_tracks if n_total_tracks > 0 else 0
        pct_pileup = 100 * n_pileup / n_total_tracks if n_total_tracks > 0 else 0
        print(f"Track composition: Hard scatter {n_hard_scatter} ({pct_hard_scatter:.1f}%) | Pileup {n_pileup} ({pct_pileup:.1f}%)")

        # Print deposited energy vs total cluster energy statistics
        if "deps_hard_scatter_energy_deps_in_cluster" in self.full_data_array and "total_cluster_energy" in self.full_data_array:
            total_deps_energy = self.full_data_array["deps_hard_scatter_energy_deps_in_cluster"].sum().item()
            total_cluster_energy = self.full_data_array["total_cluster_energy"].sum().item()
            deps_to_cluster_ratio = total_deps_energy / total_cluster_energy if total_cluster_energy > 0 else 0
            print(f"Energy composition: Total deposited HS energy {total_deps_energy:.2e} | Total cluster energy {total_cluster_energy:.2e} | Ratio {deps_to_cluster_ratio:.4f}")

        self.print_deltaR_stats(window_size=self.window_size)

    def print_deltaR_stats(self, n_sample: int = 100, window_size: int = 512) -> None:
        """Sample events and print delta R statistics within a Z-order sorted window.

        Nodes are sorted by their Morton (Z-order) index — the same ordering used by
        the windowed transformer attention — then, for each node, delta R is computed
        to every other node inside the sliding window of `window_size` positions.
        This lets you directly assess how well the window_size config covers local
        structure in eta-phi space.
        """
        n_sample = min(n_sample, self.num_events)
        sample_indices = np.random.choice(self.num_events, size=n_sample, replace=False)
        half_w = window_size // 2
        all_mean_window_dR = []
        all_max_window_dR = []
        all_n_nodes = []

        for evt_idx in sample_indices:
            t_start, t_end = self.track_cumsum[evt_idx], self.track_cumsum[evt_idx + 1]
            c_start, c_end = self.cluster_cumsum[evt_idx], self.cluster_cumsum[evt_idx + 1]
            eta = torch.cat([self.full_data_array["track_eta"][t_start:t_end],
                             self.full_data_array["cluster_eta"][c_start:c_end]])
            phi = torch.cat([self.full_data_array["track_phi"][t_start:t_end],
                             self.full_data_array["cluster_phi"][c_start:c_end]])
            n = len(eta)
            if n < 2:
                continue
            all_n_nodes.append(n)

            # Sort by Morton index — same as the transformer encoder
            sort_idx = torch.argsort(morton_encode(eta, phi))
            eta = eta[sort_idx]
            phi = phi[sort_idx]

            # For each node compute delta R to all nodes within the window
            window_dR = []
            for i in range(n):
                j_start = max(0, i - half_w)
                j_end = min(n, i + half_w + 1)
                nbrs = torch.arange(j_start, j_end)
                nbrs = nbrs[nbrs != i]
                deta = eta[i] - eta[nbrs]
                dphi = phi[i] - phi[nbrs]
                dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                dR = torch.sqrt(deta ** 2 + dphi ** 2)
                window_dR.append(dR)

            window_dR = torch.cat(window_dR)
            all_mean_window_dR.append(window_dR.mean().item())
            all_max_window_dR.append(window_dR.max().item())

        if all_mean_window_dR:
            print(f"--- Delta R window analysis ({n_sample} sampled events, window_size={window_size}) ---")
            print(f"Avg nodes per event: {np.mean(all_n_nodes):.0f}")
            print(f"Mean delta R within window: {np.mean(all_mean_window_dR):.4f}  "
                  f"[Q1={np.percentile(all_mean_window_dR, 25):.4f}, "
                  f"Q3={np.percentile(all_mean_window_dR, 75):.4f}]")
            print(f"Mean max delta R within window: {np.mean(all_max_window_dR):.4f}  "
                  f"[Q1={np.percentile(all_max_window_dR, 25):.4f}, "
                  f"Q3={np.percentile(all_max_window_dR, 75):.4f}]")
            print(f"-------------------------------------------------------------------------")

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
        # 1. Calculate Slices
        # ---------------------------------------------------------------------
        n_tracks = self.n_tracks[idx]
        n_clusters = self.n_clusters[idx]
        n_nodes = n_tracks + n_clusters
        
        n_particles = self.n_particles[idx]

        t_start, t_end = self.track_cumsum[idx], self.track_cumsum[idx+1]
        c_start, c_end = self.cluster_cumsum[idx], self.cluster_cumsum[idx+1]
        d_start, d_end = self.deps_cumsum[idx], self.deps_cumsum[idx+1]
        p_start, p_end = self.particle_cumsum[idx], self.particle_cumsum[idx+1]
        rd_start, rd_end = self.raw_deps_cumsum[idx], self.raw_deps_cumsum[idx+1]

        # 2. Extract & Pad Input Features
        # ---------------------------------------------------------------------
        # Helper to get tensor slice
        def get_t(name, start, end):
            return self.full_data_array[name][start:end]

        # --- Tracks ---
        t_d0 = get_t("track_d0", t_start, t_end)
        t_z0 = get_t("track_z0", t_start, t_end)
        t_phi = get_t("track_phi", t_start, t_end)
        t_pt = get_t("track_pt", t_start, t_end)
        #t_qop = get_t("track_qop", t_start, t_end)
        t_eta = get_t("track_eta", t_start, t_end)
        t_sinphi = get_t("track_sinphi", t_start, t_end)
        t_cosphi = get_t("track_cosphi", t_start, t_end)
        t_eta_int = get_t("track_eta_int", t_start, t_end)
        t_phi_int = get_t("track_phi_int", t_start, t_end)
        t_cosphi_int = get_t("track_cosphi_int", t_start, t_end)
        t_sinphi_int = get_t("track_sinphi_int", t_start, t_end)
        t_tanlambda = get_t("track_tanlambda", t_start, t_end)
        t_omega = get_t("track_omega", t_start, t_end)
        t_vertex_primary = get_t("track_majority_particle_vertex_primary", t_start, t_end)
        t_vertex_primary_mask = (t_vertex_primary == 1).float() # New mask for primary vertex tracks
        t_particle_idx = get_t("track_particle_idx", t_start, t_end).long()

        # --- Clusters ---
        c_e = get_t("cluster_e", c_start, c_end)
        c_eta = get_t("cluster_eta", c_start, c_end)
        c_phi = get_t("cluster_phi", c_start, c_end)
        c_sinphi = get_t("cluster_sinphi", c_start, c_end)
        c_cosphi = get_t("cluster_cosphi", c_start, c_end)
        c_rho = get_t("cluster_rho", c_start, c_end)
        c_sigma_eta = get_t("sigma_eta", c_start, c_end)
        c_sigma_phi = get_t("sigma_phi", c_start, c_end)
        c_sigma_rho = get_t("sigma_rho", c_start, c_end)
        c_hcal_fraction = get_t("hcal_fraction", c_start, c_end)
        c_cluster_time = get_t("cluster_time", c_start, c_end)
        c_number_of_hits = get_t("number_of_hits", c_start, c_end)
        c_energy_hits_std = get_t("energy_hits_std", c_start, c_end)
        c_max_hit_energy = get_t("max_hit_energy", c_start, c_end)

        # --- Energy Deposits ---
        d_cluster_idx = get_t("deps_cluster_idx", d_start, d_end).long()
        d_energy_hard_scatter = get_t("deps_hard_scatter_energy_deps_in_cluster", d_start, d_end)
        
        d_energy_hard_scatter_frac = torch.zeros_like(c_e)
        d_energy_hard_scatter_frac[d_cluster_idx] = d_energy_hard_scatter /(c_e[d_cluster_idx] + 1e-6)
        d_energy_hard_scatter_energy = torch.zeros_like(c_e)
        d_energy_hard_scatter_energy[d_cluster_idx] = d_energy_hard_scatter

        # --- Split HS energy by neutral (trackless) vs charged (tracked) particles ---
        d_neutral = get_t("deps_hs_neutral_energy_in_cluster", d_start, d_end)
        d_charged = get_t("deps_hs_charged_energy_in_cluster", d_start, d_end)
        hs_neutral_energy = torch.zeros_like(c_e)
        hs_charged_energy = torch.zeros_like(c_e)
        hs_neutral_energy[d_cluster_idx] = d_neutral
        hs_charged_energy[d_cluster_idx] = d_charged

        node_features = {
            # Common freatures
            "phi": torch.cat([t_phi, c_phi], -1), # Usually not scaled, pos encoded
            "cosphi": torch.cat([t_cosphi, c_cosphi], -1),
            "sinphi": torch.cat([t_sinphi, c_sinphi], -1),
            "eta": torch.cat([self.scaler.transforms["eta"].transform(t_eta), self.scaler.transforms["eta"].transform(c_eta)], -1),
            # interaction features
            "eta_int": torch.cat([self.scaler.transforms["eta"].transform(t_eta_int), torch.zeros(n_clusters, dtype=torch.float32),], -1,),
            "phi_int": torch.cat([self.scaler.transforms["phi"].transform(t_phi_int), torch.zeros(n_clusters, dtype=torch.float32),], -1,),
            "cosphi_int": torch.cat([t_cosphi_int, torch.zeros(n_clusters, dtype=torch.float32)], -1),
            "sinphi_int": torch.cat([t_sinphi_int, torch.zeros(n_clusters, dtype=torch.float32)], -1),
            # track features set to 0 for clusters
            "pt": torch.cat([self.scaler.transforms["pt"].transform(t_pt), torch.zeros(n_clusters, device=t_pt.device)], -1),
            "d0": torch.cat([self.scaler.transforms["d0"].transform(t_d0), torch.zeros(n_clusters, device=t_d0.device)], -1),
            "z0": torch.cat([self.scaler.transforms["z0"].transform(t_z0), torch.zeros(n_clusters, device=t_z0.device)], -1),
            #"qop": torch.cat([self.scaler.transforms["qop"].transform(t_qop), torch.zeros(n_clusters, device=t_qop.device)], -1),
            "tanlambda": torch.cat([self.scaler.transforms["tanlambda"].transform(t_tanlambda),torch.zeros(n_clusters, dtype=torch.float32),], -1,),
            "omega": torch.cat([self.scaler.transforms["omega"].transform(t_omega),torch.zeros(n_clusters, dtype=torch.float32),], -1,),
            #radiusofinnermosthit, ndf, chi2 missing from ODD tracks

            # cluster features set to 0 for tracks
            "e": torch.cat([torch.zeros(n_tracks, device=c_e.device), self.scaler.transforms["e"].transform(c_e)], -1),
            "rho": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["rho"].transform(c_rho)], -1),
            "sigma_eta": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["sigma_eta"].transform(c_sigma_eta)], -1),
            "sigma_phi": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["sigma_phi"].transform(c_sigma_phi)], -1),
            "sigma_rho": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["sigma_rho"].transform(c_sigma_rho)], -1),
            "hcal_fraction": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), c_hcal_fraction], -1),
            "cluster_time": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["cluster_time"].transform(c_cluster_time)], -1),
            "number_of_hits": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["number_of_hits"].transform(c_number_of_hits)], -1),
            "energy_hits_std": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["energy_hits_std"].transform(c_energy_hits_std)], -1),
            "max_hit_energy": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), self.scaler.transforms["max_hit_energy"].transform(c_max_hit_energy)], -1),

            # flags
            "is_track": torch.cat([torch.ones(n_tracks, dtype=torch.float32), torch.zeros(n_clusters, dtype=torch.float32),],-1,),
            "is_cluster": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), torch.ones(n_clusters, dtype=torch.float32),],-1,),
        }

        # Raw features (for loss computation and analysis)
        # All raw features must be aligned with node ordering: [tracks, clusters]
        node_raw_features = {
            "total_e": torch.cat([torch.zeros(n_tracks), c_e], -1),
            "is_track": torch.cat([torch.ones(n_tracks, dtype=torch.float32), torch.zeros(n_clusters, dtype=torch.float32)], -1),
            "calo_raw_hard_scatter_energy": torch.cat([torch.zeros(n_tracks), d_energy_hard_scatter_energy], -1),
            "calo_raw_hard_scatter_energy_frac": torch.cat([torch.zeros(n_tracks), d_energy_hard_scatter_frac], -1),
            "track_vertex_primary_mask": torch.cat([t_vertex_primary_mask, torch.zeros(n_clusters, dtype=torch.float32)], -1),
            # Raw (unscaled) kinematic quantities for plotting
            "node_pt":  torch.cat([t_pt,  torch.zeros(n_clusters, device=t_pt.device)], -1),
            "node_eta": torch.cat([t_eta, c_eta], -1),
            "node_phi": torch.cat([t_phi, c_phi], -1),
            "node_z0":  torch.cat([t_z0,  torch.zeros(n_clusters, device=t_z0.device)], -1),
            "calo_hs_neutral_energy": torch.cat([torch.zeros(n_tracks), hs_neutral_energy], -1),
            "calo_hs_charged_energy": torch.cat([torch.zeros(n_tracks), hs_charged_energy], -1),
        }

        # Compute Z-order (Morton) index from raw eta/phi for locality-preserving sort
        _eta_cat = torch.cat([t_eta, c_eta], -1)
        _phi_cat = torch.cat([t_phi, c_phi], -1)
        node_deltaR_idx = morton_encode(_eta_cat, _phi_cat)
        node_deltaR_idx_shifted = morton_encode(_eta_cat, _phi_cat, shift_phi=True)

        # Pad everything
        for key, val in node_features.items():
            node_features[key] = do_padding(val, self.max_nodes)
        for key, val in node_raw_features.items():
            node_raw_features[key] = do_padding(val, self.max_nodes)
        node_deltaR_idx = do_padding(node_deltaR_idx, self.max_nodes)
        node_deltaR_idx[n_nodes:] = float('inf')  # padding sorts to end so kv_mask stays aligned
        node_deltaR_idx_shifted = do_padding(node_deltaR_idx_shifted, self.max_nodes)
        node_deltaR_idx_shifted[n_nodes:] = float('inf')

        node_q_mask = torch.zeros(self.max_nodes, dtype=bool)
        node_q_mask[:n_nodes] = True
        node_inp_features = torch.stack(list(node_features.values()), dim=-1)

        # Hard scatter vertex token: single scalar (truth_vz_pt2_weighted for vertex_primary==1)
        hs_vz_raw = torch.tensor([self.hard_scatter_vz[idx]], dtype=torch.float32)
        hs_vz_scaled = self.scaler.transforms["truth_vz_pt2_weighted"].transform(hs_vz_raw)
        # Shape (1, 1): [n_vertex_tokens=1, n_features=1]
        vertex_token_features = hs_vz_scaled.unsqueeze(0)

        # =====================================================================
        # 5. Reconstruction targets: particles, incidence matrix
        # =====================================================================
        particle_class = get_t("particle_class", p_start, p_end).clone()

        # Trackless particle reclassification: charged particles w/o a matched
        # track become neutral (same logic as odd/pflow_data.py)
        # For now, we keep the trackless mask the same for inference(test) because bad coding, doesnt change anything in training
        if self.is_inference and False:
            trackless_particle_mask = torch.zeros_like(particle_class, dtype=torch.bool)
        else:
            trackless_particle_mask = torch.ones(n_particles, dtype=torch.bool)
            valid_idx = t_particle_idx[(t_particle_idx >= 0) & (t_particle_idx < n_particles)]
            trackless_particle_mask[valid_idx] = False

            trackless_chhad_and_e = trackless_particle_mask & (particle_class < 2)
            trackless_muon = trackless_particle_mask & (particle_class == 2)
            particle_class[trackless_chhad_and_e] += 3
            particle_class[trackless_muon] = 3

        is_charged = particle_class < 3

        particle_data = {
            "e": get_t("particle_energy", p_start, p_end),
            "pt": get_t("particle_pt", p_start, p_end),
            "eta": get_t("particle_eta", p_start, p_end),
            "sinphi": get_t("particle_sinphi", p_start, p_end),
            "cosphi": get_t("particle_cosphi", p_start, p_end),
            "class": particle_class,
            "is_charged": is_charged,
        }
        particle_data = self.scaler.transform(particle_data)

        # Shift particle data by 1 to make room for pileup token at position 0
        for key, val in particle_data.items():
            padded = torch.zeros(self.num_objects, *val.shape[1:]) if val.dim() > 1 else torch.zeros(self.num_objects)
            padded[1:n_particles + 1] = val[:n_particles]
            particle_data[key] = padded

        # Pileup token at position 0: dummy class 0 (ignored in loss via ignore_index=-100)
        particle_data["class"][0] = 0
        particle_data["is_charged"][0] = 0

        # --- Incidence matrix (num_objects x n_nodes) ---
        # Row 0 = pileup particle (calo only), rows 1..n_particles = real HS particles
        incidence_matrix = np.zeros((self.num_objects, n_nodes))
        indicator = torch.zeros(self.num_objects)

        # Tracks: only valid HS track→particle assignments go to rows 1..n_particles.
        # PU tracks (track_particle_idx == -1) are left unassigned (all-zero column).
        track_idx = np.arange(len(t_particle_idx), dtype=np.int64)
        t_pidx_np = t_particle_idx.numpy() if isinstance(t_particle_idx, torch.Tensor) else np.asarray(t_particle_idx)
        valid_track_assoc = (t_pidx_np >= 0) & (t_pidx_np < n_particles)
        if np.any(valid_track_assoc):
            incidence_matrix[t_pidx_np[valid_track_assoc] + 1, track_idx[valid_track_assoc]] = 1.0

        # Cluster deposits from raw_deps (HS particle → cluster), shifted +1
        d_particle_idx_np = get_t("raw_deps_particle_idx", rd_start, rd_end).numpy()
        d_cluster_idx_np = get_t("raw_deps_cluster_idx", rd_start, rd_end).numpy()
        d_energy_np = get_t("raw_deps_total_energy_deps_in_cluster", rd_start, rd_end).numpy()
        incidence_matrix[d_particle_idx_np + 1, d_cluster_idx_np + n_tracks] = d_energy_np

        # Pileup particle (row 0): PU cluster energy = total - HS (calo only, no tracks)
        pu_cluster_energy = c_e.numpy() - d_energy_hard_scatter_energy.numpy()
        pu_cluster_energy = np.clip(pu_cluster_energy, 0, None)
        incidence_matrix[0, n_tracks:n_tracks + n_clusters] = pu_cluster_energy

        # Column-normalize
        incidence_matrix /= np.clip(incidence_matrix.sum(axis=0, keepdims=True), a_min=1e-6, a_max=None)

        incidence = torch.tensor(incidence_matrix, dtype=torch.float32)
        incidence = torch.nn.functional.pad(incidence, (0, self.max_nodes - n_nodes, 0, 0))

        # Indicator: pileup valid + real particles with class < 5
        indicator[0] = 1.0  # pileup particle is valid
        is_not_res_mask = particle_class < 5
        indicator[1:n_particles + 1][is_not_res_mask] = 1.0

        return {
            "node_inp_features": node_inp_features,
            "node_raw_features": node_raw_features,
            "node_q_mask": node_q_mask,
            "node_eta": node_features["eta"],
            "node_phi": node_features["phi"],
            "node_deltaR_idx": node_deltaR_idx,
            "vertex_token_features": vertex_token_features,
            # Reconstruction data
            "particle_data": particle_data,
            "incidence_truth": incidence,
            "indicator_truth": indicator,
            "n_particles": n_particles,
            "trackless_particle_mask": trackless_particle_mask,
        }

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """
        Get a single sample.

        Returns:
            Tuple of (inputs, labels) dictionaries
        """
        inputs = {}
        labels = {}

        # load event
        data_dict = self.load_event(idx)

        inputs = {
            "node_features": data_dict["node_inp_features"],
            "node_valid": data_dict["node_q_mask"],
            "node_eta": data_dict["node_eta"],
            "node_phi": data_dict["node_phi"],
            "node_deltaR_idx": data_dict["node_deltaR_idx"],
            "node_e": data_dict["node_raw_features"]["total_e"],
            "node_is_track": data_dict["node_raw_features"]["is_track"],
            "vertex_token_features": data_dict["vertex_token_features"],
            "tracks_mask": data_dict["node_raw_features"]["track_vertex_primary_mask"],
            # Raw node features needed by IncidenceBasedRegressionTask
            "node_pt": data_dict["node_raw_features"]["node_pt"],
            "node_sinphi": torch.cat([
                torch.sin(self.full_data_array["track_phi"][self.track_cumsum[idx]:self.track_cumsum[idx+1]]),
                torch.sin(self.full_data_array["cluster_phi"][self.cluster_cumsum[idx]:self.cluster_cumsum[idx+1]]),
            ], -1),
            "node_cosphi": torch.cat([
                torch.cos(self.full_data_array["track_phi"][self.track_cumsum[idx]:self.track_cumsum[idx+1]]),
                torch.cos(self.full_data_array["cluster_phi"][self.cluster_cumsum[idx]:self.cluster_cumsum[idx+1]]),
            ], -1),
        }
        # Pad sinphi/cosphi
        inputs["node_sinphi"] = do_padding(inputs["node_sinphi"], self.max_nodes)
        inputs["node_cosphi"] = do_padding(inputs["node_cosphi"], self.max_nodes)

        labels["tracks_mask"] = data_dict["node_raw_features"]["track_vertex_primary_mask"]
        labels["node_valid"] = data_dict["node_q_mask"].bool()
        labels["is_track"] = data_dict["node_raw_features"]["is_track"]

        labels["node_e"] = data_dict["node_raw_features"]["total_e"]
        labels["calo_hard_scatter_energy"] = data_dict["node_raw_features"]["calo_raw_hard_scatter_energy"]
        labels["calo_hard_scatter_energy_frac"] = data_dict["node_raw_features"]["calo_raw_hard_scatter_energy_frac"]
        labels["node_pt"]  = data_dict["node_raw_features"]["node_pt"]
        labels["node_eta"] = data_dict["node_raw_features"]["node_eta"]
        labels["node_phi"] = data_dict["node_raw_features"]["node_phi"]
        labels["node_z0"]  = data_dict["node_raw_features"]["node_z0"]
        labels["calo_hs_neutral_energy"] = data_dict["node_raw_features"]["calo_hs_neutral_energy"]
        labels["calo_hs_charged_energy"] = data_dict["node_raw_features"]["calo_hs_charged_energy"]

        labels["event_number"] = torch.tensor(self.event_number[idx], dtype=torch.int64)

        # MaskFormer targets (1 target object = 1 hard scatter vertex)
        labels["particle_valid"] = torch.tensor([True])  # shape (1,)

        hs_track = data_dict["node_raw_features"]["track_vertex_primary_mask"].bool()
        is_track_mask = data_dict["node_raw_features"]["is_track"].bool()
        particle_node_valid = hs_track & is_track_mask
        labels["particle_node_valid"] = particle_node_valid.unsqueeze(0).float()  # (1, max_nodes)

        # Vertex z regression target (truth hard-scatter vz)
        labels["particle_vz"] = torch.tensor([self.hard_scatter_vz[idx]], dtype=torch.float32)

        # Per-node is_track accessible as target for per-node tasks
        labels["node_is_track"] = data_dict["node_raw_features"]["is_track"]

        # =====================================================================
        # Reconstruction targets (reco_particle_*)
        # =====================================================================
        n_particles = data_dict["n_particles"]

        # Class labels: 5 = null for padded/unmatched slots
        reco_class = data_dict["particle_data"]["class"].long()
        reco_class[data_dict["indicator_truth"] == 0] = 5
        labels["reco_particle_class"] = reco_class
        labels["reco_particle_valid"] = data_dict["indicator_truth"].bool()

        # Incidence
        incidence_mask = data_dict["incidence_truth"] > self.incidence_cutval
        labels["reco_particle_node_valid"] = incidence_mask
        labels["reco_particle_incidence"] = data_dict["incidence_truth"]

        # Regression targets (position 0 = pileup = NaN, positions 1..n = real particles)
        for label in self.targets["particle"]:
            tgt = torch.full((self.num_objects,), torch.nan)
            tgt[1:n_particles + 1] = data_dict["particle_data"][label][1:n_particles + 1]
            labels[f"reco_particle_{label}"] = tgt

        return inputs, labels

    def preprocess_hook(self, file_dir: Path, index: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Preprocess raw parquet files for one shard.

        Returns (df_particles, df_clusters, df_deps, df_tracks, df_raw_deps).
        df_deps: aggregated HS energy per cluster (for pileup labels).
        df_raw_deps: per-particle-per-cluster deposits (for incidence matrix).
        """
        df_particles = pl.read_parquet(file_dir / f"target_particles-{index:05d}.parquet")
        df_clusters = pl.read_parquet(file_dir / f"calo_clusters-{index:05d}.parquet")
        df_deps_raw_parquet = pl.read_parquet(file_dir / f"target_particles_deps-{index:05d}.parquet")
        df_tracks = pl.read_parquet(file_dir / f"tracks-{index:05d}.parquet")

        # Align column names to match expected schema
        rename_map = {}
        if "vx" in df_tracks.columns:
            rename_map["vx"] = "majority_particle_vx"
        if "vy" in df_tracks.columns:
            rename_map["vy"] = "majority_particle_vy"
        if "vz" in df_tracks.columns:
            rename_map["vz"] = "majority_particle_vz"
        if "vertex_primary" in df_tracks.columns:
            rename_map["vertex_primary"] = "majority_particle_vertex_primary"
        if rename_map:
            df_tracks = df_tracks.rename(rename_map)

        cols_to_explode = [col for col in df_tracks.columns if col != 'event_id']
        tracks_lazy = df_tracks.lazy().explode(cols_to_explode)

        if not self.is_inference:
            # Remove double-matched tracks only in non-inference stages.
            tracks_lazy = (
                tracks_lazy
                .with_columns([
                    pl.col('pt').len().over(['event_id', 'particle_id']).alias('_count'),
                    pl.col('pt').max().over(['event_id', 'particle_id']).alias('_max_pt')
                ])
                .filter(
                    ~(
                        (pl.col('particle_id') != -1) &
                        (pl.col('_count') > 1) &
                        ((pl.col('pt') - pl.col('_max_pt')).abs() > 1e-3)
                    )
                )
            )

        # Always compute hard_scatter_vz; downstream feature construction requires it.
        df_tracks = (
            tracks_lazy
            .with_columns(
                [
                    (
                        (pl.col("majority_particle_vz") * pl.col("pt").pow(2)).sum().over(["event_id", "majority_particle_vertex_primary"]) /
                        pl.col("pt").pow(2).sum().over(["event_id", "majority_particle_vertex_primary"])
                    ).alias("truth_vz_pt2_weighted")
                ]
            )
            .with_columns(
                pl.when(pl.col("majority_particle_vertex_primary") == 1)
                .then(pl.col("truth_vz_pt2_weighted"))
                .otherwise(None)
                .max()
                .over("event_id")
                .alias("hard_scatter_vz")
            )
            .group_by('event_id', maintain_order=True)
            .agg(pl.col(cols_to_explode + ["truth_vz_pt2_weighted", "hard_scatter_vz"]))
            .sort('event_id')
            .collect()
        )

        # Build per-particle has_track lookup for neutral/charged energy split
        _particles_exploded = (
            df_particles.lazy()
            .select("event_id", "particle_idx", "has_track")
            .explode(["particle_idx", "has_track"])
        )

        # --- Aggregated HS deps (for pileup labels) ---
        df_deps = (
            df_deps_raw_parquet.lazy()
            .select("event_id", "total_energy_deps_in_cluster", "cluster_idx", "particle_idx")
            .explode(["total_energy_deps_in_cluster", "cluster_idx", "particle_idx"])
            .join(_particles_exploded, on=["event_id", "particle_idx"], how="left")
            .with_columns(pl.col("has_track").fill_null(False))
            .group_by("event_id", "cluster_idx", maintain_order=True)
            .agg(
                pl.col("total_energy_deps_in_cluster").sum().alias("hard_scatter_energy_deps_in_cluster"),
                pl.col("total_energy_deps_in_cluster").filter(~pl.col("has_track")).sum().alias("hs_neutral_energy_in_cluster"),
                pl.col("total_energy_deps_in_cluster").filter(pl.col("has_track")).sum().alias("hs_charged_energy_in_cluster"),
            )
            .filter(pl.col("hard_scatter_energy_deps_in_cluster") > self.hard_scatter_energy_threshold)
            .group_by("event_id", maintain_order=True)
            .agg('cluster_idx', 'hard_scatter_energy_deps_in_cluster', 'hs_neutral_energy_in_cluster', 'hs_charged_energy_in_cluster')
            .collect()
        )

        # --- Raw deps (for incidence matrix: per-particle-per-cluster deposits) ---
        df_raw_deps = (
            df_deps_raw_parquet.lazy()
            .select("event_id", "total_energy_deps_in_cluster", "cluster_idx", "particle_idx")
            .explode(["total_energy_deps_in_cluster", "cluster_idx", "particle_idx"])
            .group_by("event_id", maintain_order=True)
            .agg("total_energy_deps_in_cluster", "cluster_idx", "particle_idx")
            .sort("event_id")
            .collect()
        )

        return df_particles, df_clusters, df_deps, df_tracks, df_raw_deps


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
            16: -1,  # nu_tau,
            3212:3, #"Σ0",
            -3212:3,# anti Σ0",
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
            #"qop",
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
            "pdg_id",
            "has_track",
        ]

        # Auxiliary variables for track-particle and cluster-particle associations
        self.aux_vars = [
            "particle_id",   # track -> particle
            # TODO: Add cluster-particle association variable if available
        ]

class ODDDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        batch_size: int,
        num_workers: int,
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        scale_dict_path: str = "",
        num_val_workers: int | None = None,
        num_test_workers: int | None = None,
        inputs: dict | None = None,
        targets: dict | None = None,
        num_objects: int = 400,
        max_nodes: int = 900,
        remove_wrong_idxs: bool = True,
        incidence_cutval: float = 1e-4,
        is_inference: bool = False,
        test_path: str | None = None,
        pin_memory: bool = True,
        test_suff: str | None = None,
        unify_path: str | None = None,
        enable_split: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        val_and_test_split_same: bool = False,
        overtrain: bool = False,
        seed: int = 42,
        hard_scatter_energy_threshold: float = 0.03,
        window_size: int = 512,
        **kwargs,
    ):
        super().__init__()

        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.test_path = test_path
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers
        self.num_test_workers = num_test_workers if num_test_workers is not None else num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.test_suff = test_suff
        self.scale_dict_path = scale_dict_path

        self.unify_path = unify_path
        self.enable_split = enable_split
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.val_and_test_split_same = val_and_test_split_same
        self.overtrain = overtrain
        self.seed = seed

        # Dataset kwargs
        self.dataset_kwargs = {
            "inputs": inputs,
            "targets": targets,
            "num_objects": num_objects,
            "max_nodes": max_nodes,
            "remove_wrong_idxs": remove_wrong_idxs,
            "incidence_cutval": incidence_cutval,
            "is_inference": is_inference,
            "hard_scatter_energy_threshold": hard_scatter_energy_threshold,
            "window_size": window_size,
        }
        self.dataset_kwargs.update(kwargs)

    def setup(self, stage: str):
        is_global_zero = True
        if self.trainer is not None:
             is_global_zero = self.trainer.is_global_zero
        
        if is_global_zero:
            print("-" * 100)

        train_files = None
        val_files = None
        test_files = None

        if self.enable_split and self.unify_path:
            import random
            path = Path(self.unify_path)
            # Must use sorted to ensure determinism before shuffle
            all_files = sorted(list(path.glob("target_particles-*.parquet")))
            
            # Deterministic shuffle
            rng = random.Random(self.seed)
            rng.shuffle(all_files)
            
            n_total = len(all_files)
            n_train = int(n_total * self.train_split)
            
            train_files = all_files[:n_train]
            if self.overtrain:
                val_files = all_files[:n_train]
                test_files = all_files[:n_train]
                print("Overtraining enabled: Using SAME files for Train, Validation, and Test")
                if is_global_zero:
                    print("Overtraining enabled: Using SAME files for Train, Validation, and Test")
            elif self.val_and_test_split_same:
                val_files = all_files[n_train:]
                test_files = all_files[n_train:]
                if is_global_zero:
                    print("Using SAME files for Validation and Test (Rest of dataset)")
            else:
                n_val = int(n_total * self.val_split)
                val_files = all_files[n_train:n_train+n_val]
                test_files = all_files[n_train+n_val:]
            
            print(f"Splitting {n_total} files from {self.unify_path}")
            print(f"Train: {(train_files)} files")
            print(f"Val: {(val_files)} files")
            print(f"Test: {(test_files)} files")

        # create training and validation datasets
        if stage == "fit":
            kw = self.dataset_kwargs.copy()
            path = self.train_path
            if self.enable_split and self.unify_path:
                 kw['files_list'] = train_files
                 path = self.unify_path

            self.train_dset = ODDDatasetPileup(
                filepath=path,
                num_events=self.num_train,
                scale_dict_path=self.scale_dict_path,
                **kw,
            )

        if stage == "fit":
            kw = self.dataset_kwargs.copy()
            path = self.valid_path
            if self.enable_split and self.unify_path:
                 kw['files_list'] = val_files
                 path = self.unify_path
            
            self.val_dset = ODDDatasetPileup(
                filepath=path,
                num_events=self.num_val,
                scale_dict_path=self.scale_dict_path,
                **kw,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            kw = self.dataset_kwargs.copy()
            path = self.test_path
            
            if self.enable_split and self.unify_path:
                kw['files_list'] = test_files
                path = self.unify_path
            elif self.test_path is None:
                 assert self.test_path is not None, "No test file specified, see --data.test_path"
            
            self.test_dset = ODDDatasetPileup(
                filepath=path,
                num_events=self.num_test,
                scale_dict_path=self.scale_dict_path,
                **kw,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: ODDDatasetPileup, shuffle: bool, num_workers: int | None = None):
        nw = num_workers if num_workers is not None else self.num_workers
        print(f"Creating {stage} dataloader with {len(dataset):,} events")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            sampler=None,
            num_workers=nw,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=nw > 0,
        )

    def train_dataloader(self):
        rank = 0 if self.trainer is None else self.trainer.local_rank
        print("Instantiating train dataloader on rank", rank)
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        rank = 0 if self.trainer is None else self.trainer.local_rank
        print("Instantiating validation dataloader on rank", rank)
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False, num_workers=self.num_val_workers)

    def test_dataloader(self):
        rank = 0 if self.trainer is None else self.trainer.local_rank
        print("Instantiating test dataloader on rank", rank)
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False, num_workers=self.num_test_workers)
