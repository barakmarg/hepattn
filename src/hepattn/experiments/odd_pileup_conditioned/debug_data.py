import sys
import os
from tqdm import tqdm
import torch

# Add src to path so imports work
sys.path.append("/storage/agrp/barakma/hepattn/src")

try:
    from hepattn.experiments.odd_pileup_conditioned.pflow_data import ODDDatasetPileup
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
    from hepattn.experiments.odd_pileup_conditioned.pflow_data import ODDDatasetPileup

def debug_load():
    inputs = {
        "node": ["d0", "z0", "phi", "theta", "qop", "energy", "eta", "is_track"]
    }
    targets = {
        "particle": ["e", "pt", "eta", "sinphi", "cosphi"]
    }

    data_path = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200"
    scale_path = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup/configs/odd_var_transform.yaml"

    print("Initializing Dataset...")
    dataset = ODDDatasetPileup(
        filepath=data_path,
        inputs=inputs,
        targets=targets,
        scale_dict_path=scale_path,
        num_events=200,
        max_nodes=5500,
        remove_wrong_idxs=True,
        incidence_cutval=0.01,
        is_inference=False,
    )

    print(f"Dataset initialized. Number of valid events: {len(dataset)}")

    # Accumulators for cluster type distribution across events
    type_counts = {"pu_only": 0, "mix_pu": 0, "balanced": 0, "mix_hs": 0, "hs_only": 0}

    print("Iterating over all events (calling __getitem__)...")
    for i in tqdm(range(len(dataset))):
        try:
            inp, labels = dataset[i]

            # --- Verify expected input keys ---
            for key in ["node_features", "node_valid", "node_eta", "node_phi", "node_e", "node_is_track"]:
                assert key in inp, f"Missing input key: {key}"

            # --- Verify expected label keys ---
            for key in [
                "tracks_mask", "node_valid", "is_track",
                "node_e",
                "calo_hard_scatter_energy",
                "calo_hard_scatter_energy_frac",
                "event_number",
            ]:
                assert key in labels, f"Missing label key: {key}"

            # --- Shape sanity checks ---
            max_nodes = inp["node_features"].shape[0]
            for key in ["node_valid", "node_e", "node_is_track"]:
                assert inp[key].shape[0] == max_nodes, f"Shape mismatch for input {key}"
            for key in ["node_valid", "is_track", "node_e", "calo_hard_scatter_energy", "calo_hard_scatter_energy_frac"]:
                assert labels[key].shape[0] == max_nodes, f"Shape mismatch for label {key}"

            # --- Cluster type distribution ---
            node_valid = labels["node_valid"].bool()
            is_track = labels["is_track"].bool()
            cluster_mask = node_valid & (~is_track)
            hs_frac = labels["calo_hard_scatter_energy_frac"][cluster_mask]

            type_counts["pu_only"]  += (hs_frac < 0.02).sum().item()
            type_counts["mix_pu"]   += ((hs_frac >= 0.03) & (hs_frac <= 0.30)).sum().item()
            type_counts["balanced"] += ((hs_frac > 0.30)  & (hs_frac < 0.70)).sum().item()
            type_counts["mix_hs"]   += ((hs_frac >= 0.70) & (hs_frac <= 0.97)).sum().item()
            type_counts["hs_only"]  += (hs_frac > 0.98).sum().item()

            if i == 0:
                print(f"\n--- First event ---")
                print(f"  node_features shape:  {inp['node_features'].shape}")
                print(f"  node_valid (n valid): {node_valid.sum().item()}")
                print(f"  n tracks:             {(node_valid & is_track).sum().item()}")
                print(f"  n clusters:           {cluster_mask.sum().item()}")
                print(f"  node_e (cluster) range: [{labels['node_e'][cluster_mask].min():.3f}, {labels['node_e'][cluster_mask].max():.3f}]")
                print(f"  calo_hs_energy_frac range: [{hs_frac.min():.3f}, {hs_frac.max():.3f}]")

        except Exception as e:
            print(f"Failed at index {i}: {e}")
            raise

    print("\nSuccessfully iterated over all events.")
    total = sum(type_counts.values())
    print(f"\n--- Cluster type distribution (across {len(dataset)} events) ---")
    for name, count in type_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {name:<12}: {count:>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    debug_load()
