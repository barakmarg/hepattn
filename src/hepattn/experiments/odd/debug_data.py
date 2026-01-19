import sys
import os
from tqdm import tqdm
import torch

# Add src to path so imports work
sys.path.append("/storage/agrp/barakma/hepattn/src")

try:
    from hepattn.experiments.odd.pflow_data import ODDDataset
except ImportError:
    # If running from different location, adjust path
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
    from hepattn.experiments.odd.pflow_data import ODDDataset

def debug_load():
    inputs = {
        "node": ["d0", "z0", "phi", "theta", "qop", "energy", "eta", "is_track"]
    }
    targets = {
        "particle": ["e", "pt", "eta", "sinphi", "cosphi"]
    }
    
    # Path to your data - using train path from config
    data_path = "/storage/agrp/barakma/PileupODD/data/train"
    # Using absolute path for config
    scale_path = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd/configs/odd_var_transform.yaml"
    
    print("Initializing Dataset...")
    # NOTE: Set num_events filters how many events are LOADED into memory from parquet
    # The filter inside load_data will further reduce this based on max_nodes/num_objects
    dataset = ODDDataset(
        filepath=data_path,
        inputs=inputs,
        targets=targets,
        scale_dict_path=scale_path,
        num_events=-1, # Load all available
        num_objects=350,
        max_nodes=900,
        remove_wrong_idxs=True,
        incidence_cutval=0.01,
        is_inference=False
    )
    
    print(f"Dataset initialized. Number of valid events: {len(dataset)}")
    
    print("Iterating over all events (calling __getitem__)...")
    for i in tqdm(range(len(dataset))):
        try:
            d = dataset[i]
            inputs, labels = d
            
            # Check for NaN values in inputs
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():

                    max_val = torch.abs(val).max().item()
                    if max_val > 1e-1:
                        pass#print(f"Event {i}: Large value in {key}: {max_val}")
                    
                    if torch.isnan(val).any():
                        print(f"Event {i}: NaN values found in {key}")
                    if torch.isinf(val).any():
                        print(f"Event {i}: Inf values found in {key}")
            
            # Check for NaN, Inf, and large values in labels
            for key, val in labels.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    max_val = torch.abs(val).max().item()
                    if max_val > 1e2:
                        pass#print(f"Event {i}: Large value in labels {key}: {max_val}")
                    
                    if torch.isnan(val).any():
                        print(f"Event {i}: NaN values found in labels {key}")
                    if torch.isinf(val).any():
                        print(f"Event {i}: Inf values found in labels {key}")
            
        except Exception as e:
            print(f"Failed at index {i}: {e}")
            # Optionally raise to see full traceback
            # raise e
            
    print("Successfully iterated over all events.")

if __name__ == "__main__":
    debug_load()
