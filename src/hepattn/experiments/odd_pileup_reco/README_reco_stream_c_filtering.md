# Stream C Reconstruction Filtering: Code Walkthrough

This document explains the reconstruction filtering block in `TwoStreamMaskFormer._forward_reco()` from `model.py`.

Scope: the section that starts at `# --- Top-k calo selection by energy ---` and ends right before running `self.reco_model(reco_inputs)`.

## Why This Block Exists

The reconstruction decoder should not process all input nodes. The full node set can be large and mostly irrelevant for hard-scatter reconstruction.

This block does three things:

1. Keeps only relevant calo nodes (top-k by energy among selected calo candidates).
2. Merges them with selected track nodes.
3. Packs the result into fixed-size tensors for the inner reconstruction MaskFormer.

## Dimension Conventions

Symbolic dimensions:

- `B`: batch size
- `N`: total padded node count per event (tracks + clusters)
- `D`: embedding size
- `K`: selected calo count cap (`max_reco_calo_nodes`)
- `R`: selected reconstruction node cap (`max_reco_nodes`)

Configured values from `configs/base.yaml`:

- `batch_size = 64` (last batch can be smaller)
- `max_nodes = 5500` -> `N = 5500`
- `dim = 128` -> `D = 128`
- `max_reco_calo_nodes = 1200` -> `K <= 1200`
- `max_reco_nodes = 1400` -> `R <= 1400`

Because data loading pads to `max_nodes`, tensors in this path are typically fixed-width in node dimension (`N=5500`).

## Variable Glossary

- `reco_track_mask` (`bool`, shape `(B, N)`): selected track nodes for reco.
- `reco_calo_mask` (`bool`, shape `(B, N)`): candidate calo nodes for reco.
- `node_e` (`float`, shape `(B, N)`): per-node energy.
- `reco_calo_topk` (`bool`, shape `(B, N)`): top-energy calo subset.
- `reco_node_mask` (`bool`, shape `(B, N)`): final selected nodes = tracks OR top-k calo.
- `reco_embed` (`float`, shape `(B, N, D)`): embedding fed to reco filtering (with skip add).
- `reco_node_indices` (`long`, shape `(B, R)`): compacted node indices.
- `filtered_valid` (`bool`, shape `(B, R)`): validity mask in compacted node space.
- `filtered_embed` (`float`, shape `(B, R, D)`): compacted node embeddings.
- `filtered_raw[k]` (`float`, shape `(B, R)`): compacted raw scalar features.
- `reco_inputs` (dict): final input package for inner reco MaskFormer.

## Line-by-Line Behavior With Shapes

### 1) Calo top-k by energy

```python
node_e = x.get("node_e", torch.zeros(batch_size, x["key_embed"].shape[1], device=device))
if node_e.dim() > 2:
    node_e = node_e.squeeze(-1)
calo_energy_masked = torch.where(reco_calo_mask, node_e, torch.zeros_like(node_e))
```

- `node_e`: `(B, N)` (or squeezed from `(B, N, 1)`).
- `calo_energy_masked`: `(B, N)`.
- Non-calo-candidate nodes are forced to `0.0` energy.

```python
max_k = min(self.max_reco_calo_nodes, calo_energy_masked.shape[-1])
_, topk_indices = calo_energy_masked.topk(max_k, dim=-1, sorted=False)
reco_calo_topk = torch.zeros_like(reco_calo_mask)
reco_calo_topk.scatter_(1, topk_indices, True)
reco_calo_topk = reco_calo_topk & reco_calo_mask
```

- `max_k = min(1200, N)` -> usually `1200`.
- `topk_indices`: `(B, max_k)`.
- `reco_calo_topk`: `(B, N)` boolean mask.
- Final `& reco_calo_mask` is a safety clamp so only valid calo candidates remain.

### 2) Merge track and calo selections

```python
reco_node_mask = reco_track_mask | reco_calo_topk
```

- Shape: `(B, N)`.
- Final selected set for reconstruction.

### 3) Apply skip-add embedding

```python
reco_embed = x["key_embed"] + initial_encoder_embed
```

- Both are `(B, N, D)`, output `(B, N, D)`.
- This preserves and reinforces encoder signal before compacting.

### 4) Build compact index list for selected nodes

```python
D = reco_embed.shape[2]
max_rn = self.max_reco_nodes
raw_keys = ["node_e", "node_pt", "node_eta", "node_sinphi", "node_cosphi", "node_is_track"]

sort_idx = torch.argsort(reco_node_mask.long(), dim=-1, descending=True, stable=True)
reco_node_indices = sort_idx[:, :max_rn]
filtered_valid = reco_node_mask.gather(1, reco_node_indices)
```

- `D = 128`, `max_rn = 1400`.
- `sort_idx`: `(B, N)`.
- `reco_node_indices`: `(B, R)` with `R=min(1400, N)` -> usually `1400`.
- `filtered_valid`: `(B, R)`.
- `stable=True` keeps original ordering among ties.

### 5) Gather embeddings and metadata in compact space

```python
idx_3d = reco_node_indices.unsqueeze(-1).expand(-1, -1, D)
filtered_embed = reco_embed.gather(1, idx_3d)
filtered_is_track = is_track.float().gather(1, reco_node_indices)
```

- `idx_3d`: `(B, R, D)`.
- `filtered_embed`: `(B, R, D)`.
- `filtered_is_track`: `(B, R)`.

```python
filtered_raw = {}
for k in raw_keys:
    if k in x:
        src = x[k]
        if src.dim() > 2:
            src = src.squeeze(-1)
        filtered_raw[k] = src.gather(1, reco_node_indices)
    else:
        filtered_raw[k] = torch.zeros(batch_size, max_rn, device=device)
```

- Each `filtered_raw[k]` becomes `(B, R)`.
- Missing keys are zero-filled to maintain shape consistency.

### 6) Save mapping and build `reco_inputs`

```python
self._reco_node_indices = reco_node_indices
self._reco_filtered_valid = filtered_valid

reco_inputs = {
    "node_embed_raw": filtered_embed,
    "node_valid": filtered_valid,
    "node_is_track": filtered_is_track,
}
for k in raw_keys:
    reco_inputs[k] = filtered_raw[k]
```

- Saved members are used later to reindex targets inside `loss()`.
- Final key shapes in `reco_inputs`:
  - `node_embed_raw`: `(B, R, D)`
  - `node_valid`: `(B, R)`
  - each raw key: `(B, R)`

## Design Choices (Why)

- Top-k calo by energy: keeps the most informative neutral activity while bounding compute.
- Track + calo union: tracks are crucial for charged reconstruction; calo adds neutral/leftover context.
- Stable mask sort + gather: turns sparse selection into contiguous tensors efficiently, no Python loops over batch.
- Fixed caps (`1200`, `1400`): keeps compute predictable and compile-friendly.

## Practical Notes and Gotchas

- `reco_inputs["node_is_track"]` is assigned twice: first from `filtered_is_track`, then overwritten by `filtered_raw["node_is_track"]` in the loop. This is usually harmless if values match, but it is good to know.
- If many masked energies are zero, `topk` can include arbitrary zero-energy positions; the final `& reco_calo_mask` prevents invalid entries from surviving.
- The output compact width is fixed (`R=1400`), while `node_valid` indicates which slots are truly selected.

## Quick Shape Snapshot (Typical)

For `B=64`, `N=5500`, `D=128`, `K=1200`, `R=1400`:

- `reco_track_mask`: `(64, 5500)`
- `reco_calo_mask`: `(64, 5500)`
- `topk_indices`: `(64, 1200)`
- `reco_node_mask`: `(64, 5500)`
- `reco_node_indices`: `(64, 1400)`
- `filtered_embed`: `(64, 1400, 128)`
- `reco_inputs["node_valid"]`: `(64, 1400)`

This is the tensor package sent to the inner `MaskFormer` reconstruction model.
