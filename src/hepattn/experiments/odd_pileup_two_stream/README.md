# Two-Stream Causally-Conditioned Hybrid Query MaskFormer

## Motivation

The single-stream MaskFormer (`odd_pileup_maskformer`) uses one query to handle both track HS classification and calo energy fraction prediction. This forces a single query embedding to serve two fundamentally different goals, entangling gradients and limiting calo performance. The two-stream architecture decouples these tasks with strict gradient isolation.

## Architecture Overview

```
Raw Nodes (tracks + clusters)
    -> InputNet -> Encoder (8-layer windowed attention, fixed Morton order)
        -> Stream A: PileupMaskFormerDecoder (4 layers, 1 query, bidirectional CA)
            -> Track Tasks: ObjectHitMaskTask (HS mask) + ObjectRegressionTask (vz)
                |
                | Bridge: extract HS track embeddings, .detach(), pad to fixed size
                | Training: teacher-forced GT HS tracks
                | Inference: predicted HS tracks from Stream A
                v
        -> Stream B: CaloFlashCrossAttentionDecoder (4 layers, 206 hybrid queries, flash-varlen bidirectional CA)
            -> Calo Tasks: CaloHitMaskTask (calo HS mask) + PileupCaloFractionTaskV2 (energy fraction)
```

## Key Design Decisions

### Gradient Isolation
- Node embeddings are **`.detach()`ed** before building hybrid queries for Stream B
- Calo losses cannot backpropagate into the shared encoder or Stream A
- Stream A track F1 is protected from interference by the calo objective

### Teacher Forcing (Bridge)
- **Training:** GT HS track mask (`tracks_mask & node_is_track`) selects which track embeddings become hybrid queries
- **Inference:** Stream A's predicted mask (`sigmoid(logit) >= 0.5 & is_track`) is used instead
- This ensures Stream B sees clean HS track context during training, learns to handle predicted masks at test time

### Hybrid Queries
- Slots 0..15: **Learnable latent queries** (16 × `nn.Parameter`) — capture neutral/shared calo context
- Slots 16..205: **Detached HS track embeddings** — each carries physics context about one HS track, enabling per-track calo association
- Zero-padded to fixed size (`num_latent_queries=16` + `max_hs_tracks=190` = 206 total) with `query_valid` mask so padded slots are ignored in attention

### CaloFlashCrossAttentionDecoder (Stream B)
- Each layer: **query self-attention** → **bidirectional cross-attention** (queries ↔ calo nodes)
- Uses `flash-varlen` (unpadded sequences) for both self-attention and cross-attention — avoids materialising padded positions
- Calo isolation: only calo nodes (`~node_is_track & valid`) participate as keys/values
- No intermediate task outputs — loss computed only at the final layer (`has_intermediate_loss: false`)

### Multi-Query Calo Mask (CaloHitMaskTask)
- Each of 206 queries produces its own mask over all nodes via dot-product: `(B, Q, N)`
- **Per-query masks** drive the attention threshold (each query attends to its claimed calo nodes at `attn_threshold=0.1`)
- **Master mask** = `max(dim=Q)` pools per-query masks to `(B, 1, N)` for BCE + Tversky loss against target
- Target: `calo_hard_scatter_energy_frac > hs_frac_threshold` **AND** `calo_hard_scatter_energy > hs_energy_threshold` (configurable, defaults 0.05 / 0.15)
- Imbalance handled by `sample_weight`: HS nodes get weight 1.0, pileup nodes get `null_weight=0.05` (20× downweighting)

### Calo Fraction Loss Isolation
- `PileupCaloFractionTaskV2` computes L1 loss **only on true HS clusters** (both `calo_hard_scatter_energy_frac > 0.05` and `calo_hard_scatter_energy > 0.15`)
- Pileup-only clusters are excluded from the regression loss entirely
- Prediction: `E_HS_pred = frac * E_total`, loss: `|E_HS_pred - E_HS_true|`

### Gated Inference
- At inference, calo fraction is multiplied by the calo mask prediction: `calo_frac * (calo_hs_prob > 0.5)`
- Clusters predicted as pileup-only get zero HS energy fraction
- Note: `pred_threshold=0.2` controls the `calo_node_valid` binary prediction output; the fraction gate uses a hardcoded 0.5

### Physics Priors
- **Stream A:** `prior = node_is_track` — query attends only to tracks, never calo
- **Stream B:** calo isolation is enforced by `CaloFlashCrossAttentionDecoder` directly (constructs `calo_mask = key_valid & ~node_is_track` for flash-varlen unpadding)

---

## Files

```
odd_pileup_two_stream/
  __init__.py              # Exports
  model.py                 # TwoStreamMaskFormer (forward, loss, predict, bridge logic)
  decoder.py               # CaloMaskFormerDecoder (legacy) + CaloFlashCrossAttentionDecoder (active)
  tasks.py                 # CaloHitMaskTask + PileupCaloFractionTaskV2
  lightning_module.py       # ODDPFlowTwoStream (metrics, epoch-end plots)
  main.py                  # CLI entry point
  configs/
    base.yaml              # Full training config
```

**One modification to existing code:**
- `odd_pileup_maskformer/pflow_data.py`: Added `tracks_mask` to `inputs` dict for teacher forcing

---

## Data Input

### `inputs` dict (same as baseline + `tracks_mask`)

| Key | Shape | Description |
|-----|-------|-------------|
| `node_features` | `(B, 5500, 21)` | Per-node feature vector |
| `node_valid` | `(B, 5500)` | Bool mask for real nodes |
| `node_eta` | `(B, 5500)` | Raw eta |
| `node_phi` | `(B, 5500)` | Raw phi |
| `node_deltaR_idx` | `(B, 5500)` | Morton sort index |
| `node_e` | `(B, 5500)` | Raw total energy |
| `node_is_track` | `(B, 5500)` | 1=track, 0=calo |
| `tracks_mask` | `(B, 5500)` | **NEW** Bool GT HS track mask (for teacher forcing) |

### `targets` dict

| Key | Shape | Description |
|-----|-------|-------------|
| `tracks_mask` | `(B, 5500)` | GT HS track mask |
| `node_valid` | `(B, 5500)` | Valid node mask |
| `node_is_track` | `(B, 5500)` | Track/cluster indicator |
| `node_e` | `(B, 5500)` | Total energy |
| `calo_hard_scatter_energy` | `(B, 5500)` | True HS energy per cluster |
| `calo_hard_scatter_energy_frac` | `(B, 5500)` | True HS energy fraction |
| `particle_valid` | `(B, 1)` | Always True |
| `particle_node_valid` | `(B, 1, 5500)` | HS track mask for ObjectHitMaskTask |
| `particle_vz` | `(B, 1)` | True vertex z |

---

## Forward Pass Detail

### Stage 1: Shared Embedding + Encoding

```
node_features (B, 5500, 21)
    -> Dense(21 -> 128) + FourierPosEnc(eta, phi -> 128)
    -> node_embed (B, 5500, 128)
    -> Encoder (8 layers, flash-varlen, window=256, Morton sort)
    -> key_embed (B, 5500, 128)
```

### Stage 2: Stream A — Track MaskFormer

```
query_embed (B, 1, 128)  <-- learnable nn.Parameter (track_query_initial)
key_embed   (B, 5500, 128) <-- from encoder

4 decoder layers, each:
  1. ObjectHitMaskTask.forward() -> mask_logit (B, 1, 5500)
     learned_mask = sigmoid(logit) >= 0.5
  2. Physics prior: attn_mask = learned_mask & node_is_track
  3. Cross-attention: query attends to HS tracks only
  4. Self-attention + FFN on query
  5. Bidirectional CA: nodes attend back to query
  6. Deep supervision: BCE + Dice on mask at every layer

Final tasks:
  - ObjectHitMaskTask -> pflow_node_logit (B, 1, 5500)
  - ObjectRegressionTask -> pflow_vz (B, 1)
```

### Stage 3: Bridge

```
IF training + teacher_forcing:
    hs_mask = tracks_mask & node_is_track     (GT)
ELSE:
    hs_mask = (sigmoid(logit) >= 0.5) & node_is_track   (predicted)

node_embed_detached = node_embed.detach()     # GRADIENT WALL

hybrid_queries (B, 206, 128):
    slots 0..15:    learnable latent queries (calo_latent_queries)
    slots 16..N+15: detached HS track embeddings
    slots N+16..:   zero-padded

query_valid (B, 206):
    slots 0..15:    True
    slots 16..N+15: True
    slots N+16..:   False (masked in attention)
```

### Stage 4: Stream B — Calo (CaloFlashCrossAttentionDecoder)

```
query_embed (B, 206, 128)  <-- hybrid_queries
key_embed   (B, 5500, 128) <-- from encoder (calo nodes only participate)

4 decoder layers, each:
  1. Query self-attention (flash-varlen over valid queries)
  2. Bidirectional cross-attention (flash-varlen, queries ↔ calo nodes only)
     - a->b: queries attend to calo
     - b->a: calo nodes attend back to queries
  3. No mask attention, no intermediate task outputs

Final tasks (applied once, outside decoder):
  - CaloHitMaskTask -> calo_node_logit (B, 1, 5500)  [max-pooled over query dim]
  - PileupCaloFractionTaskV2 -> calo_frac (B, 5500)
```

---

## Loss Summary

### Stream A (Track) Losses

| Loss key | Task | Computed at | Target |
|---|---|---|---|
| `track_layer_{0-3}_mask_mask_bce` | ObjectHitMaskTask | Each decoder layer | `particle_node_valid` |
| `track_layer_{0-3}_mask_mask_dice` | ObjectHitMaskTask | Each decoder layer | `particle_node_valid` |
| `track_final_mask_mask_bce` | ObjectHitMaskTask | Final | `particle_node_valid` |
| `track_final_mask_mask_dice` | ObjectHitMaskTask | Final | `particle_node_valid` |
| `track_final_vz_regression_smooth_l1` | ObjectRegressionTask | Final only | `particle_vz` |

### Stream B (Calo) Losses

No intermediate losses (`has_intermediate_loss: false` for both calo tasks).

| Loss key | Task | Computed at | Target |
|---|---|---|---|
| `calo_final_calo_mask_mask_bce` | CaloHitMaskTask | Final only | `calo_hs_frac > 0.05 & calo_hs_energy > 0.15` |
| `calo_final_calo_mask_mask_tversky` | CaloHitMaskTask | Final only | `calo_hs_frac > 0.05 & calo_hs_energy > 0.15` |
| `calo_final_calo_fraction_l1` | PileupCaloFractionTaskV2 | Final only | `calo_hard_scatter_energy` (HS clusters only) |

Loss weights: `mask_bce=5.0`, `mask_tversky=2.0`, `calo_fraction l1=0.1`.
Tversky parameters: `alpha=0.2, beta=0.8` (FN penalised 4× more than FP — HS-favouring).

---

## Prediction Outputs

```python
preds = {
    # Stream A — intermediate + final
    "track_layer_0": {"mask": {"pflow_node_valid": (B,1,5500), "pflow_node_prob": (B,1,5500)}},
    "track_layer_1": {"mask": {...}},
    "track_layer_2": {"mask": {...}},
    "track_layer_3": {"mask": {...}},
    "track_final": {
        "mask": {"pflow_node_valid": (B,1,5500), "pflow_node_prob": (B,1,5500)},
        "vz_regression": {"pflow_vz": (B, 1)},
    },

    # Stream B — final only (no intermediate outputs from CaloFlashCrossAttentionDecoder)
    "calo_final": {
        "calo_mask": {"calo_node_prob": (B,5500), "calo_node_valid": (B,5500)},
        "calo_fraction": {"calo_hs_fraction": (B,5500)},   # GATED by calo_hs_prob > 0.5
    },
}
```

---

## Metrics Logged

### Track Metrics (Stream A)
| Metric | Source |
|--------|--------|
| `val/track_f1` | `track_final.mask.pflow_node_prob` vs `tracks_mask` (track nodes only) |
| `val/track_precision` | same |
| `val/track_recall` | same |
| `val/vz_mae` | `track_final.vz_regression.pflow_vz` vs `particle_vz` |

### Calo Metrics (Stream B)
| Metric | Source |
|--------|--------|
| `val/calo_mask_f1` | `calo_final.calo_mask.calo_node_prob` vs `calo_hs_frac > 0.05 & calo_hs_energy > 0.15` |
| `val/calo_mask_precision` | same |
| `val/calo_mask_recall` | same |
| `val/calo_frac_mae` | `calo_final.calo_fraction.calo_hs_fraction` vs `calo_hard_scatter_energy_frac` |
| `val/calo_frac_mae_{pu_only,mix_pu,balanced,mix_hs,hs_only}` | Per-type MAE by true HS fraction |
| `val/calo_hs_energy_ratio` | `sum(pred HS E) / sum(true HS E)` |

### Epoch-End Plots (CometML)
| Plot | Description |
|------|-------------|
| `calo/energy_corr` | Predicted vs true HS energy correlation |
| `calo/frac_corr` | Predicted vs true HS fraction correlation |
| `calo/frac_dist` | Predicted and true HS fraction distributions |
| `calo/energy_resid` | HS energy residual distribution |
| `calo/hs_energy_dist` | Predicted and true HS energy distributions |
| `track/score_dist` | Track HS score distribution (HS vs PU) |
| `track/eff_vs_pt` | Track efficiency vs pT |
| `track/rej_vs_pt` | Pileup rejection vs pT |
| `track/mistag_eta` | Mistag rate vs eta |
| `track/score_by_pt` | Track score distributions by pT bin |
| `track/roc` | ROC curve |
| `track/z0_dist` | Track z0 distribution (HS vs PU) |
| `track/pt_dist` | Track pT distribution (HS vs PU) |

---

## Configurable Parameters

| Parameter | Location | Value | Description |
|-----------|----------|-------|-------------|
| `max_hs_tracks` | `model.init_args` | 190 | Max HS tracks in hybrid queries |
| `num_latent_queries` | `model.init_args` | 16 | Learnable latent query slots (always valid) |
| `teacher_forcing` | `model.init_args` | true | Use GT HS tracks during training bridge |
| `hs_energy_threshold` | `CaloHitMaskTask`, `PileupCaloFractionTaskV2` | 0.15 | Min absolute HS energy to count as HS |
| `hs_frac_threshold` | `CaloHitMaskTask`, `PileupCaloFractionTaskV2` | 0.05 | Min HS energy fraction to count as HS |
| `null_weight` | `CaloHitMaskTask` | 0.05 | BCE sample weight for pileup nodes (1.0 for HS → 20:1 ratio) |
| `pred_threshold` | `CaloHitMaskTask` | 0.2 | Threshold for `calo_node_valid` binary prediction |
| `logit_scale` | `CaloHitMaskTask` | 4 | Dot-product logit scaling factor |
| `signal_weight` | `PileupCaloFractionTaskV2` | 25.3 | Reserved weight parameter for fraction task |
| `batch_size` | `data` | 64 | Training batch size |

---

## Training

```bash
cd src/hepattn/experiments/odd_pileup_two_stream
python main.py fit --config configs/base.yaml
```

---

## Differences from Single-Stream Baseline (`odd_pileup_maskformer`)

| Aspect | Single-Stream | Two-Stream |
|--------|---------------|------------|
| Decoders | 1 (PileupMaskFormerDecoder, 4 layers) | 2 (Track: 4 layers + Calo: 4 layers) |
| Queries | 1 learnable query | 16 latent queries + up to 190 HS track embeddings (206 total) |
| Calo mask | None (implicit via fraction task) | Explicit CaloHitMaskTask (final layer only) |
| Calo mask loss | — | BCE (×5) + Tversky α=0.2,β=0.8 (×2) |
| Calo fraction loss | All clusters (weighted) | HS clusters only (`hs_frac > 0.05 & hs_energy > 0.15`) |
| Gradient flow | Single path (encoder ↔ decoder) | Isolated: calo loss detached from encoder/tracks |
| Inference | Direct fraction output | Gated: `frac * (calo_mask > 0.5)` |
| Calo decoder | — | CaloFlashCrossAttentionDecoder (flash-varlen, bidirectional q↔calo) |
| Parameters | ~10.5M | ~20.7M |
