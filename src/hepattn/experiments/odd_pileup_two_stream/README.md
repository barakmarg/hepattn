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
        -> Stream B: CaloMaskFormerDecoder (4 layers, 191 hybrid queries, unidirectional CA)
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
- Slot 0: **Global calo query** (learnable `nn.Parameter`) -- captures isolated neutral energy not associated with any track
- Slots 1..190: **Detached HS track embeddings** -- each carries physics context about one HS track, enabling per-track calo association
- Zero-padded to fixed size (`max_hs_tracks=190`) with `query_valid` mask so padded slots are ignored in attention

### Multi-Query Calo Mask (CaloHitMaskTask)
- Each of 191 queries produces its own mask over all nodes via dot-product: `(B, Q, N)`
- **Per-query masks** drive masked cross-attention (each query attends to its claimed calo nodes)
- **Master mask** = `max(dim=Q)` pools per-query masks to `(B, 1, N)` for BCE+Dice loss against target
- Target: `calo_hard_scatter_energy > hs_energy_threshold` (configurable, default 0.15)

### Calo Fraction Loss Isolation
- `PileupCaloFractionTaskV2` computes L1 loss **only on true HS clusters** (`calo_hard_scatter_energy > 0.15`)
- Pileup-only clusters are excluded from the regression loss entirely
- Prediction: `E_HS_pred = frac * E_total`, loss: `|E_HS_pred - E_HS_true|`

### Gated Inference
- At inference, calo fraction is multiplied by the calo mask prediction: `calo_frac * (calo_hs_prob > 0.5)`
- Clusters predicted as pileup-only get zero HS energy fraction

### Physics Priors
- **Stream A:** `prior = node_is_track` -- query attends only to tracks, never calo
- **Stream B:** `prior = ~node_is_track` -- queries attend only to calo, never tracks (inverted in `CaloMaskFormerDecoder`)

### Memory Optimization
- Stream B uses `bidirectional_ca: false` -- nodes do not attend back to the 191 queries
- This saves ~4 GiB per decoder layer (avoids the `(B, 16, 5500, 191)` reverse attention matrix)
- Stream A keeps bidirectional CA (only 1 query, negligible cost)

---

## Files

```
odd_pileup_two_stream/
  __init__.py              # Exports
  model.py                 # TwoStreamMaskFormer (forward, loss, predict, bridge logic)
  decoder.py               # CaloMaskFormerDecoder (inverted prior, query_valid masking)
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
    -> Dense(21 -> 256) + FourierPosEnc(eta, phi -> 256)
    -> node_embed (B, 5500, 256)
    -> Encoder (8 layers, flash-varlen, window=256, Morton sort)
    -> key_embed (B, 5500, 256)
```

### Stage 2: Stream A -- Track MaskFormer

```
query_embed (B, 1, 256)  <-- learnable nn.Parameter (track_query_initial)
key_embed   (B, 5500, 256) <-- from encoder

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

hybrid_queries (B, 191, 256):
    slot 0:      calo_global_query (learnable)
    slots 1..N:  detached HS track embeddings
    slots N+1..: zero-padded

query_valid (B, 191):
    slot 0:      True
    slots 1..N:  True
    slots N+1..: False (masked in attention)
```

### Stage 4: Stream B -- Calo MaskFormer

```
query_embed (B, 191, 256)  <-- hybrid_queries
key_embed   (B, 5500, 256) <-- from encoder (shared, not detached at key level)

4 decoder layers, each:
  1. CaloHitMaskTask.forward() ->
     per_query_logit (B, 191, 5500)   for attention masking
     calo_node_logit (B, 1, 5500)     max-pooled for loss
  2. Inverted physics prior: attn_mask = learned_mask & ~node_is_track
  3. Cross-attention: queries attend to calo nodes only
  4. Self-attention + FFN on queries
  5. NO bidirectional CA (unidirectional, saves memory)
  6. Deep supervision: BCE + Dice on master mask at every layer

Final tasks:
  - CaloHitMaskTask -> calo_node_logit (B, 1, 5500)
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

| Loss key | Task | Computed at | Target |
|---|---|---|---|
| `calo_layer_{0-3}_calo_mask_mask_bce` | CaloHitMaskTask | Each decoder layer | `calo_hs_energy > 0.15` |
| `calo_layer_{0-3}_calo_mask_mask_dice` | CaloHitMaskTask | Each decoder layer | `calo_hs_energy > 0.15` |
| `calo_final_calo_mask_mask_bce` | CaloHitMaskTask | Final | `calo_hs_energy > 0.15` |
| `calo_final_calo_mask_mask_dice` | CaloHitMaskTask | Final | `calo_hs_energy > 0.15` |
| `calo_final_calo_fraction_l1` | PileupCaloFractionTaskV2 | Final only | `calo_hard_scatter_energy` (HS clusters only) |

---

## Prediction Outputs

```python
preds = {
    # Stream A
    "track_layer_0": {"mask": {"pflow_node_valid": (B,1,5500), "pflow_node_prob": (B,1,5500)}},
    "track_layer_1": {"mask": {...}},
    "track_layer_2": {"mask": {...}},
    "track_layer_3": {"mask": {...}},
    "track_final": {
        "mask": {"pflow_node_valid": (B,1,5500), "pflow_node_prob": (B,1,5500)},
        "vz_regression": {"pflow_vz": (B, 1)},
    },

    # Stream B
    "calo_layer_0": {"calo_mask": {"calo_node_prob": (B,5500), "calo_node_valid": (B,5500)}},
    "calo_layer_1": {"calo_mask": {...}},
    "calo_layer_2": {"calo_mask": {...}},
    "calo_layer_3": {"calo_mask": {...}},
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
| `val/calo_mask_f1` | `calo_final.calo_mask.calo_node_prob` vs `calo_hs_energy > 0.15` |
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

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `max_hs_tracks` | `model.init_args` | 190 | Max HS tracks in hybrid queries (set >= max in data) |
| `teacher_forcing` | `model.init_args` | true | Use GT HS tracks during training bridge |
| `hs_energy_threshold` | `CaloHitMaskTask`, `PileupCaloFractionTaskV2` | 0.15 | Min HS energy to count a cluster as HS |
| `null_weight` | `CaloHitMaskTask` | 0.1 | Weight for non-HS clusters in BCE loss |
| `bidirectional_ca` | `calo_decoder.decoder_layer_config` | false | Disabled for memory savings in Stream B |
| `batch_size` | `data` | 48 | Reduced from 64 for GPU memory |

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
| Queries | 1 learnable query | 1 track query + 191 hybrid calo queries |
| Calo mask | None (implicit via fraction task) | Explicit CaloHitMaskTask with deep supervision |
| Calo fraction loss | All clusters (weighted) | HS clusters only (`E_HS > 0.15`) |
| Gradient flow | Single path (encoder <-> decoder) | Isolated: calo loss detached from encoder/tracks |
| Inference | Direct fraction output | Gated: `frac * (calo_mask > 0.5)` |
| Calo decoder CA | Bidirectional | Unidirectional (q->kv only) |
| Parameters | ~10.5M | ~20.7M |
