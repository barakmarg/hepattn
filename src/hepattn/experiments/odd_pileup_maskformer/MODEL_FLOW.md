# Proxy PV MaskFormer — Model Flow Documentation

## Overview

A single "Proxy PV" query sweeps across all tracks in the event, builds a representation
of the hard scatter vertex, then broadcasts that context to every node for per-node predictions.

```
Raw Nodes (tracks + clusters)
    → InputNet → Encoder (windowed attention)
        → PileupMaskFormerDecoder (4 layers, 1 query, iterative mask + vz deep supervision)
            → 3 Task Heads (mask, vz, calo_fraction)
```

---

## Data Input

### What enters the model (`inputs` dict)

| Key | Shape | Description |
|-----|-------|-------------|
| `node_features` | `(B, 5500, 21)` | Stacked per-node feature vector (tracks padded + clusters padded) |
| `node_valid` | `(B, 5500)` | Bool mask — True for real nodes, False for padding |
| `node_eta` | `(B, 5500)` | Raw eta coordinate |
| `node_phi` | `(B, 5500)` | Raw phi coordinate |
| `node_deltaR_idx` | `(B, 5500)` | Z-order Morton index for windowed attention sorting (base ordering) |
| `node_deltaR_idx_shifted` | `(B, 5500)` | Morton index with phi rotated by π — moves the periodic boundary from ±π to 0 (shifted ordering) |
| `node_e` | `(B, 5500)` | Raw total energy per node (raw variable, passed through) |
| `node_is_track` | `(B, 5500)` | 1 for tracks, 0 for calo clusters (raw variable, passed through) |

**Feature ordering in `node_features[:, :, 21]`:**
Tracks use: `phi, cosphi, sinphi, eta, eta_int, phi_int, cosphi_int, sinphi_int, pt, d0, z0, tanlambda, omega` + zeros for calo fields + `is_track=1, is_cluster=0`
Clusters use: zeros for track fields + `phi, cosphi, sinphi, eta, e, rho, sigma_eta, sigma_phi, sigma_rho, hcal_fraction` + `is_track=0, is_cluster=1`

**Layout:** First `N_tracks` positions = tracks, next `N_clusters` positions = clusters, remaining = padding zeros.

### What enters the loss (`targets` dict)

| Key | Shape | Description |
|-----|-------|-------------|
| `tracks_mask` | `(B, 5500)` | Bool — True if track belongs to hard scatter vertex |
| `node_valid` | `(B, 5500)` | Bool mask for valid nodes |
| `node_is_track` | `(B, 5500)` | 1=track, 0=cluster |
| `node_e` | `(B, 5500)` | Raw total energy |
| `calo_hard_scatter_energy` | `(B, 5500)` | True HS energy per cluster (0 for tracks) |
| `calo_hard_scatter_energy_frac` | `(B, 5500)` | True HS energy fraction per cluster |
| `particle_valid` | `(B, 1)` | Always `[True]` — 1 hard scatter per event |
| `particle_node_valid` | `(B, 1, 5500)` | **Mask task target** — True only for HS tracks |
| `particle_vz` | `(B, 1)` | True hard scatter vertex z position |

---

## Stage 1: InputNet + Embedding

**File:** `hepattn/models/input.py`

```
node_features  (B, 5500, 21)
    → Dense MLP (21 → 256)
    + FourierPositionEncoder(eta, phi → 256)  [additive]
    → node_embed  (B, 5500, 256)
```

Also sets `key_is_node` — a bool vector of length 5500 (all True, since we only have one input type).

---

## Stage 2: Windowed Self-Attention Encoder (Shifted-Window Morton)

**File:** `hepattn/experiments/odd_pileup_maskformer/models.py` (`ShiftedWindowEncoder`)

```
node_embed  (B, 5500, 256)

  For each of the 8 encoder layers:
    even layers (0, 2, 4, 6): sort by node_deltaR_idx         (base Morton order)
    odd  layers (1, 3, 5, 7): sort by node_deltaR_idx_shifted (phi-rotated Morton order)

    → sort tokens into current layer's order
    → (flash-varlen) unpad to remove padding tokens
    → windowed self-attention (16 heads, window_size=512) + FFN + residual
    → repad
    → unsort back to original token order

  → key_embed  (B, 5500, 256)    [updated, context-aware node embeddings]
```

**Why two orderings?**
phi is periodic: particles at phi=+π and phi=−π are genuine ΔR≈0 neighbours, but a
single Morton ordering places them at opposite ends of the sorted sequence, so they
never share a window. By rotating phi by π in alternating layers (Swin-style shifted
windows), the "tear" in the ordering migrates between ±π and 0 each layer:

- Base order: particles near phi=0 are co-located; ±π boundary is a blind spot.
- Shifted order: particles near ±π are co-located; phi=0 becomes the blind spot.

Each token pair that is split in even layers is guaranteed to be within the same
window in odd layers, and vice-versa. Every region of the eta-phi cylinder gets full
coverage over two consecutive layers.

Each node now encodes its local neighbourhood context (within ΔR ≈ 0.5) with no
systematic dead zone at the phi=±π boundary.

---

## Stage 3: MaskFormer Decoder — 4 Iterative Layers

**File:** `hepattn/models/decoder.py`

### Initial State
```
query_embed  (B, 1, 256)    ← learnable parameter (nn.Parameter, random init)
key_embed    (B, 5500, 256) ← from encoder
```

### Per-Layer Loop (repeated 4 times)

#### Step A — ObjectHitMaskTask + physics prior produce attention mask
```
query_embed  (B, 1, 256)
    → object_net  Dense(256 → 256)
    → x_object   (B, 1, 256)

key_embed    (B, 5500, 256)
    → hit_net    Dense(256 → 256)
    → x_hit      (B, 5500, 256)

mask_logit = logit_scale * einsum("bqd,bnd->bqn", x_object, x_hit)
           = (B, 1, 5500)

learned_mask = sigmoid(mask_logit) >= 0.5
             = (B, 1, 5500)  bool

physics_prior = node_is_track.unsqueeze(1)
              = (B, 1, 5500)  bool   ← hard constraint: calo always False

attn_mask = learned_mask & physics_prior
          = (B, 1, 5500)  bool
```

Two-stage mask:
1. **Hard prior** (`PileupMaskFormerDecoder`): calo clusters are gated out unconditionally.
   The query can never attend to calo regardless of what the learned mask says.
2. **Learned mask** (ObjectHitMaskTask): among tracks, learns to focus on HS tracks
   and suppress PU tracks. This is the only thing the mask needs to learn.

Ground truth target (`particle_node_valid`) = HS tracks only, consistent with both constraints.

#### Step B — Cross-Attention (query attends to nodes through mask)
```
q_ca: query_embed  (B, 1, 256)
      cross-attends to  key_embed  (B, 5500, 256)
      using  attn_mask  (B, 1, 5500)  ← only attend to high-probability nodes
      → query_embed  (B, 1, 256)    [updated: "what do the likely-HS nodes look like?"]
```

#### Step C — Query Self-Attention
```
q_sa: query_embed  (B, 1, 256)
      self-attends (trivial with 1 query, acts as normalisation)
      → query_embed  (B, 1, 256)
```

#### Step D — Query FFN
```
q_dense: query_embed  (B, 1, 256)
         → FFN (256 → 256)
         → query_embed  (B, 1, 256)
```

#### Step E — Bidirectional: Nodes attend back to query
```
kv_ca: key_embed   (B, 5500, 256)
       cross-attends to  query_embed  (B, 1, 256)
       using transposed mask  (B, 5500, 1)
       → key_embed  (B, 5500, 256)    [updated: each node "sees" the PV query context]

kv_dense: key_embed  → FFN → key_embed  (B, 5500, 256)
```

After bidirectional CA, every node embedding carries information about the Proxy PV.

#### Step F — Intermediate vz Regression (deep supervision)
```
query_embed  (B, 1, 256)
    → Dense(256 → 1)
    → pflow_regr  (B, 1, 1)   ← predicted vz at this layer

Loss: smooth_L1(pflow_regr, particle_vz)
```

vz is predicted at every decoder layer (deep supervision), forcing the query to learn the
vertex position early. This gives later layers a better-informed query for masking.

#### Step G — Intermediate Loss (logged per-layer)
```
outputs["layer_0"]["mask"] = {
    "pflow_node_logit": (B, 1, 5500)   ← mask logits at this layer
}
outputs["layer_0"]["vz_regression"] = {
    "pflow_regr": (B, 1, 1)            ← vz prediction at this layer
}
```

Losses computed at every intermediate layer:
- `mask_bce`:  5.0 × BCE(sigmoid(logit), particle_node_valid)  [null_weight=0.06 for class imbalance]
- `mask_dice`: 1.0 × Dice(sigmoid(logit), particle_node_valid)
- `vz_regression_smooth_l1`: smooth_L1(pflow_regr, particle_vz)

This forces both the mask and vz to sharpen progressively across layers.

---

## Stage 4: Final Task Heads

After 4 decoder layers, the final `query_embed (B, 1, 256)` and `node_embed (B, 5500, 256)`
are passed to all 3 tasks.

### Task 1 — ObjectHitMaskTask (final layer)

Same as intermediate computation above, but now used for final loss.

```
Output:  pflow_node_logit  (B, 1, 5500)
Predict: pflow_node_valid  (B, 1, 5500)  bool  [sigmoid >= 0.5]
         pflow_node_prob   (B, 1, 5500)  float [sigmoid probability]

Loss targets:  particle_node_valid  (B, 1, 5500)  [HS tracks only]
Loss:
  final_mask_mask_bce  = 5.0 × BCE(logit, target)   [null_weight=0.06 for class imbalance]
  final_mask_mask_dice = 1.0 × Dice(logit, target)
```

**Interpretation:** Query[0] has learned to distinguish HS tracks from pileup tracks.
The mask is the model's unified track classifier — `pflow_node_prob` gives continuous HS probability.
`null_weight=0.06` downweights PU samples in BCE (equivalent to pos_weight≈16.24).

---

### Task 2 — ObjectRegressionTask: Vz Regression (with deep supervision)

```
Input:   query_embed  (B, 1, 256)
         → Dense(256 → 1)
Output:  pflow_regr  (B, 1, 1)   ← predicted hard scatter vertex z

Loss target:  particle_vz  (B, 1)   [truth vz, NOT given to the model]
Loss:
  final_vz_regression_smooth_l1 = smooth_L1(pflow_regr, particle_vz)
```

**Deep supervision:** Unlike the final-only pattern, vz regression runs at every decoder layer
(same as the mask task). This forces the query to learn the vertex position from Layer 0 onwards,
rather than deferring it to the final stage.

**Why this matters:** The query must encode the vertex position to classify tracks by z0 proximity.
Vz is in the loss only — the model must discover it from the pattern of track z0 values.

---

### Task 3 — PileupCaloFractionTask

```
Input:   query_embed  (B, 1, 256)    → query_net  Dense(256→256) → (B, 1, 256)
         node_embed   (B, 5500, 256) → hit_net    Dense(256→256) → (B, 5500, 256)

         query expanded: (B, 5500, 256)
         concatenated:   (B, 5500, 512)
         → output_net   Dense(512→1, hidden=[256,128,64,32], SiLU, final=Sigmoid)

Output:  calo_frac  (B, 5500)   ∈ [0, 1]   (fraction of cluster energy from HS)

Predict:
  calo_hs_fraction  (B, 5500)

Loss (clusters only, where node_valid & ~node_is_track):
  E_pred_HS = calo_frac[cluster_mask] * node_e[cluster_mask]
  E_true_HS = calo_hard_scatter_energy[cluster_mask]
  weights   = 1 + 25.3 * (E_true_HS > 0.05)    [upweight signal clusters]
  final_calo_fraction_l1 = mean(|E_pred_HS - E_true_HS| * weights)
```

---

## Loss Summary

All losses flow back through the entire network.

| Loss key (logged) | Task | Computed at | Target |
|---|---|---|---|
| `layer_0_mask_mask_bce` | ObjectHitMaskTask | Decoder layer 0 | `particle_node_valid` |
| `layer_0_mask_mask_dice` | ObjectHitMaskTask | Decoder layer 0 | `particle_node_valid` |
| `layer_0_vz_regression_smooth_l1` | ObjectRegressionTask | Decoder layer 0 | `particle_vz` |
| `layer_1_mask_mask_bce` | ObjectHitMaskTask | Decoder layer 1 | `particle_node_valid` |
| `layer_1_mask_mask_dice` | ObjectHitMaskTask | Decoder layer 1 | `particle_node_valid` |
| `layer_1_vz_regression_smooth_l1` | ObjectRegressionTask | Decoder layer 1 | `particle_vz` |
| `layer_2_mask_mask_bce` | ObjectHitMaskTask | Decoder layer 2 | `particle_node_valid` |
| `layer_2_mask_mask_dice` | ObjectHitMaskTask | Decoder layer 2 | `particle_node_valid` |
| `layer_2_vz_regression_smooth_l1` | ObjectRegressionTask | Decoder layer 2 | `particle_vz` |
| `layer_3_mask_mask_bce` | ObjectHitMaskTask | Decoder layer 3 | `particle_node_valid` |
| `layer_3_mask_mask_dice` | ObjectHitMaskTask | Decoder layer 3 | `particle_node_valid` |
| `layer_3_vz_regression_smooth_l1` | ObjectRegressionTask | Decoder layer 3 | `particle_vz` |
| `final_mask_mask_bce` | ObjectHitMaskTask | Final | `particle_node_valid` |
| `final_mask_mask_dice` | ObjectHitMaskTask | Final | `particle_node_valid` |
| `final_vz_regression_smooth_l1` | ObjectRegressionTask | Final | `particle_vz` |
| `final_calo_fraction_l1` | PileupCaloFractionTask | Final | `calo_hard_scatter_energy` |

---

## Prediction Outputs (from `model.predict()`)

```python
preds = {
    "final": {
        "mask": {
            "pflow_node_valid": (B, 1, 5500),  # bool, HS node association mask
            "pflow_node_prob":  (B, 1, 5500),  # float, continuous HS probability
        },
        "vz_regression": {
            "pflow_vz": (B, 1),  # predicted hard scatter vertex z
        },
        "calo_fraction": {
            "calo_hs_fraction": (B, 5500),  # HS energy fraction per node ∈ [0,1]
        },
    },
    "layer_0": { "mask": { ... }, "vz_regression": { ... } },
    "layer_1": { ... },
    "layer_2": { ... },
    "layer_3": { ... },
}
```

---

## Metrics Logged (val/test)

| Metric | Source |
|--------|--------|
| `val/track_f1` | `mask.pflow_node_prob` vs `tracks_mask` (track nodes only) |
| `val/track_precision` | same |
| `val/track_recall` | same |
| `val/calo_frac_mae` | `calo_fraction.calo_hs_fraction` vs `calo_hard_scatter_energy_frac` |
| `val/calo_frac_mae_{type}` | same, split by pileup/mixed/HS energy fraction bins |
| `val/calo_hs_energy_ratio` | sum(pred HS energy) / sum(true HS energy) |
| `val/vz_mae` | `vz_regression.pflow_vz` vs `particle_vz` |

---

## Component Communication Diagram

```
Dataset
  ├── inputs["node_features"]  (B,5500,21) ──→ InputNet ──→ node_embed (B,5500,256)
  ├── inputs["node_deltaR_idx"]         ───────────────────→ base sort key for even encoder layers
  ├── inputs["node_deltaR_idx_shifted"] ───────────────────→ shifted sort key for odd encoder layers
  └── inputs["node_e"], ["node_is_track"] ───────────────→ x (raw variables, for loss + prior)

InputNet
  └── node_embed (B,5500,256) ──→ Encoder

Encoder (8 layers, window=512)
  └── key_embed (B,5500,256) ──→ PileupMaskFormerDecoder

PileupMaskFormerDecoder (4 layers)
  ├── query_embed (B,1,256) ← learnable initial parameter
  │
  └─── For each layer:
        ├── ObjectHitMaskTask.forward(query_embed, key_embed)
        │     └── learned_mask (B,1,5500) bool  [HS vs PU tracks]
        │
        ├── ObjectRegressionTask.forward(query_embed)  [deep supervision]
        │     └── pflow_regr (B,1,1) → vz prediction at this layer
        │
        ├── Physics prior (hard constraint)
        │     └── attn_mask = learned_mask & node_is_track  (B,1,5500)
        │         calo positions always False — enforced, not learned
        │
        ├── Decoder Layer (cross-attn with attn_mask)
        │     ├── query attends to tracks only, focused on HS tracks
        │     └── key attends back to query → all nodes (incl. calo) get PV context
        │
        └── intermediate loss: BCE+Dice on mask + smooth_L1 on vz

Final Task Heads (receive final query_embed + key_embed)
  ├── ObjectHitMaskTask (unified track classifier)
  │     ├── in:  query_embed (B,1,256) + key_embed (B,5500,256)
  │     └── out: pflow_node_logit (B,1,5500) → loss: mask_bce + mask_dice
  │         also: pflow_node_prob (B,1,5500) → used for track metrics
  │
  ├── ObjectRegressionTask
  │     ├── in:  query_embed (B,1,256)
  │     └── out: pflow_regr (B,1,1) → loss: smooth_L1 vs particle_vz
  │
  └── PileupCaloFractionTask
        ├── in:  cat(key_embed, query_embed.expand) (B,5500,512)
        └── out: calo_frac (B,5500) → loss: L1[cluster nodes] vs calo_hard_scatter_energy
```
