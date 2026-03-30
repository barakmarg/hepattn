# TwoStreamMaskFormer — Architecture Walkthrough

This document is a deep-dive into `TwoStreamMaskFormer` ([model.py](model.py)), comparing it line-by-line to the original
`MaskFormer` ([../../models/maskformer.py](../../models/maskformer.py)). All code excerpts are lifted verbatim from those
two files.

---

## Table of Contents

1. [Problem Statement & Design Goals](#1-problem-statement--design-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Constructor Comparison](#3-constructor-comparison)
4. [Forward Pass — Step by Step](#4-forward-pass--step-by-step)
   - [4.1 Embedding & Shared Encoder](#41-embedding--shared-encoder)
   - [4.2 Stream A — Track MaskFormer](#42-stream-a--track-maskformer)
   - [4.3 Bridge — Hybrid Query Construction](#43-bridge--hybrid-query-construction)
   - [4.4 Stream B — Calo MaskFormer](#44-stream-b--calo-maskformer)
   - [4.5 Stream C — Reconstruction](#45-stream-c--reconstruction)
5. [Training vs Inference: Teacher Forcing & Noise Sampling](#5-training-vs-inference-teacher-forcing--noise-sampling)
6. [Loss Computation](#6-loss-computation)
7. [Prediction](#7-prediction)
8. [Gradient Flow Diagram](#8-gradient-flow-diagram)
9. [Data Pipeline Summary](#9-data-pipeline-summary)
10. [Config Key Differences](#10-config-key-differences)

---

## 1. Problem Statement & Design Goals

An LHC collision event contains ~200 simultaneous proton-proton interactions ("pileup"). Only one of them — the Hard Scatter (HS) — is physically interesting. Raw detector output (tracks + calorimeter clusters) contains a mix of HS and pileup contributions at a ratio of roughly 1:200. Directly running particle reconstruction on this messy mixture degrades performance.

**Goal:** Build a model that (a) identifies the HS tracks and calo clusters and (b) runs particle reconstruction *only* on the cleaned HS nodes — all in a single end-to-end differentiable pipeline.

**Design decisions:**
- A single shared encoder sees all nodes (HS + pileup together) for full context.
- Pileup removal and reconstruction are split into sequential streams so each specializes.
- Reconstruction gradients flow back through the shared encoder (not detached) so the encoder learns to highlight HS-useful features.
- Stream B nodes are detached before forming hybrid queries so Stream B gradients do not corrupt Stream A's learned representation.

---

## 2. Architecture Overview

```
Input nodes (tracks + calo, ~5500 total including pileup)
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │  InputNet   (embed to dim=128)                 │
  │  Shared Encoder (8 × windowed flash-varlen SA) │
  └────────────────────────────────────────────────┘
        │                         │
        │                         └─────────────────────────┐
        │                                                     │ (skip connection for Stream C)
        ▼                                                     │
  ┌──────────────────────────┐                               │
  │  STREAM A                │                               │
  │  Track MaskFormerDecoder │                               │
  │  (4 layers, 1 query)     │                               │
  │  → HS track mask         │                               │
  └──────────┬───────────────┘                               │
             │  teacher forcing (train) / predicted (infer)   │
             ▼                                               │
  ┌──────────────────────────────────────────────────┐       │
  │  BRIDGE                                          │       │
  │  detach node embeddings                          │       │
  │  → num_latent + HS_track embeddings as queries   │       │
  └──────────┬───────────────────────────────────────┘       │
             ▼                                               │
  ┌──────────────────────────┐                               │
  │  STREAM B                │                               │
  │  CaloFlashCrossAttn Dcdr │                               │
  │  (4 layers, hybrid Q)    │                               │
  │  → calo HS/PU mask       │                               │
  └──────────┬───────────────┘                               │
             │  teacher forcing + noise sampling (train)      │
             │  / predicted (infer)                          │
             ▼                                               │
  ┌──────────────────────────────────────────────────────────┴──┐
  │  STREAM C                                                    │
  │  Filter to HS nodes (≤ max_reco_calo_nodes=1200 calo,        │
  │               all HS tracks)                                 │
  │  Skip: reco_embed = encoder_out + initial_encoder_embed      │
  │  Repack to (B, max_reco_nodes=1400, D)                       │
  │  → encapsulated MaskFormer (4 layers, 400 queries)           │
  │     • Classification (6 classes)                             │
  │     • Hit mask (BCE + Dice)                                  │
  │     • Incidence regression (KL-div)                          │
  │     • Incidence-based regression (pt, eta, phi, e)           │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Constructor Comparison

### Original `MaskFormer.__init__`

```python
# maskformer.py
class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        decoder: MaskFormerDecoder,   # single decoder
        tasks: nn.ModuleList,          # single task list
        dim: int,
        target_object: str = "particle",
        matcher: nn.Module | None = None,
        ...
    ):
        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.tasks = tasks
        self.tasks = tasks
        self.matcher = matcher
        # ONE set of learnable latent queries
        self.query_initial = nn.Parameter(torch.randn(self.num_queries, dim))
```

### `TwoStreamMaskFormer.__init__`

```python
# model.py
class TwoStreamMaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        track_decoder: MaskFormerDecoder,   # Stream A decoder
        track_tasks: nn.ModuleList,
        calo_decoder: nn.Module,             # Stream B decoder (different type!)
        calo_tasks: nn.ModuleList,
        reco_decoder: MaskFormerDecoder | None = None,  # Stream C (optional)
        reco_tasks: nn.ModuleList | None = None,
        reco_matcher: nn.Module | None = None,
        dim: int = 128,
        max_hs_tracks: int = 200,
        num_latent_queries: int = 1,
        teacher_forcing: bool = True,
        ...
    ):
        # THREE separate decoders, each with their own tasks
        self.track_decoder = track_decoder
        self.track_decoder.tasks = track_tasks
        self.calo_decoder = calo_decoder
        self.reco_decoder = reco_decoder

        # THREE separate query sets with different semantics:
        self.track_query_initial = nn.Parameter(torch.randn(1, dim))       # 1 latent for track classification
        self.calo_latent_queries = nn.Parameter(torch.randn(num_latent_queries, dim))  # latent + track copies
        # Stream C queries live inside self.reco_model (an encapsulated MaskFormer)

        # Encapsulate a full standard MaskFormer as Stream C
        if self.reco_decoder is not None:
            self.reco_model = MaskFormer(
                input_nets=nn.ModuleList([PassThroughInputNet("node", "node_embed_raw")]),
                encoder=None,          # no encoder — we provide pre-embedded nodes
                decoder=self.reco_decoder,
                tasks=self.reco_tasks,
                dim=self.dim,
                target_object=self.reco_target_object,
                matcher=self.reco_matcher,
                raw_variables=["node_e", "node_pt", "node_eta", "node_sinphi", "node_cosphi", "node_is_track"]
            )
```

**Key difference:** `MaskFormer` has one decoder + one task list + one query set. `TwoStreamMaskFormer` has three decoders, three task lists, three query semantics — and Stream C is literally a full `MaskFormer` instance embedded inside.

The `PassThroughInputNet` is a minimal adapter:

```python
# model.py
class PassThroughInputNet(nn.Module):
    """Passes pre-embedded nodes directly through — bypasses the InputNet projection."""
    def __init__(self, name="node", key="node_embed_raw"):
        super().__init__()
        self.input_name = name
        self.key = key
        self.posenc = None

    def forward(self, inputs):
        return inputs[self.key]  # No-op: just returns the pre-computed embedding
```

Compared to a real `InputNet` which runs a Dense projection + positional encoding, this is intentionally trivial.

---

## 4. Forward Pass — Step by Step

### 4.1 Embedding & Shared Encoder

**Identical pattern in both models.** Both embed inputs with `input_nets`, merge into `key_embed`, and pass through an optional encoder:

```python
# maskformer.py — MaskFormer.forward()
for input_net in self.input_nets:
    x[input_name + "_embed"] = input_net(inputs)
    x[input_name + "_valid"] = inputs[input_name + "_valid"]

x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

if self.encoder is not None:
    x["key_embed"] = self.encoder(x["key_embed"],
                                   x_sort_value=x.get(f"key_{self.input_sort_field}"),
                                   kv_mask=x.get("key_valid"))
```

```python
# model.py — TwoStreamMaskFormer.forward()  (same pattern, one addition)
# ... identical embedding + encoder code ...

# ADDITION: save initial encoder output for Stream C skip connection
initial_encoder_embed = x["key_embed"]
```

This single extra line is what makes the Stream C skip connection work. There's no `.clone()` or `.detach()` here — the reference is kept live so Stream C gradients flow back through the shared encoder.

---

### 4.2 Stream A — Track MaskFormer

**Original `MaskFormer`** runs its one decoder:

```python
# maskformer.py
x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)
x["query_valid"] = torch.full((batch_size, self.num_queries), True, ...)
x, outputs = self.decoder(x, input_names)
outputs["final"] = {}
for task in self.tasks:
    outputs["final"][task.name] = task(x)
```

**`TwoStreamMaskFormer` Stream A** is a direct parallel of this but uses `track_query_initial` (1 query) and operates on a shallow copy `x_track` so it doesn't contaminate the shared `x` dict:

```python
# model.py
x_track = dict(x)  # shallow copy — same tensors, separate dict
x_track["query_embed"] = self.track_query_initial.expand(batch_size, -1, -1)  # 1 learnable query
x_track["query_valid"] = torch.full((batch_size, 1), True, device=...)

x_track, track_outputs = self.track_decoder(x_track, input_names)

track_outputs["final"] = {}
for task in self.track_tasks:
    track_outputs["final"][task.name] = task(x_track)
    if isinstance(task, IncidenceRegressionTask):
        x_track["incidence"] = track_outputs["final"][task.name][task.outputs[0]].detach()
    if isinstance(task, ObjectClassificationTask):
        x_track["class_probs"] = track_outputs["final"][task.name][task.outputs[0]].detach()
```

The `ObjectHitMaskTask` here produces a *single scalar mask per track* (is this track from the HS vertex?). There is no Hungarian matching in Stream A — this is a node-level binary classification, not a set-prediction problem.

---

### 4.3 Bridge — Hybrid Query Construction

This is entirely absent from the original `MaskFormer`.

The bridge extracts which tracks are classified as HS (using truth during training, predictions at inference) and combines them with a few learnable latent queries to form the "hybrid queries" for Stream B:

```python
# model.py — Bridge
is_track = x["node_is_track"].bool().squeeze(-1)  # (B, N)

# Teacher forcing: truth mask during training, prediction at inference
if self.training and self.teacher_forcing:
    hs_track_mask = x["tracks_mask"].bool() & is_track
else:
    mask_logits = track_outputs["final"]["mask"]["pflow_node_logit"]  # (B, 1, N)
    hs_track_mask = (mask_logits.squeeze(1).sigmoid() >= 0.5) & is_track

# CRITICAL: detach before building hybrid queries
# Stream B gradients must not propagate back into Stream A's space
node_embed_detached = x["node_embed"].detach()  # (B, N, D) — gradient blocked here

hybrid_queries, query_valid = self._build_hybrid_queries(
    node_embed_detached, hs_track_mask, batch_size
)
```

`_build_hybrid_queries` packs the queries into a padded tensor of fixed size `(B, num_latent_queries + max_hs_tracks, D)`:

```python
# model.py
def _build_hybrid_queries(self, node_embed, hs_mask, batch_size):
    Q = self.max_hs_tracks + NL  # fixed-size padded tensor (NL = num_latent_queries)
    queries = torch.zeros(batch_size, Q, D, ...)
    valid   = torch.zeros(batch_size, Q, ...)

    queries[:, :NL, :] = self.calo_latent_queries  # first NL slots: learnable latents
    valid[:, :NL] = True

    for b in range(batch_size):
        track_indices = hs_mask[b].nonzero(as_tuple=True)[0]
        n = min(len(track_indices), self.max_hs_tracks)
        if n > 0:
            queries[b, NL : NL + n, :] = node_embed[b, track_indices[:n], :]
            valid[b, NL : NL + n] = True

    return queries, valid
```

The original `MaskFormer` query preparation is simply:
```python
# maskformer.py — comparison
x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)  # static, same for all events
```

The bridge queries are *dynamic per event* — they encode physical information specific to that collision.

---

### 4.4 Stream B — Calo MaskFormer

**Original `MaskFormer`** would run:
```python
# maskformer.py — generic decoder call
x, outputs = self.decoder(x, input_names)
```

**`TwoStreamMaskFormer` Stream B** runs the calo-specialized `CaloFlashCrossAttentionDecoder` with the hybrid queries, on a separate `x_calo` copy:

```python
# model.py
x_calo = dict(x)  # another shallow copy
x_calo["query_embed"] = hybrid_queries   # (B, Q, D) where Q = latent + HS tracks
x_calo["query_valid"] = query_valid      # (B, Q) — padded HS tracks marked invalid

x_calo, calo_outputs = self.calo_decoder(x_calo, input_names)

calo_outputs["final"] = {}
for task in self.calo_tasks:
    calo_outputs["final"][task.name] = task(x_calo)
```

The `CaloFlashCrossAttentionDecoder` uses flash-varlen attention internally, strips out track nodes from the key sequence (calo-only cross-attention), and runs bidirectional cross-attention between hybrid queries and calo nodes.

The `CaloNodeMaskTask` then runs a Dense MLP over each enriched calo node embedding to predict HS probability — a node-level binary classification (no matching needed).

---

### 4.5 Stream C — Reconstruction

This is where `TwoStreamMaskFormer` diverges most from `MaskFormer`, and where the design is most elaborate.

#### Step 1: Build the HS node mask

```python
# model.py — _forward_reco()
# Training: truth track mask + truth calo mask + sampled noise from predictions
if self.training and self.teacher_forcing and targets is not None:
    reco_track_mask = targets["tracks_mask"].bool() & is_track

    truth_calo_mask = (
        (calo_hs_frac > self.calo_hs_frac_threshold)
        & (calo_hs_energy > self.calo_hs_energy_threshold)
        & ~is_track & node_valid
    )

    # Sample ~N(reco_calo_noise_mean, reco_calo_noise_std) predicted-but-wrong nodes
    # to make Stream C robust to imperfect pileup removal at inference time
    extra_pred = pred_calo_mask & ~truth_calo_mask   # predicted HS but truth says pileup
    n_extra = int(max(0, torch.normal(mean=..., std=...).item()))
    # ... random sample n_extra of extra_pred per batch element ...
    reco_calo_mask = truth_calo_mask | sampled_extra
else:
    # Inference: use predicted masks from Streams A and B
    reco_track_mask = (track_logits.sigmoid() >= 0.5) & is_track
    reco_calo_mask  = (calo_logits.sigmoid() >= self.calo_pred_threshold) & ~is_track
```

The original `MaskFormer` runs on all input nodes unconditionally. This filtering step reduces the sequence length by ~4× before Stream C, which has a quadratic attention cost.

#### Step 2: Top-k calo selection

```python
# model.py
# Keep only top max_reco_calo_nodes calo nodes by cluster energy
node_e = x.get("node_e", ...)
calo_energy_masked = torch.where(reco_calo_mask, node_e, torch.zeros_like(node_e))
max_k = min(self.max_reco_calo_nodes, calo_energy_masked.shape[-1])
_, topk_indices = calo_energy_masked.topk(max_k, dim=-1, sorted=False)
reco_calo_topk = torch.zeros_like(reco_calo_mask)
reco_calo_topk.scatter_(1, topk_indices, True)
reco_calo_topk = reco_calo_topk & reco_calo_mask   # AND to keep only HS-predicted ones

reco_node_mask = reco_track_mask | reco_calo_topk  # final HS node mask
```

#### Step 3: Skip connection + gather filtered nodes

```python
# model.py
# Skip connection: sum of final encoder output with the saved initial output
# This gives Stream C access to both local and global information
reco_embed = x["key_embed"] + initial_encoder_embed  # (B, N, D)

# Repack variable-length HS nodes into a fixed-size padded tensor
filtered_embed = torch.zeros(batch_size, max_rn, D, ...)
filtered_valid  = torch.zeros(batch_size, max_rn, dtype=torch.bool, ...)
reco_node_indices = torch.zeros(batch_size, max_rn, dtype=torch.long, ...)

for b in range(batch_size):
    indices = reco_node_mask[b].nonzero(as_tuple=True)[0]
    n = min(len(indices), max_rn)
    if n > 0:
        filtered_embed[b, :n]     = reco_embed[b, indices[:n]]
        filtered_valid[b, :n]     = True
        reco_node_indices[b, :n]  = indices[:n]   # saved for loss reindexing
        for k in raw_keys:
            filtered_raw[k][b, :n] = x[k][b][indices[:n]]

self._reco_node_indices    = reco_node_indices  # used in _build_reco_targets
self._reco_filtered_valid  = filtered_valid
```

#### Step 4: Run encapsulated standard MaskFormer

```python
# model.py
reco_inputs = {
    "node_embed_raw": filtered_embed,   # pre-embedded, PassThroughInputNet returns this directly
    "node_valid":     filtered_valid,
    "node_is_track":  filtered_is_track,
    **filtered_raw,                     # node_e, node_pt, node_eta, etc.
}
reco_outputs = self.reco_model(reco_inputs)
```

Compare this to the original `MaskFormer.forward()`:

```python
# maskformer.py — what reco_model.forward() does internally
for input_net in self.input_nets:
    x[input_name + "_embed"] = input_net(inputs)   # PassThroughInputNet: just returns inputs["node_embed_raw"]
    x[input_name + "_valid"] = inputs[input_name + "_valid"]

x["key_embed"] = ... concat ...
# encoder=None, so skipped
x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)  # 400 learnable reco queries
x["query_valid"]  = torch.full((batch_size, self.num_queries), True, ...)
x, outputs = self.decoder(x, input_names)  # standard MaskFormerDecoder, 4 layers
outputs["final"] = {}
for task in self.tasks:
    outputs["final"][task.name] = task(x)  # classification, mask, incidence, regression
```

**The `reco_model` is a fully unmodified `MaskFormer` — zero customization.** The entire complexity lives in the filtering + packaging that prepares its inputs.

---

## 5. Training vs Inference: Teacher Forcing & Noise Sampling

| Stage | Track mask | Calo mask |
|---|---|---|
| **Training** | Ground truth `targets["tracks_mask"]` | Ground truth HS calo + ~N(300, 50) sampled predicted-but-wrong nodes |
| **Inference** | Stream A predicted sigmoid ≥ 0.5 | Stream B predicted sigmoid ≥ `calo_pred_threshold=0.2` |

The noise sampling during training is essential for robustness:

```python
# model.py
# Extra predicted nodes not in truth — pileup that Stream B incorrectly predicted as HS
extra_pred = pred_calo_mask & ~truth_calo_mask

# Sample a random number of them to inject into Stream C's inputs
n_extra = int(max(0, torch.normal(
    mean=torch.tensor(float(self.reco_calo_noise_mean)),   # ~300
    std=torch.tensor(self.reco_calo_noise_std),            # 50
).item()))

sampled_extra = torch.zeros_like(extra_pred)
for b in range(batch_size):
    extra_indices = extra_pred[b].nonzero(as_tuple=True)[0]
    if len(extra_indices) > 0 and n_extra > 0:
        k = min(n_extra, len(extra_indices))
        perm = torch.randperm(len(extra_indices), device=device)[:k]
        sampled_extra[b, extra_indices[perm]] = True

reco_calo_mask = truth_calo_mask | sampled_extra
```

Without this, Stream C trains only on clean HS nodes and would degrade at inference time when Stream B makes imperfect predictions. The sampled noise teaches Stream C to be robust to ~300 residual pileup nodes.

The original `MaskFormer` has no teacher forcing concept — it processes all nodes equally in all stages.

---

## 6. Loss Computation

### Original `MaskFormer.loss()`

```python
# maskformer.py
def loss(self, outputs, targets):
    if self.matcher is not None:
        # 1. Compute cost matrix from all tasks for each decoder layer
        costs = {}
        for layer_name, layer_outputs in outputs.items():
            layer_costs = None
            for task in self.tasks:
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue
                task_costs = task.cost(layer_outputs[task.name], targets)
                for cost in task_costs.values():
                    layer_costs = cost if layer_costs is None else layer_costs + cost
            costs[layer_name] = layer_costs.detach()

        # 2. Hungarian matching + permute predictions for each layer
        for layer_name, cost in costs.items():
            pred_idxs = self.matcher(cost, targets[f"{self.target_object}_valid"])
            for task in self.tasks:
                if not task.permute_loss: continue
                for output_name in task.outputs:
                    outputs[layer_name][task.name][output_name] = \
                        outputs[layer_name][task.name][output_name][batch_idxs, pred_idxs]

    # 3. Compute losses
    losses = {}
    for layer_name in outputs:
        losses[layer_name] = {}
        for task in self.tasks:
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)
    return losses
```

### `TwoStreamMaskFormer.loss()`

Stream A and B use **direct binary classification** (no matching), while Stream C reuses the exact `MaskFormer.loss()` logic by delegating:

```python
# model.py
def loss(self, outputs, targets):
    losses = {}

    # Stream A: direct per-node binary loss, no matching
    for layer_name, layer_out in outputs.items():
        if layer_name.startswith("track_"):
            real_layer = layer_name[len("track_"):]
            losses[layer_name] = {}
            for task in self.track_tasks:
                if real_layer != "final" and not task.has_intermediate_loss:
                    continue
                if task.name in layer_out:
                    losses[layer_name][task.name] = task.loss(layer_out[task.name], targets)

        # Stream B: same pattern
        elif layer_name.startswith("calo_"):
            ...

    # Stream C: reindex targets to filtered node space, then delegate entirely
    if hasattr(self, "reco_model") and any(k.startswith("reco_") for k in outputs):
        reco_targets = self._build_reco_targets(targets)  # reindex node dim: 5500 → 1400
        reco_layer_outputs = {k[len("reco_"):]: v for k, v in outputs.items() if k.startswith("reco_")}
        reco_losses = self.reco_model.loss(reco_layer_outputs, reco_targets)  # full Hungarian matching
        for layer_name, layer_losses in reco_losses.items():
            losses[f"reco_{layer_name}"] = layer_losses

    return losses
```

`_build_reco_targets` reindexes the `(B, num_objects, max_nodes=5500)` incidence matrix to `(B, num_objects, max_reco_nodes=1400)` using `torch.gather`:

```python
# model.py
def _build_reco_targets(self, targets):
    indices = self._reco_node_indices    # (B, 1400) — stored during forward()

    for key in ["node_valid", "node_incidence"]:
        src_key = f"{self.reco_target_object}_{key}"
        if src_key in reco_targets and reco_targets[src_key].dim() == 3:
            num_objects = reco_targets[src_key].shape[1]
            idx_expanded = indices.unsqueeze(1).expand(-1, num_objects, -1)  # view, zero-copy
            reco_targets[src_key] = torch.gather(reco_targets[src_key], 2, idx_expanded)
            # (B, 400, 5500) → (B, 400, 1400) in one GPU kernel
```

---

## 7. Prediction

### Original `MaskFormer.predict()`

```python
# maskformer.py
def predict(self, outputs):
    preds = {}
    for layer_name, layer_outputs in outputs.items():
        preds[layer_name] = {}
        for task in self.tasks:
            if layer_name != "final" and not task.has_intermediate_loss:
                continue
            preds[layer_name][task.name] = task.predict(layer_outputs[task.name])
    return preds
```

### `TwoStreamMaskFormer.predict()`

```python
# model.py
def predict(self, outputs):
    preds = {}

    for layer_name, layer_out in outputs.items():
        preds[layer_name] = {}
        if layer_name.startswith("track_"):
            for task in self.track_tasks:
                ...
                preds[layer_name][task.name] = task.predict(layer_out[task.name])
        elif layer_name.startswith("calo_"):
            ...

    # Stream C: delegate to encapsulated reco_model
    if hasattr(self, "reco_model") and any(k.startswith("reco_") for k in outputs):
        reco_layer_outputs = {k[len("reco_"):]: v for k, v in outputs.items() if k.startswith("reco_")}
        reco_preds = self.reco_model.predict(reco_layer_outputs)  # full predict with thresholding
        for layer_name, layer_preds in reco_preds.items():
            preds[f"reco_{layer_name}"].update(layer_preds)
```

The prefix juggling (`reco_` → strip → delegate → re-add `reco_`) keeps `reco_model`'s internal naming convention intact while namespacing its outputs in the outer model.

---

## 8. Gradient Flow Diagram

```
Shared Encoder
     ║
     ╠══════════════════════╗
     ║                      ║ (initial_encoder_embed — live reference, not clone)
     ▼                      ║
  Stream A tasks             ║
     │ (gradients flow       ║
     │  back through         ║
     │  shared encoder)      ║
     │                       ║
     │ detach()              ║
     ▼                       ║
  Bridge queries             ║
     │                       ║
     ▼                       ║
  Stream B tasks             ║
     │ (Stream B cannot      ║
     │  affect encoder via   ║
     │  query path —         ║
     │  detach blocked it)   ║
     │                       ║
     └──────────►  Stream C ◄╝
                  │ (gradients flow back through
                  │  both x["key_embed"] AND initial_encoder_embed
                  │  → shared encoder trained jointly by reco signal)
```

**Key contrast with original `MaskFormer`:** All tasks in the standard model share exactly one gradient path through one encoder. Here, Stream A's track classification gradient and Stream C's reconstruction gradient both flow through the shared encoder (in parallel), while Stream B's gradient is intentionally blocked.

---

## 9. Data Pipeline Summary

The dataset ([pflow_data.py](pflow_data.py)) loads Parquet files and builds the following labels for Stream C:

| Target key | Shape | Description |
|---|---|---|
| `reco_particle_class` | `(num_objects,)` | PDG→class mapping (0=photon, 1=electron, 2=muon, 3=charged hadron, 4=neutral hadron, 5=null/padding) |
| `reco_particle_valid` | `(num_objects,)` | `class < 5` |
| `reco_particle_node_valid` | `(num_objects, max_nodes)` | incidence > cutval (binary hit association mask) |
| `reco_particle_incidence` | `(num_objects, max_nodes)` | energy-weighted, column-normalized particle→node matrix |
| `reco_particle_e/pt/eta/sinphi/cosphi` | `(num_objects,)` | regression targets (scaled) |

The incidence matrix construction follows the `odd` experiment:

```python
# pflow_data.py
incidence = torch.zeros(self.num_objects, self.max_nodes)

# Track association: particle → track (binary)
incidence[t_particle_idx, track_idx] = 1.0

# Cluster association: particle → cluster (energy-weighted)
incidence[d_particle_idx, d_cluster_idx + n_tracks] = d_energy

# Column-normalize: each node's associations sum to 1
col_sums = incidence.sum(0, keepdim=True).clamp(min=1e-6)
incidence = incidence / col_sums
```

Trackless charged particles are reclassified to their neutral counterpart:

```python
# pflow_data.py
is_charged = particle_class.isin([1, 2, 3])  # electron, muon, charged hadron
has_track  = particle_has_track.bool()
# class 1→0 (e→photon), 2→2 (muon stays), 3→4 (charged hadron→neutral hadron)
reclassify_map = {1: 0, 3: 4}
particle_class[(is_charged & ~has_track)] = reclassify_map.get(particle_class, particle_class)
```

---

## 10. Config Key Differences

**Stream C specific config in [configs/base.yaml](configs/base.yaml):**

```yaml
model:
  model:
    class_path: hepattn.experiments.odd_pileup_reco.model.TwoStreamMaskFormer
    init_args:
      # Noise injection parameters (training only)
      reco_calo_noise_mean: 300    # ~N(300, 50) extra pileup nodes per event
      reco_calo_noise_std: 50.0
      max_reco_calo_nodes: 1200    # Top-k calo filter
      max_reco_nodes: 1400         # Padded sequence length for Stream C
      num_reco_queries: 400        # Max reconstructed particles
      reco_target_object: reco_particle
      calo_pred_threshold: 0.2     # Stream B threshold for inference filtering
      teacher_forcing: true

      reco_decoder:
        class_path: hepattn.models.decoder.MaskFormerDecoder
        init_args:
          num_decoder_layers: 4
          num_queries: 400
          mask_attention: true
          decoder_layer_config:
            dim: 128
            hybrid_norm: true
            bidirectional_ca: true
            attn_kwargs:
              num_heads: 16

      reco_matcher:
        class_path: hepattn.models.matcher.Matcher
        init_args:
          default_solver: scipy
          parallel_solver: true
          n_jobs: 16

      reco_tasks:
        # 1. 6-class classification (0-4 + null=5)
        - class_path: hepattn.models.task.ObjectClassificationTask
          init_args:
            num_classes: 5
            null_weight: 0.5
            loss_class_weights: [1.0, 3.0, 8.0, 1.5, 1.0]

        # 2. Hit mask (BCE + Dice)
        - class_path: hepattn.models.task.ObjectHitMaskTask

        # 3. Incidence regression (KL-div)
        - class_path: hepattn.models.task.IncidenceRegressionTask

        # 4. Incidence-based kinematics regression
        #    input_size = 2*dim + 6 raw vars = 2*128 + 6 = 262
        - class_path: hepattn.models.task.IncidenceBasedRegressionTask
          init_args:
            fields: ["e", "pt", "eta", "sinphi", "cosphi"]
            net:
              input_size: 262   # ← 262, not 518 — dim=128 not 256
```

Compare to the original `odd` experiment where `dim=256` and `input_size=518`. Here `dim=128` halves the channel dimension; the raw-variable count (6) stays the same.

---

## Quick Reference: What Maps to What

| Concept | `MaskFormer` | `TwoStreamMaskFormer` |
|---|---|---|
| Input embedding | `input_nets[i](inputs)` | same |
| Shared encoder | `self.encoder(key_embed)` | same, + `initial_encoder_embed` saved |
| Latent queries | `self.query_initial` (static) | `self.track_query_initial` (A), `self.calo_latent_queries` (B), `self.reco_model.query_initial` (C) |
| Decoder | `self.decoder(x, ...)` | A: `track_decoder`, B: `calo_decoder`, C: `reco_model.decoder` |
| Tasks | `self.tasks` | A: `track_tasks`, B: `calo_tasks`, C: inside `reco_model` |
| Hungarian matching | inside `self.matcher` | only for Stream C, delegated to `reco_model` |
| Loss | `self.loss(outputs, targets)` | A+B: direct, C: `self.reco_model.loss(reco_layer_outputs, reco_targets)` |
| Predict | `self.predict(outputs)` | A+B: direct, C: `self.reco_model.predict(reco_layer_outputs)` |
| Gradient isolation | none | `node_embed.detach()` before Bridge |
| Skip connection | none | `x["key_embed"] + initial_encoder_embed` for C |
| Noise robustness | none | ~N(300,50) extra predicted calo nodes injected into C during training |
