# `odd_pileup_reco` — Three-Stream Pileup Removal + Particle-Flow Reconstruction

This document is a **standalone architecture summary** of the `odd_pileup_reco` experiment. It is
written so that you can understand the whole system *without reading the source code* — every
concept is explained in plain language first, and code snippets are included only as illustrations.
File names are given as pointers ("where this lives"), not as the explanation itself.

The experiment trains a single neural network that takes the raw output of an LHC detector — a
messy mixture of ~200 overlapping proton-proton collisions — and produces a clean list of the
particles from the *one* collision we actually care about.

---

## Table of Contents

1. [The Physics Problem](#1-the-physics-problem)
2. [Core Idea: Three Streams](#2-core-idea-three-streams)
3. [End-to-End Architecture](#3-end-to-end-architecture)
4. [Vocabulary (read this first)](#4-vocabulary-read-this-first)
5. [The Data Pipeline](#5-the-data-pipeline)
   - [5.1 What an event looks like](#51-what-an-event-looks-like)
   - [5.2 Node features](#52-node-features)
   - [5.3 The incidence matrix & the pileup token](#53-the-incidence-matrix--the-pileup-token)
   - [5.4 Particle classes & trackless reclassification](#54-particle-classes--trackless-reclassification)
   - [5.5 Truth labels produced](#55-truth-labels-produced)
6. [Shared Encoder](#6-shared-encoder)
   - [6.1 Locality ordering — Morton (Z-order) encoding](#61-locality-ordering--morton-z-order-encoding)
7. [Stream A — Find the Hard-Scatter Tracks](#7-stream-a--find-the-hard-scatter-tracks)
8. [The Bridge — Hybrid Queries](#8-the-bridge--hybrid-queries)
9. [Stream B — Find the Hard-Scatter Calo Clusters](#9-stream-b--find-the-hard-scatter-calo-clusters)
10. [Stream C — Reconstruct the Particles](#10-stream-c--reconstruct-the-particles)
11. [The Pileup Token (Query 0)](#11-the-pileup-token-query-0)
12. [Training vs Inference](#12-training-vs-inference)
13. [Loss & Gradient Flow](#13-loss--gradient-flow)
14. [Training Loop & Metrics](#14-training-loop--metrics)
15. [Configuration Reference](#15-configuration-reference)
16. [Quick Reference: What Each Stream Does](#16-quick-reference-what-each-stream-does)
17. [File Map](#17-file-map)

---

## 1. The Physics Problem

When two bunches of protons cross inside the LHC, they do not produce a single collision. At the
design conditions simulated here ("PU200"), roughly **200 separate proton-proton interactions**
happen at essentially the same time and place. Only **one** of them — the **Hard Scatter (HS)** —
contains the high-energy physics we want to study. The other ~199 are **pileup (PU)**: low-energy
junk that we want to throw away.

The detector cannot tell us which is which. It just reports two kinds of objects per crossing:

- **Tracks** — curved trajectories of charged particles measured by the inner tracker (each comes
  with a momentum, direction, and an estimate of where along the beam it originated).
- **Calorimeter clusters** — blobs of deposited energy in the calorimeter (each comes with a total
  energy, position, and shower-shape information).

A single event in this dataset can contain **thousands** of these objects, and roughly **1 in 200**
of them belongs to the hard scatter. If you try to reconstruct particles directly on this soup, the
pileup overwhelms the signal and the result is poor.

**The goal:** in one end-to-end trainable model, (a) decide which tracks and clusters belong to the
hard scatter, then (b) reconstruct the hard-scatter particles using only those cleaned objects.

---

## 2. Core Idea: Three Streams

Rather than do everything with one undifferentiated network, the model is split into three
specialised **streams** that run one after another, all sitting on top of a single **shared
encoder** that has seen the entire (HS + pileup) event:

| Stream | Job | Output |
|---|---|---|
| **A — Tracks** | Which tracks come from the HS vertex? | a per-track HS/PU score |
| **B — Calo** | Which calorimeter clusters come from the HS? | a per-cluster HS/PU score |
| **C — Reconstruction** | Given only the HS objects, what particles produced them? | a list of particles (class + kinematics + which detector hits belong to each) |

Three design decisions make this work, and they recur throughout the document:

1. **One shared encoder.** Every node (track or cluster, HS or pileup) is embedded and passed
   through a single Transformer encoder, so each object is interpreted *in the context of the whole
   messy event*. Separating signal from pileup is only possible with that global context.
2. **Reconstruction gradients reach the encoder; the calo-query path does not.** Stream C's training
   signal is allowed to flow back into the shared encoder so the encoder learns features that make
   reconstruction easier. Stream B's queries, by contrast, are built from a **detached** copy of the
   embeddings, so Stream B cannot corrupt the representation that Stream A relies on.
3. **The streams are sequential and conditioned on each other.** Stream B is told which tracks A
   thinks are HS; Stream C is told which tracks *and* clusters A and B think are HS. During training
   this conditioning uses the truth (teacher forcing) plus a little injected noise; at inference it
   uses the streams' own predictions.

---

## 3. End-to-End Architecture

```
   Raw event:  ~thousands of tracks + calo clusters  (HS signal ≈ 1 in 200, padded to 5500 nodes)
        │
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  InputNet         24 features ─► 256-dim embedding          │
  │                   + Fourier positional encoding (η, φ)      │
  │  Shared Encoder   8 layers, windowed flash-varlen attention │
  └──────────────────────────────────────────────────────────┘
        │  (the encoder output is also saved for a Stream-C skip connection)
        │
        ├──────────────────────────────────────────────────────────────────┐
        ▼                                                                    │ skip
  ┌─────────────────────────┐                                               │ connection
  │  STREAM A  (tracks)      │  decoder: 4 layers, 1 query                   │
  │  → per-track HS score    │  + regress the HS vertex position vz          │
  └───────────┬─────────────┘                                               │
              │  which tracks are HS?  (truth in training / predicted at inference)
              ▼                                                              │
  ┌──────────────────────────────────────────────┐                          │
  │  BRIDGE                                        │                          │
  │  detach embeddings, then build hybrid queries: │                          │
  │  16 learnable latents + up to 190 HS-track     │                          │
  │  embeddings                                    │                          │
  └───────────┬────────────────────────────────────┘                         │
              ▼                                                              │
  ┌─────────────────────────┐                                               │
  │  STREAM B  (calo)        │  calo-only cross-attention, 4 layers          │
  │  → per-cluster HS score  │                                               │
  └───────────┬─────────────┘                                               │
              │  which tracks & clusters are HS?  (truth+noise in training / predicted at inference)
              ▼                                                              │
  ┌────────────────────────────────────────────────────────────────────────┴──┐
  │  STREAM C  (reconstruction)                                                 │
  │  • keep HS tracks + up to 1200 highest-energy HS clusters                   │
  │  • pack the survivors into a fixed buffer of 1400 nodes                     │
  │  • node features = encoder_output + saved_encoder_output  (skip)            │
  │  • run a full MaskFormer (4 layers, 400 particle queries) that predicts:    │
  │       – particle class (charged hadron / electron / muon / neutral / γ)     │
  │       – which nodes belong to each particle (hit mask + incidence)          │
  │       – particle kinematics (energy, pT, η, sinφ, cosφ)                      │
  │  • query 0 is a dedicated "pileup sink" (see §11)                           │
  └─────────────────────────────────────────────────────────────────────────────┘
```

Sizes shown are the values actually used in the shipped config (`configs/base.yaml`): embedding
dimension **256**, up to **5500** input nodes per event, **190** HS-track query slots + **16**
learnable latents in the bridge, **1200** calo nodes / **1400** total nodes kept for
reconstruction, and **400** reconstruction queries.

---

## 4. Vocabulary (read this first)

- **Node** — one detector object: either a track or a calorimeter cluster. The network treats both
  as elements of a single sequence (length up to 5500, zero-padded).
- **Hard Scatter (HS) / Pileup (PU)** — the interesting collision vs the ~199 junk ones.
- **Query** — a learnable (or constructed) vector that "asks a question" of the node sequence via
  cross-attention. In Stream C, each query is responsible for reconstructing one particle.
- **Incidence matrix** — a `(particles × nodes)` table whose entry `[p, n]` says *what fraction of
  node n's energy belongs to particle p*. Each node's column sums to 1. It is the soft "which hits
  belong to which particle" ground truth.
- **Teacher forcing** — during training, feeding a later stage the *true* answer of an earlier stage
  (instead of the earlier stage's prediction) so it can learn even before the earlier stage is good.
- **Hungarian matching** — an algorithm that finds the cheapest one-to-one pairing between predicted
  particles and true particles, so the loss can compare "the right" prediction to each truth. Needed
  because the network outputs an *unordered set* of particles.
- **Null query** — a reconstruction query that ends up matched to nothing; it is trained to predict
  the special "null" class (class 5).
- **flash-varlen attention** — a memory-efficient attention kernel that packs variable-length
  sequences together (no wasted compute on padding).

---

## 5. The Data Pipeline

*Where this lives: `pflow_data.py` (`ODDDatasetPileup`, `ODDDataModule`).*

### 5.1 What an event looks like

The data is stored as **Parquet** files, sharded into groups, with four parallel files per shard:

- `target_particles-*.parquet` — the truth HS particles (energy, η, φ, PDG id, whether it left a track).
- `tracks-*.parquet` — reconstructed tracks (d0, z0, pT, φ, θ, η, curvature, which truth particle it belongs to, whether it came from the primary HS vertex).
- `calo_clusters-*.parquet` — calorimeter clusters (energy, position, shower shapes, hit counts).
- `target_particles_deps-*.parquet` — per-particle energy deposits in each cluster (used to build the incidence matrix and to compute how much of each cluster's energy is HS).

The data module can either read separate train/val/test paths or, more commonly here, take one
**unified** dataset (`unify_path`) and split it 90% / 10% / 10% with a fixed seed. Events that are
too large — more than `max_nodes` (5500) nodes, or more than `num_objects − 1` (399) HS particles —
are dropped so everything fits in fixed-size tensors.

### 5.2 Node features

Every node is described by a **24-dimensional feature vector**. Tracks and clusters fill different
subsets of these features (the irrelevant ones are zero), plus two flags that tell the network which
kind of object it is. The continuous features are normalised using statistics stored in
`configs/odd_var_transform.yaml` (each variable gets either symmetric min-max scaling or
standardisation, sometimes after a `sqrt` to tame long tails).

| Feature group | Filled for tracks | Filled for clusters |
|---|---|---|
| Direction: `eta`, `cosphi`, `sinphi` | ✓ | ✓ |
| Track-only: `pt`, `d0`, `z0`, `tanlambda`, `omega`, interaction-point `eta_int/phi_int/cos/sin` | ✓ | 0 |
| Cluster-only: `e`, `rho`, `sigma_eta`, `sigma_phi`, `sigma_rho`, `hcal_fraction`, `number_of_hits`, `energy_hits_std`, `max_hit_energy` | 0 | ✓ |
| Flags: `is_track`, `is_cluster` | (1, 0) | (0, 1) |

In addition to these 24 *network-input* features, the loader keeps a set of **raw** (un-scaled)
quantities used by the loss and by reconstruction — e.g. each node's true energy, pT, η, sinφ, cosφ,
the HS energy deposited in each cluster, and the HS energy fraction of each cluster.

The model also computes a **`deltaR_idx`** ordering of the nodes — a Morton (Z-order) code of each
node's (η, φ). The shared encoder uses windowed attention, and sorting nodes by this locality index
means each window covers a compact region of the detector (exact implementation in [§6.1](#61-locality-ordering--morton-z-order-encoding)).

### 5.3 The incidence matrix & the pileup token

The reconstruction ground truth is an **incidence matrix** of shape `(400 particles × up to 5500
nodes)`. Row `p`, column `n` holds the fraction of node `n`'s energy attributable to particle `p`.

The crucial twist is **row 0 = the pileup token**. Instead of just listing HS particles, the matrix
reserves its first row for a single virtual "pileup particle" that *absorbs all the pileup energy*.
The real HS particles are shifted to rows 1, 2, 3, …

Construction (illustrative, from `pflow_data.py`):

```python
incidence = np.zeros((num_objects, n_nodes))           # (400, n_nodes)

# HS tracks → their real particle, shifted by +1 to leave room for row 0
incidence[hs_track_particle_idx + 1, track_col] = 1.0
# Pileup tracks (particle_idx == -1) → row 0
incidence[0, pu_track_col] = 1.0

# HS cluster deposits → real particle (shifted +1)
incidence[dep_particle_idx + 1, dep_cluster_col] = dep_energy
# Pileup token claims each cluster's NON-hard-scatter energy
pu_cluster_energy = np.clip(cluster_energy - cluster_hs_energy, 0, None)
incidence[0, cluster_cols] = pu_cluster_energy

# Column-normalise: every node's energy is split among particles, summing to 1
incidence /= np.clip(incidence.sum(axis=0, keepdims=True), 1e-6, None)
```

Why a pileup token? Without it, the pileup energy in each cluster would have nowhere to go and the
column normalisation would be meaningless. With it, every node's column is a proper probability
distribution over "which particle (including 'pileup') do I belong to", which gives a clean KL-
divergence training target. The pileup token gets the special class **5** and is handled specially
everywhere downstream (see §11).

A node is considered "associated" with a particle (the binary `node_valid` mask) wherever its
incidence value exceeds `incidence_cutval` (config: **0.01**).

### 5.4 Particle classes & trackless reclassification

Truth particles are mapped from their PDG id to **5 physics classes**, with a 6th reserved for
"null/pileup":

| Class | Meaning |
|---|---|
| 0 | Charged hadron (π±, K±, p, …) |
| 1 | Electron / positron |
| 2 | Muon |
| 3 | Neutral hadron (n, K⁰_L, …) |
| 4 | Photon |
| 5 | **Null** (padding / unmatched queries) **and the pileup token** |

**Trackless reclassification.** A charged particle is only reconstructable as "charged" if it
actually left a track. If a particle that *should* be charged (class 0 or 1) or a muon (class 2)
has **no matching track**, it is reclassified to its neutral-equivalent class 3, because from the
detector's point of view it now looks like neutral energy. Illustratively:

```python
trackless = <particles with no matched track>
particle_class[trackless & (particle_class < 2)] += 3   # charged hadron→3, electron→... (shifts to neutral)
particle_class[trackless & (particle_class == 2)] = 3   # muon→neutral hadron
```

### 5.5 Truth labels produced

For each event the loader emits (shapes use N = up to 5500 nodes, P = 400 particle slots):

| Label key | Shape | Meaning |
|---|---|---|
| `tracks_mask` | (N,) | true HS tracks (came from the primary vertex) |
| `node_valid` | (N,) | which node slots are real (not padding) |
| `node_is_track` | (N,) | 1 for tracks, 0 for clusters |
| `calo_hard_scatter_energy` | (N,) | HS energy deposited in each cluster |
| `calo_hard_scatter_energy_frac` | (N,) | HS fraction of each cluster's energy |
| `particle_node_valid` | (1, N) | HS-track mask as a single MaskFormer "object" (Stream A target) |
| `particle_vz` | (1,) | true HS vertex z position (Stream A regression target) |
| `reco_particle_class` | (P,) | class per particle slot; 0 = pileup token, padding = 5 |
| `reco_particle_valid` | (P,) | which particle slots are real (slot 0 = pileup = valid) |
| `reco_particle_incidence` | (P, N) | the column-normalised incidence matrix |
| `reco_particle_node_valid` | (P, N) | binary hit-association mask (incidence > cutval) |
| `reco_particle_{e,pt,eta,sinphi,cosphi}` | (P,) | kinematics; slot 0 (pileup) is NaN and excluded from the regression loss |

---

## 6. Shared Encoder

*Where this lives: the `encoder` block in `configs/base.yaml`, plus the embedding logic in
`model.py::forward`.*

Every node's 24 features are projected to a **256-dim** embedding by a small dense network
(`InputNet`), and a **Fourier positional encoding** computed from each node's (η, φ) is added so the
network knows where in the detector each object sits.

The embeddings are then processed by an **8-layer Transformer encoder** using **windowed
flash-varlen attention** (window size 256). Windowing keeps the cost manageable for thousands of
nodes; sorting nodes by `deltaR_idx` first means each window is a coherent detector region. The
encoder uses "hybrid norm" and a value-residual connection (standard stability tricks).

### 6.1 Locality ordering — Morton (Z-order) encoding

Windowed attention only saves work if nodes that are *physically* close (small ΔR in the η–φ plane)
also end up *adjacent* in the 1-D sequence the encoder slides its window over. The dataset achieves
this by sorting nodes along a **Morton code** (a.k.a. Z-order curve) of their (η, φ) position — this
is exactly the `deltaR_idx` field the encoder sorts on (`input_sort_field: deltaR_idx`).

A Morton code maps a 2-D coordinate to a single integer by **bit-interleaving** the two quantised
axes. Points that are near each other in 2-D share high-order bits and therefore land near each
other on the 1-D curve with high probability — so a contiguous window of 256 sorted positions is, in
practice, a compact η–φ neighbourhood. This makes windowed attention a cheap approximation of true
ΔR-local attention.

The exact implementation (`morton_encode()` in `pflow_data.py`):

```python
def morton_encode(eta, phi, n_bits=16, shift_phi=False):
    # 1. Normalise each axis to [0, 1]:
    eta_norm = clamp((eta + 6.0) / 12.0, 0, 1)        # η assumed within [-6, +6]
    phi_norm = clamp((phi + π) / (2π), 0, 1)          # φ within [-π, +π]
    if shift_phi:                                     # optional Swin-style shift
        phi_norm = fmod(phi_norm + 0.5, 1.0)          # rotate φ by π (move the wrap-seam)

    # 2. Quantise each axis to n_bits = 16-bit integers (grid of 65535 steps):
    scale = (1 << n_bits) - 1                         # 65535
    ei = long(eta_norm * scale)
    pi = long(phi_norm * scale)

    # 3. Interleave the bits: η takes the odd bit positions, φ the even ones:
    result = 0
    for bit in range(n_bits):
        result |= ((ei >> bit & 1) << (2*bit + 1))    # η bit -> position 2*bit+1
        result |= ((pi >> bit & 1) << (2*bit))        # φ bit -> position 2*bit
    return result.double()                            # float64 (52-bit mantissa) is exact here
```

Points worth noting for the paper:

- **Fixed normalisation ranges.** η is mapped from the assumed range `[-6, +6]` and φ from
  `[-π, +π]`; both are clamped, so out-of-range values saturate rather than wrap.
- **16-bit quantisation per axis** → a 65535 × 65535 grid; the interleaved code uses 32 bits and is
  returned as `float64` (whose 52-bit mantissa represents it exactly).
- **Bit layout:** η occupies the odd bit positions (`2·bit+1`), φ the even positions (`2·bit`).
- **Padding:** padded (non-existent) node slots are given `deltaR_idx = +inf` so they always sort to
  the end and never fall inside a real window.
- **Shifted variant (`shift_phi=True`).** A second code `node_deltaR_idx_shifted` is computed by
  rotating φ by π before quantisation, which moves the periodic "tear" at φ = ±π so that genuinely
  ΔR-close tokens straddling ±π become adjacent — intended for Swin-style **shifted-window**
  attention. In the shipped configuration the encoder uses the **unshifted** index only
  (`window_wrap: false`); the shifted code is computed but not consumed.

The same Morton ordering is used by the dataset's `print_deltaR_stats()` diagnostic, which sorts a
sample of events by `deltaR_idx` and measures how well a window of `window_size` covers each node's
true ΔR neighbours — i.e. how good the locality approximation is for a given window size.

One small but important detail: right after the encoder runs, the model **saves a reference to the
encoder output** (`initial_encoder_embed`). This same tensor is reused much later as a skip
connection into Stream C (§10). It is kept *live* (not copied or detached), which is what lets
reconstruction gradients flow all the way back into the encoder.

---

## 7. Stream A — Find the Hard-Scatter Tracks

*Where this lives: `track_decoder` + `track_tasks` in the config; class `PileupMaskFormerDecoder`.*

Stream A asks a single question of the event: **which tracks come from the hard-scatter vertex?**

It uses a 4-layer decoder with **one learnable query**. A custom decoder (`PileupMaskFormerDecoder`,
configured with `prior_mask_key: node_is_track`) restricts the query's attention so it focuses on
the track nodes. The query produces:

- **A per-track HS score** (task name `mask`, an `ObjectHitMaskTask`). This is *not* a set-prediction
  problem and there is no Hungarian matching — it is a straightforward per-node binary classification
  "is this track HS or pileup?", trained with a weighted **BCE + Dice** loss (weights 5.0 and 1.0;
  pileup down-weighted via `null_weight = 0.06`).
- **The HS vertex z position** (task `vz_regression`, an `ObjectRegressionTask` with a smooth-L1
  loss). Knowing where along the beam the hard scatter happened is a useful auxiliary signal.

Because Stream A runs on a shallow copy of the shared state, its work does not disturb the embeddings
that Streams B and C will read.

---

## 8. The Bridge — Hybrid Queries

*Where this lives: `model.py::_build_hybrid_queries` and the bridge block of `forward`.*

Stream B needs to know which tracks Stream A flagged as HS, because those tracks tell it where in the
detector to look for the matching calorimeter energy. The **bridge** turns that information into a
set of **hybrid queries** for Stream B.

Two things happen:

1. **Detach.** The node embeddings are detached from the computation graph before being used to
   build queries. This is deliberate: it means any gradient Stream B generates through its queries
   **cannot** flow back into the shared encoder and disturb Stream A's hard-won track representation.

2. **Build the query set.** The hybrid query tensor has a fixed size of `16 + 190` slots:
   - the first **16** are **learnable latent queries** (free parameters, the same for every event);
   - the remaining **190** are filled with the **embeddings of the HS tracks** themselves (so the
     queries literally carry the physical information of this specific collision). Events with fewer
     than 190 HS tracks leave the extra slots padded-invalid.

   Which tracks count as HS depends on the mode: during training with teacher forcing it uses the
   *true* HS-track mask; at inference it uses Stream A's predicted scores (threshold 0.5).

Illustratively, the HS-track embeddings are gathered with a vectorised argsort (no Python loop):

```python
sort_idx      = torch.argsort(hs_mask.long(), dim=-1, descending=True, stable=True)
track_indices = sort_idx[:, :max_hs_tracks]          # the up-to-190 HS tracks, front-loaded
queries[:, :16, :] = self.calo_latent_queries        # learnable latents
queries[:, 16:, :] = node_embed.gather(1, track_indices_3d)   # HS-track embeddings
```

Contrast with a plain MaskFormer, whose queries are a fixed learnable set identical for every event.
Here, most of the queries are **dynamic and event-specific**.

---

## 9. Stream B — Find the Hard-Scatter Calo Clusters

*Where this lives: `decoder.py::CaloFlashCrossAttentionDecoder` + `tasks.py::CaloNodeMaskTask`.*

Stream B asks: **which calorimeter clusters come from the hard scatter?** It receives the hybrid
queries from the bridge and runs a specialised 4-layer decoder, `CaloFlashCrossAttentionDecoder`,
which:

- **Strips all track nodes out of the key/value sequence**, so the queries cross-attend to *calo
  clusters only*. (The tracks already did their job in Stream A; here we only care about calo.)
- Uses **flash-varlen** attention to pack the variable number of valid calo nodes efficiently.
- Runs **bidirectional cross-attention** each layer (queries attend to calo nodes *and* calo nodes
  attend back to queries), interleaved with query self-attention, so the calo-node embeddings get
  enriched with query context and vice-versa.

After the decoder, a `CaloNodeMaskTask` runs a small dense MLP (256 → 128 → 64 → 32 → 1, SiLU
activations) over **each enriched calo-node embedding** to predict its HS probability. Like Stream A,
this is direct per-node binary classification — no matching.

Its training target is "true HS cluster", defined as a cluster whose HS energy fraction exceeds
**0.10** *and* whose absolute HS energy exceeds **0.15**. The loss combines **BCE + Tversky** (each
weighted 5.0). Pileup nodes are down-weighted with a per-sample weight:

```python
sample_weight = target + null_weight * (1 - target)   # null_weight = 0.05
```

so the network is not swamped by the overwhelming majority of pileup clusters.

#### Tversky loss — exact α / β and why

The Tversky loss generalises Dice by penalising false positives and false negatives with **separate**
weights:

```
tp = Σ p·t        fp = Σ p·(1−t)        fn = Σ (1−p)·t
tversky = (tp + 1) / (tp + α·fp + β·fn + 1)
loss    = 1 − tversky
```

| Parameter | Value | Role |
|---|---|---|
| **α (FP penalty)** | **0.25** | weight on false positives (pileup wrongly called HS) |
| **β (FN penalty)** | **0.75** | weight on false negatives (true HS cluster missed) |

These are the function defaults in `loss.py::mask_tversky_loss` — `CaloNodeMaskTask` calls the loss
without overriding them, so `α = 0.25, β = 0.75` are what run. (With `α = β = 0.5` this would reduce
exactly to Dice.) Because **β > α**, a missed HS cluster is penalised **3× more** than a false alarm —
the loss is deliberately biased toward **recall**.

Why bias toward recall here? Three reinforcing reasons, all rooted in Stream B's role as the
HS-cluster gate feeding Stream C:

1. **A miss is unrecoverable; a false positive is filterable.** If a true HS cluster is dropped here
   (FN), its energy is permanently lost from reconstruction — no later stage can bring it back. A
   spurious pileup cluster that leaks through (FP) is merely extra input that Stream C's Hungarian
   matching and pileup token (§11) can still absorb or reject. The costlier error gets the larger
   weight.
2. **It matches the asymmetric node-selection downstream.** Stream C intentionally over-includes —
   true HS clusters **plus up to 250 sampled false positives** (§12). Penalising FN more heavily than
   FP is the loss-level expression of that same "rather over-keep than miss" philosophy.
3. **Severe class imbalance.** HS clusters are rare versus pileup, so a symmetric objective could win
   by simply under-predicting HS. The recall-biased Tversky (β = 0.75) and the BCE `sample_weight`
   (null cells down-weighted 20× via `null_weight = 0.05`) together counteract that collapse.

The pairing of **BCE + Tversky** is also intentional: BCE supplies stable per-cell gradients, while
Tversky supplies the region-overlap, recall-biased signal. (Streams A and C instead use the symmetric
**BCE + Dice**, because their decoders do not perform this same recall-critical HS pre-filtering.)

> Note: `tasks.py` also contains alternative calo heads that were tried during development
> (`CaloHitMaskTask`, which gives each hybrid query its own mask and max-pools them; and energy-
> *fraction* regression variants `PileupCaloFractionTaskV2` / `CaloNodeFractionTask`). The shipped
> config uses the simple, robust `CaloNodeMaskTask` described above.

---

## 10. Stream C — Reconstruct the Particles

*Where this lives: `model.py::_forward_reco`, the encapsulated `reco_model`, and the `reco_*` config.*

This is where the model finally does particle-flow reconstruction — but **only on the cleaned HS
nodes**, which is the whole point of Streams A and B.

**Step 1 — decide which nodes to keep.** Build the HS-node mask. In the default inference path this
is simply "tracks Stream A predicted HS" OR "clusters Stream B predicted HS (prob ≥ 0.3)". (The
training path adds noise; see §12.)

**Step 2 — cap the calo nodes.** Of the predicted-HS clusters, keep at most the **1200 highest-energy**
ones (`max_reco_calo_nodes`). Reconstruction attention is quadratic in sequence length, so this cap
keeps it affordable while retaining the energetic clusters that matter.

**Step 3 — skip connection.** The node features handed to reconstruction are the **sum of the final
encoder output and the saved initial encoder output** (`reco_embed = key_embed + initial_encoder_embed`).
This gives Stream C both the fully-contextualised representation and a more "local" view, and — because
the saved tensor was never detached — it opens a gradient path from reconstruction straight back to
the encoder.

**Step 4 — pack into a fixed buffer.** All surviving HS nodes (tracks + the top-1200 clusters) are
gathered into a fixed buffer of **1400** nodes (`max_reco_nodes`) using a vectorised argsort+gather.
The indices used for this gather are stored, because the truth incidence matrix has to be reindexed
the same way when computing the loss (§13).

**Step 5 — run a full MaskFormer.** The packed nodes are fed to an **encapsulated standard
MaskFormer** (`reco_model`). Because the nodes are already embedded, a trivial `PassThroughInputNet`
hands them straight in (no re-embedding, no encoder). This inner MaskFormer has a 4-layer decoder
with **400 particle queries** and four tasks:

| # | Task | What it predicts | Loss |
|---|---|---|---|
| 1 | `ObjectClassificationTask` | particle class (6-way: classes 0–4 + null) | weighted cross-entropy (`object_ce`, weight 2); per-class weights `[1, 3, 8, 1.5, 1]`, null weight 0.8 |
| 2 | `ObjectHitMaskTask` | which of the 1400 nodes belong to each particle | BCE (5.0) + Dice (1.0) |
| 3 | `IncidenceRegressionTask` | the soft incidence distribution per particle | KL-divergence (1.0) |
| 4 | `IncidenceBasedRegressionTask` | kinematics: energy, pT, η, sinφ, cosφ | L1 (weight 10) |

Because the model emits an **unordered set** of 400 candidate particles, the loss uses **Hungarian
matching** to pair each prediction with the best-fitting truth particle before scoring. Queries that
match nothing are pushed toward the null class (5).

> The kinematics head's input is `2·256 + 6 = 518`-dimensional (a query embedding, an incidence-
> pooled node embedding, and 6 raw node variables). *(A stale comment in the config still reads
> "2*128 + 6 = 262" from when `dim` was 128 — the real value is 518, matching `dim = 256`.)*

---

## 11. The Pileup Token (Query 0)

*Where this lives: `maskformer.py` (the experiment-local MaskFormer used by `reco_model`).*

Reconstruction query **0** is permanently reserved as a **pileup sink**. It is paired with the
incidence matrix's row 0 (the pileup particle from §5.3). Its job is to soak up the residual pileup
energy that survived Streams A and B, so the *real* particle queries are not forced to explain it.

This requires four special-case tweaks to the otherwise-standard MaskFormer, all in `maskformer.py`:

1. **Excluded from Hungarian matching.** Matching is run on the sub-problem `cost[:, 1:, 1:]`
   (everything except query 0 and target 0); query 0 is then prepended with a fixed assignment to
   target 0. So pileup always matches pileup, and never steals a real particle.
2. **Excluded from the classification loss.** The target class at position 0 is set to `-100`, which
   PyTorch's cross-entropy treats as "ignore" — query 0 gets no classification gradient.
3. **Excluded from the kinematics regression.** Its validity flag is forced False (pileup has no
   meaningful energy/pT/η).
4. **Always "valid" at inference.** In `predict()`, query 0 is hardcoded valid so it bypasses the
   usual null-class filtering and its pileup mask can be inspected.

Note that the pileup token *is* still included in the **mask** and **incidence** losses — we *do*
want the network to learn which nodes are pileup energy.

---

## 12. Training vs Inference

The streams are conditioned on each other, and how that conditioning is sourced differs by mode:

| | Stream B sees these as HS tracks | Stream C keeps these tracks | Stream C keeps these clusters |
|---|---|---|---|
| **Training** (teacher forcing) | the true HS tracks | true HS tracks **+ up to 15 sampled false positives** | true HS clusters **+ up to 250 sampled false positives** |
| **Inference** (default) | Stream A predictions (≥ 0.5) | Stream A predictions (≥ 0.5) | Stream B predictions (≥ 0.3) |
| **Debug inference** (`reco_debug_use_truth_masks`) | — | true HS tracks only | true HS clusters only (oracle) |

**Why inject noise during training?** If Stream C only ever saw perfectly clean HS nodes, it would
fall apart at inference time when Streams A and B make mistakes and let some pileup through. So during
training the model deliberately mixes in a bounded number of the *false positives* its own earlier
streams produced (up to 15 extra tracks, up to 250 extra clusters), teaching reconstruction to be
robust to residual pileup.

The sampling is a simple, fully-vectorised trick: give every eligible (predicted-but-wrong) node a
random score, give ineligible nodes a score of −1, and take the top-k:

```python
extra      = pred_mask & ~truth_mask                 # predicted HS but actually pileup
scores     = torch.where(extra, torch.rand_like(...), -1.0)
_, idx     = scores.topk(n_sample, dim=-1)           # n_sample = 250 (calo) or 15 (tracks)
sampled    = scatter idx into a bool mask, then AND with `extra`
final_mask = truth_mask | sampled
```

*(This is a deterministic-count uniform top-k, not a draw from a normal distribution — the
`reco_calo_noise_std` config field is currently vestigial.)*

---

## 13. Loss & Gradient Flow

**Loss assembly** (`model.py::loss`). Streams A and B compute their losses **directly** — per-node
binary classification, no matching. Stream C is different: the model first **reindexes the truth**
into the 1400-node filtered space (using the gather indices saved during the forward pass), then
hands the whole thing to the inner `reco_model`, which performs the Hungarian matching and per-task
losses itself (including the pileup-token special cases of §11). The outer model just re-prefixes the
results as `reco_*`.

The truth reindexing matters: the incidence matrix is born as `(400 × 5500)` but reconstruction only
sees 1400 nodes, so the columns are gathered down to `(400 × 1400)` with the exact same node ordering
the forward pass used — otherwise predictions and truth would be misaligned.

**Gradient flow.** The detach in the bridge and the live skip connection into Stream C combine to
give a deliberate gradient topology:

```
                Shared Encoder
                     ║
        ╔════════════╬════════════════════════╗
        ║            ║  (saved encoder output, ║
        ▼            ║   live reference)        ║
   Stream A loss     ║                          ║
   (track scores,    ║                          ║
    flows back into  ║                          ║
    the encoder)     ║                          ║
        │ detach ────╫───► Bridge queries        ║
        ▼            ║         │                 ║
   Stream B loss ────╨─────────┘ (CANNOT reach   ║
   (calo scores; query path is detached,         ║
    so no gradient into the encoder this way)    ║
                                                 ▼
                                          Stream C loss
                                          (reconstruction; flows back
                                           through BOTH the encoder output
                                           and the saved skip → trains the
                                           shared encoder jointly)
```

In words: **Stream A and Stream C both train the shared encoder; Stream B's query path is
intentionally walled off** so it can specialise on calo classification without rewriting the shared
representation.

---

## 14. Training Loop & Metrics

*Where this lives: `lightning_module.py::ODDPFlowTwoStream` (a `ModelWrapper` subclass);
entry point `main.py`.*

Training is driven by PyTorch Lightning. The key wrinkle is that **targets are passed into the model's
`forward`** (not just into the loss), because teacher forcing and training-time noise injection need
the truth while building Stream C's inputs.

Each step calls `model(inputs, targets)`, then `model.loss(...)`, and logs the per-stream losses. The
module computes rich per-stream validation metrics:

- **Stream A:** track F1 / precision / recall, plus efficiency-vs-pT and pileup-rejection curves.
- **Stream B:** calo-mask F1 / precision / recall, plus per-event predicted-vs-true HS energy sums
  (neutral and charged) for energy-purity studies.
- **Stream C:** classification accuracy (micro & macro), reconstruction efficiency & purity, and hit-
  mask metrics (with the truth masks reindexed into the 1400-node space, mirroring the forward pass).

At the end of each validation epoch the module un-scales the kinematics and produces a battery of
physics plots (track score distributions, efficiency vs pT, calo energy purity, particle
distributions, and jet-resolution plots from clustering the reconstructed particles), optionally
logged to CometML.

**Optimisation:** AdamW with a one-cycle-style LR schedule (warm up over the first 5% of training
from 6.25e-7 to a peak of 6.25e-5, then anneal back down), weight decay 0.01, gradient clipping 0.1,
`bf16-mixed` precision, up to 100 epochs.

---

## 15. Configuration Reference

*All values from `configs/base.yaml` (the shipped configuration). Defaults in `model.py` sometimes
differ; the config is what actually runs.*

| Group | Key | Value |
|---|---|---|
| **Data** | `num_objects` | 400 (slot 0 = pileup token) |
| | `max_nodes` | 5500 |
| | `batch_size` | 32 |
| | `incidence_cutval` | 0.01 |
| | `hard_scatter_energy_threshold` | 0.1 |
| | split | one dataset, 90 / 10 / 10 |
| **Core** | `dim` | 256 |
| | encoder | 8 layers, flash-varlen, window 256, 16 heads |
| | InputNet | Dense 24 → 256 + Fourier posenc(η, φ), scale 0.1 |
| **Stream A** | decoder | `PileupMaskFormerDecoder`, 4 layers, **1 query**, `prior_mask_key=node_is_track` |
| | tasks | HS-track mask (BCE 5.0 + Dice 1.0, null_weight 0.06); vz regression (smooth-L1, weight 0.01179) |
| **Bridge** | `num_latent_queries` | 16 |
| | `max_hs_tracks` | 190 |
| **Stream B** | decoder | `CaloFlashCrossAttentionDecoder`, 4 layers, calo-only, 16 heads |
| | task | `CaloNodeMaskTask`: Dense 256→[128,64,32]→1, BCE 5.0 + Tversky 5.0 (α 0.25 / β 0.75, recall-biased), null_weight 0.05, pred 0.3, HS thresholds frac 0.10 / energy 0.15 |
| **Stream C** | `max_reco_calo_nodes` | 1200 |
| | `max_reco_nodes` | 1400 |
| | `num_reco_queries` | 400 |
| | `calo_pred_threshold` | 0.3 |
| | `reco_calo_noise_mean` | 250 |
| | `reco_track_noise_mean` | 15 |
| | decoder | `MaskFormerDecoder`, 4 layers, 400 queries, bidirectional CA |
| | matcher | scipy Hungarian, parallel, 16 jobs |
| | tasks | classification (6-way, weights `[1,3,8,1.5,1]`, null 0.8); hit mask (BCE 5.0 + Dice 1.0); incidence (KL 1.0); kinematics (L1, weight 10, input 518) |
| **Optim** | AdamW | LR 6.25e-7 → 6.25e-5 → 6.25e-7, warmup 5%, wd 0.01 |
| **Trainer** | | bf16-mixed, grad-clip 0.1, ≤100 epochs |

---

## 16. Quick Reference: What Each Stream Does

| Concept | Stream A (tracks) | Stream B (calo) | Stream C (reconstruction) |
|---|---|---|---|
| **Question** | Which tracks are HS? | Which clusters are HS? | What particles made these HS nodes? |
| **Queries** | 1 learnable | 16 latents + 190 HS-track embeds (dynamic) | 400 learnable (query 0 = pileup sink) |
| **Decoder** | `PileupMaskFormerDecoder` | `CaloFlashCrossAttentionDecoder` (calo-only) | standard `MaskFormerDecoder` inside `reco_model` |
| **Prediction type** | per-track binary | per-cluster binary | set of particles (class + mask + incidence + kinematics) |
| **Hungarian matching** | none | none | yes (pileup token excluded) |
| **Gradient to encoder** | yes | **no** (queries detached) | yes (incl. live skip connection) |
| **Conditioned on** | — | A's HS-track decision | A + B HS decisions (+ noise in training) |

---

## 17. File Map

| File | Role |
|---|---|
| `model.py` | `TwoStreamMaskFormer` — the orchestrator: embedding, shared encoder, Streams A/B/C, bridge, loss/predict, the encapsulated `reco_model`, `PassThroughInputNet`. |
| `maskformer.py` | Experiment-local `MaskFormer` used as Stream C's `reco_model`; adds the pileup-token (query 0) handling in `loss`/`predict`/matching. |
| `decoder.py` | `CaloFlashCrossAttentionDecoder` (calo-only bidirectional cross-attention) and `CaloMaskFormerDecoder` (inverted-prior variant). |
| `tasks.py` | Calo task heads: `CaloNodeMaskTask` (used) plus development alternatives `CaloHitMaskTask`, `PileupCaloFractionTaskV2`, `CaloNodeFractionTask`. |
| `pflow_data.py` | `ODDDatasetPileup` / `ODDDataModule`: Parquet loading, 24-feature nodes, incidence matrix + pileup token, class mapping, all truth labels. |
| `lightning_module.py` | `ODDPFlowTwoStream`: training/validation/test steps, per-stream metrics, epoch-end physics plots. |
| `configs/base.yaml` | The full experiment configuration (data, model, three streams, optimiser, trainer). |
| `configs/odd_var_transform.yaml` | Per-variable normalisation statistics. |
| `main.py` | CLI entry point wiring `ODDPFlowTwoStream` + `ODDDataModule` to `base.yaml`. |
| `README.md` | The older, narrower deep-dive comparing `TwoStreamMaskFormer` to the stock `MaskFormer` (this document supersedes it for whole-experiment overview). |

---

*This summary was generated by reading the current source (`model.py`, `maskformer.py`,
`decoder.py`, `tasks.py`, `pflow_data.py`, `lightning_module.py`, `configs/base.yaml`). Where the
older `README.md` and in-code comments disagreed with the live code, the live code wins.*
