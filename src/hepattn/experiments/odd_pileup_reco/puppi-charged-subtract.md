# PUPPI Charged-Subtract: Detailed Algorithm and Math

This note documents the **PUPPI with full charged-particle subtraction** baseline
implemented in
[hepattn/experiments/odd_pileup_reco/puppi_charged_subtract.py](puppi_charged_subtract.py).
This variant extends [PUPPI-Perfect](puppi-perfect.md): in addition to the
HS-charged subtraction used there, it **also subtracts the PU-charged calo
deposits** from each cluster, leaving the PUPPI $\alpha$-shape weight to
discriminate only **HS-neutral vs PU-neutral** in the residual — the regime
PUPPI was originally designed for.

It is an even **stronger truth-assisted upper bound** than PUPPI-Perfect:
both HS and PU charged contributions are removed exactly using truth.

## 1) Goal and High-Level Idea

We want jet inputs that go beyond ideal PFlow linking:

- **LV (hard-scatter) tracks** are kept at full $p_T$ with weight 1.
- **PU tracks** are removed (weight 0).
- **Clusters** contribute only their **fully-charged-subtracted neutral remainder**
  (subtracting truth HS-charged *and* PU-charged calo deposits), then receive a
  PUPPI neutral weight.

Conceptually:

```
tracks (LV: w=1, PU: w=0)        clusters (e − hs_charged_e − pu_charged_e)
        │                                       │
        └───────────────┬───────────────────────┘  ← PUPPI α-shape on neutral residual
                        ▼
                   jet clustering
```

This emulates **perfect track-cluster matching** *plus* **perfect PU-charged
subtraction**, leaving PUPPI to do only neutral-vs-neutral discrimination.
It is intentionally truth-heavy and is a tighter ceiling than `puppi_perfect`.

## 2) Inputs and Truth Information Used

The implementation operates on the **all-vertices chunked dataset**:

```
/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked/
```

which exposes four aligned parquet streams per shard:

- `calo_clusters-*.parquet` — per-event cluster lists with
  `total_cluster_energy`, `cluster_eta`, `cluster_phi`.
- `tracks-*.parquet` — per-event track lists with `pt`, `eta`, `phi`,
  `vertex_primary` (1 = LV/HS, anything else = PU; `None` is mapped to `-1`).
- `target_particles-*.parquet` — per-event particle lists for **all vertices**
  (HS *and* PU), with `pt`, `eta`, `phi`, `vertex_primary`, `has_track`.
- `target_particles_deps-*.parquet` — per-event flat lists of
  `(particle_idx, cluster_idx, total_energy_deps_in_cluster)`.

Per-node arrays produced by `load_charged_subtract_events()`, with node order
`[tracks, clusters]` within each event:

- `node_pt`, `node_eta`, `node_phi`, `node_e` — kinematics; `node_e` for
  tracks is the **massless proxy** $p_T \cdot \cosh(\eta)$.
- `node_is_track` — bool.
- `node_valid` — bool, always True for loaded particles.
- `tracks_mask` — int8: $1$ for LV tracks (`vertex_primary == 1`), $0$
  otherwise. Clusters are always $0$.
- `calo_hs_charged_e`, `calo_pu_charged_e` — float32 per-node; $0$ for
  tracks; per-cluster sum of energy deposits from charged particles
  (`has_track == True`) split by `vertex_primary`.

HS-truth-particle arrays for truth-jet clustering:

- `hs_part_pt`, `hs_part_eta`, `hs_part_phi` — kinematics restricted to
  particles with `vertex_primary == 1`.

**Truth usage summary**

1. **HS/PU track labels** via `tracks_mask` (in `compute_puppi_weights`).
2. **HS-charged calorimeter deposit** via `calo_hs_charged_e`.
3. **PU-charged calorimeter deposit** via `calo_pu_charged_e` (the addition
   over PUPPI-Perfect).
4. **HS particles** for the ground-truth jet target.

This is **strictly more truth** than PUPPI-Perfect, which only knew about HS
charged deposits and `tracks_mask`.

## 3) Per-Cluster Charged-Energy Aggregation

For each event, the per-cluster HS-charged and PU-charged energies are built
from the flat `target_particles_deps` rows using only truth columns.

Let:

- $p_{vp}[k] = $ `vertex_primary` of particle $k$ (from `target_particles`).
- $p_{ht}[k] = $ `has_track` of particle $k$ (from `target_particles`).
- $(\,a_n, c_n, e_n\,) = $ the $n$-th row of `target_particles_deps`
  with `particle_idx`, `cluster_idx`, `total_energy_deps_in_cluster`.

Then for each deposit row $n$:

$$
 \text{is\_hs\_charged}_n = p_{ht}[a_n] \wedge (p_{vp}[a_n] = 1)
$$

$$
 \text{is\_pu\_charged}_n = p_{ht}[a_n] \wedge (p_{vp}[a_n] \neq 1)
$$

and per cluster $c$:

$$
 E^{\text{HS-charged}}_c = \sum_{n:\,c_n = c,\,\text{is\_hs\_charged}_n} e_n
$$

$$
 E^{\text{PU-charged}}_c = \sum_{n:\,c_n = c,\,\text{is\_pu\_charged}_n} e_n
$$

implemented with `np.add.at` (scatter-add) on the two boolean masks.

These two arrays are zero-padded for the leading $n_{\text{tracks}}$ entries
when assembled into the combined `[tracks, clusters]` node order.

## 4) Neutral Remainder Construction (Perfect PFlow Linking + PU-Charged Subtraction)

For every node $i$:

- If it is a **track**, we keep the track transverse momentum:

$$
 p_{T,i}^{\text{node}} = p_{T,i}^{\text{track}}
$$

- If it is a **cluster**, we subtract both HS-charged and PU-charged energies
  and floor at zero:

$$
 E_{i}^{\text{neutral}} = \max(E_i - E^{\text{HS-charged}}_i - E^{\text{PU-charged}}_i, 0)
$$

$$
 p_{T,i}^{\text{cluster}} = \frac{E_{i}^{\text{neutral}}}{\cosh(\eta_i)}
$$

So the combined per-node $p_T$ used for jet inputs is:

$$
 p_{T,i}^{\text{node}} =
  \begin{cases}
    p_{T,i}^{\text{track}} & \text{if } i \text{ is track} \\
    E_{i}^{\text{neutral}}/\cosh(\eta_i) & \text{if } i \text{ is cluster}
  \end{cases}
$$

This is recomputed both inside `compute_puppi_weights_charged_subtract()` (to
override `node_e`) and inside `cluster_puppi_charged_subtract_jets()` at the
event level (to feed weighted-$p_T$ to fastjet).

**Optional fallback to PUPPI-Perfect**: passing `subtract_pu_charged=False`
sets $E^{\text{PU-charged}}_i \equiv 0$ everywhere, which reproduces the
`puppi_perfect` neutral-remainder definition. This is intentional — the same
function can run both ablations.

**Why this matters**: clusters in dense PU-200 events are heavily contaminated
by charged PU. Subtracting both HS- and PU-charged deposits leaves a
near-pure neutral residual, on which $\alpha$-shape PUPPI suppression operates
in the regime it was designed for.

## 5) PUPPI Weights (Alpha-Shape) With Explicit Math

The actual weights are computed by `compute_puppi_weights()` from
[hepattn/experiments/odd_pileup_reco/puppi.py](puppi.py), called via the
wrapper `compute_puppi_weights_charged_subtract()`. The only difference from
the standard call is that the **cluster `node_e` is overridden** with the
fully-subtracted residual:

$$
 E_i \leftarrow \begin{cases}
   E_i & \text{track} \\
   \max(E_i - E^{\text{HS-charged}}_i - E^{\text{PU-charged}}_i, 0) & \text{cluster}
 \end{cases}
$$

This ensures that any cluster $p_T$ used inside PUPPI is the **neutral
remainder**, identical in form to PUPPI-Perfect but with the extra
PU-charged subtraction.

### 5.1 Alpha Definition

For each target node $i$, the local shape variable is

$$
 \alpha_i = \log \left(\sum_{j \in \text{LV tracks}} \frac{p_{T,j}^2}{\Delta R_{ij}^2}\right)
$$

with

$$
 \Delta R_{ij}^2 = (\eta_i - \eta_j)^2 + (\Delta \phi_{ij})^2,
$$

$$
 \Delta \phi_{ij} = \text{wrap}(\phi_i - \phi_j) \in (-\pi, \pi].
$$

The sum includes **only LV-track neighbors** $j$ with

$$
 0 < \Delta R_{ij}^2 < R_0^2
$$

and pairs with $\Delta R_{ij}^2 < 10^{-4}$ are discarded to avoid self-pair
blow-ups. Targets include **PU tracks**, **neutral clusters**, and **LV
tracks** (the last only for the low-PU adjustment).

If a target has no valid neighbors inside $R_0$, then $\alpha_i = 0$.

### 5.2 Calibration (Median and RMS)

The calibration sample is the set of **PU tracks** with

$$
 |\eta| < \eta_{\max} \quad \text{and} \quad p_T > p_T^{\text{min}}
$$

where $p_T^{\text{min}} = \text{rms\_pt\_min}$ and
$\eta_{\max} = \text{eta\_max\_extrap}$. Let $\{\alpha_k\}$ be this sample
(excluding zeros). Then

$$
 \alpha_{\text{med}} = \text{median}(\{\alpha_k\})
$$

$$
 \alpha_{\text{rms}} = \sqrt{\langle (\alpha_k - \alpha_{\text{med}})^2 \rangle}
$$

but with the **CMS low-PU convention**: the RMS is computed only from
$\alpha_k \le \alpha_{\text{med}}$ when possible. If fewer than two PU
tracks are available, the code falls back to the full target set; if still
insufficient, it uses $\alpha_{\text{med}} = 0$ and $\alpha_{\text{rms}} = 1$.

### 5.3 Low-PU LV Adjustment (Optional)

If `apply_lv_adjust=True`, an additional shift is applied (CMS PuppiAlgo):

$$
 l_{\text{adjust}} = \frac{N_{\text{LV} \le \alpha_{\text{med}}}}
 {N_{\text{LV} \le \alpha_{\text{med}}} + 0.5\,N_{\text{PU,cal}}}
$$

$$
 \Delta = \sqrt{\chi^2_{1,\,l_{\text{adjust}}} \cdot \alpha_{\text{rms}}}
$$

$$
 \alpha_{\text{med}} \leftarrow \alpha_{\text{med}} - \Delta,
 \quad
 \alpha_{\text{rms}} \leftarrow \max(\alpha_{\text{rms}} - \Delta, 10^{-6})
$$

where $\chi^2_{1,\,l}$ is the $l$-quantile of $\chi^2$ with 1 d.o.f.

### 5.4 Weight Mapping (Signed $\chi^2$)

For each node $i$:

$$
 \Delta_i = \alpha_i - \alpha_{\text{med}}
$$

$$
 \ell_i = \frac{\Delta_i\,|\Delta_i|}{\alpha_{\text{rms}}^2}
$$

$$
 w_i = \begin{cases}
   F_{\chi^2_1}(\ell_i) & \ell_i > 0 \\
   0 & \ell_i \le 0
 \end{cases}
$$

where $F_{\chi^2_1}$ is the CDF of $\chi^2$ with 1 d.o.f. (the CMS "signed
$\chi^2$" mapping). Category overrides then apply:

$$
 w_i = 1 \;\text{for LV tracks},\quad w_i = 0 \;\text{for PU tracks}.
$$

Two hard cuts are then enforced:

1) **Minimum weight**

$$
 w_i = 0 \quad \text{if } w_i < \text{min\_weight}
$$

2) **PU-aware neutral $p_T$ threshold**

$$
 w_i = 0 \quad \text{if } i \text{ is neutral and } w_i\,p_{T,i} < p_T^{\text{neutral}}(N_{\text{PU}})
$$

with

$$
 p_T^{\text{neutral}}(N_{\text{PU}}) = \text{min\_neutral\_pt} +
 \text{min\_neutral\_pt\_slope} \cdot N_{\text{PU}}.
$$

The PU proxy $N_{\text{PU}}$ is either user-provided (`n_pu_proxy`) or
estimated as

$$
 N_{\text{PU}} \approx \frac{N_{\text{PU-tracks, central}}}{30}
$$

using the count of PU-labeled tracks within $|\eta| < \eta_{\max}$ and
$p_T > p_T^{\text{min}}$.

## 6) Hyperparameters

Defaults in `compute_puppi_weights_charged_subtract()`:

- `R0 = 0.139`
- `rms_pt_min = 0.080`
- `min_neutral_pt = 0.504`
- `min_neutral_pt_slope = 1.837`
- `min_weight = 0.051`
- `eta_max_extrap = 2.543`
- `apply_lv_adjust = False`
- `subtract_pu_charged = True`
- `n_pu_proxy = None`

These match the `puppi_perfect` v2 Optuna-tuned operating point. The
PU-charged subtraction is an algorithmic change rather than a hyperparameter
retune; standalone re-tuning lives in
[optuna_puppi_charged_subtract.py](optuna_puppi_charged_subtract.py) and
optimizes the same combined objective:

$$
 \mathcal{L} = |\text{bias}| + \text{IQR} + 0.5 \cdot |\text{nc}_{\text{rel}}|
$$

where the metrics come from jet $\Delta p_T/p_T$ and the relative
constituent-count difference.

**Interpretation of key hyperparameters** (same role as PUPPI-Perfect):

- **$R_0$ (cone size)**: smaller than the CMS default, better matches the
  dense cluster environment and focuses on the HS jet core.
- **`min_neutral_pt + min_neutral_pt_slope * N_PU`**: PU-aware neutral
  threshold; makes neutral selection stricter as PU increases.
- **`apply_lv_adjust=False`**: the CMS low-PU adjustment worsens bias in
  ODD cluster space and is disabled.
- **`eta_max_extrap`**: extends calibration in $\eta$ for the forward region.
- **`subtract_pu_charged`**: master switch for the PU-charged subtraction. Set
  to `False` to recover `puppi_perfect` exactly.

## 7) Jet Clustering Procedure

Jets are clustered from **both tracks and clusters**, using weighted $p_T$:

$$
 p_{T,i}^{\text{weighted}} = w_i \cdot p_{T,i}^{\text{node}}
$$

Selection mask per event:

$$
 \text{selectable}_i = \text{node\_valid}_i \wedge \text{finite}(p_{T,i}^{\text{weighted}})
 \wedge \text{finite}(\eta_i) \wedge \text{finite}(\phi_i) \wedge (p_{T,i}^{\text{weighted}} > 0)
$$

Tracks are **not filtered out** by type; PU tracks drop out because
$w_i = 0$ for PU-labeled tracks.

Clustering is done event-by-event with `_cluster_jets_single(...)` from
[hepattn/experiments/odd_pileup_reco/reco_analysis.py](reco_analysis.py),
using:

- anti-$k_t$ radius `jet_R` (default 0.7)
- `min_constituents` (default 3)
- `min_pt` (default 10 GeV)

Output keys (compatible with the standard PUPPI jet interface):

- `puppi_jet_pt`, `puppi_jet_eta`, `puppi_jet_phi`, `puppi_jet_mass`,
  `puppi_jet_nconst`

## 8) Truth Jets

The truth target is built from particles with `vertex_primary == 1` (HS only)
using `cluster_truth_hs_jets_from_events(...)`, with identical fastjet
parameters (`jet_R`, `min_constituents`, `min_pt`). Output keys:

- `truth_jet_pt`, `truth_jet_eta`, `truth_jet_phi`, `truth_jet_mass`,
  `truth_jet_nconst`

These are the targets the PUPPI jets are compared against in the
$\Delta p_T/p_T$ resolution plots and Optuna objective.

## 9) Practical Interpretation for Results

This baseline answers a sharper question than PUPPI-Perfect:

*If we had perfect track-cluster matching **and perfect charged-particle
subtraction (HS + PU)**, how well does PUPPI's $\alpha$-shape suppression
discriminate neutral HS from neutral PU?*

Because the input cluster residual is essentially neutral-only at truth level,
this isolates the **neutral suppression power** of PUPPI from any charged
mis-association noise. It should be read as a tighter **upper bound** than
PUPPI-Perfect, and a fair stress test for learned neutral-suppression
methods.

---

### Quick Pseudocode

```python
# Inputs: events dict with tracks, clusters, truth HS+PU charged deps
for each cluster c:
    hs_c = sum(deps_e where has_track and vertex_primary == 1)
    pu_c = sum(deps_e where has_track and vertex_primary != 1)
    cluster_e[c] = max(cluster_e[c] - hs_c - pu_c, 0)

node_pt = track_pt if track else cluster_e / cosh(eta)
weights = compute_puppi_weights(data with modified node_e)
weighted_pt = weights * node_pt
puppi_jets = anti_kt(weighted_pt, eta, phi)

# Truth target
truth_jets = anti_kt(hs_part_pt, hs_part_eta, hs_part_phi)
```
