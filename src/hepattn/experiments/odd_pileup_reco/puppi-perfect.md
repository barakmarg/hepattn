# PUPPI-Perfect v2: Detailed Algorithm and Math

This note documents the **PUPPI with perfect PFlow linking** baseline implemented in
[hepattn/experiments/odd_pileup_reco/puppi_perfect.py](puppi_perfect.py). This variant
is an **upper-bound** baseline: it uses truth information to remove double-counting
between tracks and calorimeter clusters while still applying PUPPI alpha-shape
weights for neutral suppression.

## 1) Goal and High-Level Idea

We want jet inputs that behave like *ideal* PFlow:

- **LV (hard-scatter) tracks** are kept at full $p_T$ with weight 1.
- **PU tracks** are removed (weight 0).
- **Clusters** contribute only their **neutral remainder** energy (subtracting
  truth HS-charged energy deposit), then get a PUPPI neutral weight.

This emulates **perfect track-cluster matching** and **perfect charged subtraction**.
It is intentionally unfair (truth-assisted) and represents a ceiling for PUPPI-like
performance.

## 2) Inputs and Truth Information Used

Required per-node arrays from the evaluation dict `data`:

- `node_is_track`: boolean, track vs cluster.
- `node_pt`, `node_eta`, `node_phi`, `node_e`: kinematics.
- `calo_charged_e` (truth): per cluster energy deposited by **HS-charged** particles.
- `tracks_mask` (truth): HS vs PU label for tracks (used in `compute_puppi_weights`).
- `node_valid`: validity mask for nodes.

**Truth usage summary**

1. **HS/PU track labels** via `tracks_mask` (in `compute_puppi_weights`).
2. **HS-charged calorimeter deposit** via `calo_charged_e` to compute a
   neutral-remainder energy per cluster.

This combination is stronger than the usual truth-aware PUPPI that only uses
`tracks_mask`.

## 3) Neutral Remainder Construction (Perfect PFlow Linking)

For every node $i$:

- If it is a **track**, we keep the track transverse momentum:

$$
 p_{T,i}^{\text{node}} = p_{T,i}^{\text{track}}
$$

- If it is a **cluster**, we subtract the truth HS-charged energy deposit and
  convert the remaining energy to transverse momentum:

$$
 E_{i}^{\text{neutral}} = \max(E_i - E^{\text{HS-charged}}_i, 0)
$$

This is a direct subtraction of the HS-charged energy deposit per cluster
(no rescaling or learned correction); any negative remainder is floored to 0.

$$
 p_{T,i}^{\text{cluster}} = \frac{E_{i}^{\text{neutral}}}{\cosh(\eta_i)}
$$

So the combined per-node pT used for jet inputs is:

$$
 p_{T,i}^{\text{node}} =
  \begin{cases}
    p_{T,i}^{\text{track}} & \text{if } i \text{ is track} \\
    E_{i}^{\text{neutral}}/\cosh(\eta_i) & \text{if } i \text{ is cluster}
  \end{cases}
$$

This is implemented in `_node_pt_neutral_remainder()`.

**Why this matters**: adding LV tracks to jets would double-count their
calorimeter energy unless the charged deposit is removed. The neutral remainder
avoids that by construction.

## 4) PUPPI Weights (Alpha-Shape) With Explicit Math

Weights are computed by reusing `compute_puppi_weights()` from
[hepattn/experiments/odd_pileup_reco/puppi.py](puppi.py), **but with a modified
cluster energy**:

$$
 E_i \leftarrow \begin{cases}
   E_i & \text{track} \\
   \max(E_i - E^{\text{HS-charged}}_i, 0) & \text{cluster}
 \end{cases}
$$

This ensures that any cluster $p_T$ used inside PUPPI is the **neutral remainder**.

### 4.1 Alpha Definition

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
blowups. Targets include **PU tracks**, **neutral clusters**, and **LV tracks**
(the last only for the low-PU adjustment).

If a target has no valid neighbors inside $R_0$, then $\alpha_i = 0$.

### 4.2 Calibration (Median and RMS)

The calibration sample is the set of **PU tracks** with

$$
 |\eta| < \eta_{\max} \quad \text{and} \quad p_T > p_T^{\text{min}}
$$

where $p_T^{\text{min}} = \text{rms\_pt\_min}$ and $\eta_{\max} = \text{eta\_max\_extrap}$.
Let $\{\alpha_k\}$ be this sample (excluding zeros). Then

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

### 4.3 Low-PU LV Adjustment (Optional)

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

### 4.4 Weight Mapping (Signed $\chi^2$)

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

where $F_{\chi^2_1}$ is the CDF of $\chi^2$ with 1 d.o.f. This is the CMS
"signed $\chi^2$" mapping. Category overrides then apply:

$$
 w_i = 1 \;\text{for LV tracks},\quad w_i = 0 \;\text{for PU tracks}.
$$

Finally, two hard cuts are enforced:

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
 	ext{min\_neutral\_pt\_slope} \cdot N_{\text{PU}}.
$$

The PU proxy $N_{\text{PU}}$ is either user-provided or estimated as

$$
 N_{\text{PU}} \approx \frac{N_{\text{PU-tracks, central}}}{30}
$$

using the count of PU-labeled tracks within $|\eta| < \eta_{\max}$ and
$p_T > p_T^{\text{min}}$.

## 5) Hyperparameters (v2 Tuned Operating Point)

Default values in `compute_puppi_weights_perfect()`:

- `R0 = 0.139`
- `rms_pt_min = 0.080`
- `min_neutral_pt = 0.504`
- `min_neutral_pt_slope = 1.837`
- `min_weight = 0.051`
- `eta_max_extrap = 2.543`
- `apply_lv_adjust = False`

These defaults are an Optuna-tuned operating point (v2) optimized on ODD
cluster space for a combined objective:

$$
 \mathcal{L} = |\text{bias}| + \text{IQR} + 0.5 \cdot |\text{nc}_{\text{rel}}|
$$

where the metrics are computed from jet $\Delta p_T/p_T$ and the relative
constituent count difference. The tuned values improve performance in the
ODD cluster representation, which has many more nodes per PU vertex than
CMS PFlow.

**Interpretation of key hyperparameters**

- **$R_0$ (cone size)**: smaller than CMS default, better matches the dense
  cluster environment and focuses on the HS jet core.
- **`min_neutral_pt + min_neutral_pt_slope * N_PU`**: PU-aware neutral threshold;
  makes neutral selection stricter as PU increases.
- **`apply_lv_adjust=False`**: the CMS low-PU adjustment worsens bias in ODD
  cluster space and is disabled.
- **`eta_max_extrap`**: extends calibration in $\eta$ for forward region.

## 6) Jet Clustering Procedure

Jets are clustered from **both tracks and clusters**, using weighted $p_T$:

$$
 p_{T,i}^{\text{weighted}} = w_i \cdot p_{T,i}^{\text{node}}
$$

Selection mask:

$$
 \text{selectable}_i = \text{node\_valid}_i \wedge \text{finite}(p_{T,i})
 \wedge (p_{T,i}^{\text{weighted}} > 0)
$$

Tracks are **not filtered out** by type; PU tracks drop out because
$w_i = 0$ for PU-labeled tracks.

Clustering is done event-by-event using `_cluster_jets_single(...)` from
[hepattn/experiments/odd_pileup_reco/reco_analysis.py](reco_analysis.py), with:

- anti-$k_t$ radius `jet_R` (default 0.7)
- `min_constituents` (default 3)
- `min_pt` (default 10 GeV)

Outputs are stored in:

- `puppi_jet_pt`, `puppi_jet_eta`, `puppi_jet_phi`, `puppi_jet_mass`,
  `puppi_jet_nconst`

These keys match the standard PUPPI jet interface and can be passed into
existing jet-resolution plotting functions.

## 7) Practical Interpretation for Results

This baseline answers: *If we had perfect track-cluster matching and perfect
charged subtraction, how far can PUPPI go using only alpha-shape suppression
for neutrals?*

Because it uses truth for both track labels and charged deposition, it should
be seen as a **ceiling** rather than a fair comparison to learned models.

---

### Quick Pseudocode

```python
# Inputs: data dict with tracks, clusters, truth calo_charged_e
node_pt = track_pt if track else (node_e - calo_charged_e)/cosh(eta)
node_e  = node_e if track else max(node_e - calo_charged_e, 0)
weights = compute_puppi_weights(data with modified node_e)
weighted_pt = weights * node_pt
jets = anti_kt(weighted_pt, eta, phi)
```
