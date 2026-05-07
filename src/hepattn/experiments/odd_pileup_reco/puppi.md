# PUPPI for ODD pileup-reco

A standalone, ML-free pileup-mitigation baseline for the ODD pileup-reco pipeline.
Implemented in [puppi.py](puppi.py); plotted alongside PFlow / Calo / Calo-HS in
`plot_jet_resolution_with_calo` ([reco_analysis.py:2040](reco_analysis.py#L2040)).

## Goal

Provide a classical pileup-mitigation comparison curve in the jet-resolution
plots, mirroring CMS PUPPI ([CMSSW PileupAlgos](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos))
adapted for ODD's calo-cluster space.

## Fairness contract

What's "fair" here means: matches the information available to a real
reconstruction algorithm running online.

| Truth source                                       | Allowed | Why                                                                                                                    |
|----------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------|
| `tracks_mask` (per-track HS-vs-PU vertex label)    | ✓       | Analog of CMS's CHS Δz cut: real reconstruction associates each track with a primary vertex via Δz to the LV.          |
| `node_pt`, `node_eta`, `node_phi` (track / cluster kinematics) | ✓ | Reconstructed quantities; not truth.                                                                                   |
| `node_e` (cluster energy)                          | ✓       | Calo measurement; not truth.                                                                                           |
| `calo_hs_energy`, `calo_hs_frac`, `calo_neutral_e`, `calo_charged_e` | ✗ | These tell us per cluster how much energy is HS vs PU — exactly what PUPPI is supposed to estimate. Using them is what `cluster_calo_hs_jets` does. |
| `truth_incidence`, `object_class`                  | ✗       | Per-particle / per-cluster truth labels.                                                                               |

PUPPI weight outputs *are* used to weight cluster pT before jet clustering — that's the whole point.

Note: tuning hyperparameters by sweeping on the eval set is "test-set tuning"
and is documented separately below.

## Algorithm

For each event, classify nodes:

- **LV-charged track** (`node_is_track=1` & `tracks_mask=1`) → final weight = **1**
- **PU-charged track** (`node_is_track=1` & `tracks_mask=0`) → final weight = **0** (CHS-style veto)
- **Neutral cluster** (`node_is_track=0`) → weight from α-shape (below)

For each non-LV-track node *i* (i.e. PU tracks for calibration, neutrals for scoring),
compute the CMS-PUPPI shape variable (algoId=5):

α_i = log Σ_{j ∈ LV-tracks, ΔR_ij ∈ (0, R0)} pT_j² / ΔR_ij²

ΔR is computed in (η, φ) with proper φ-wrap. Pairs with ΔR² < 1e-4 are
excluded (CMS [PuppiContainer.cc:89](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos/src/PuppiContainer.cc#L89)).
A particle with no LV neighbors inside R0 gets α=0 and is dropped from the
calibration sample.

Calibrate α statistics on PU tracks (`tracks_mask=0`) within `|η| < 2.0`
and `pT > 0.1 GeV`:

- α_med = median of {α_i : i ∈ PU tracks, α_i ≠ 0}
- α_rms = √mean((α_i − α_med)² for α_i ≤ α_med)  ← CMS `applyLowPUCorr=True` (low-PU symmetric estimator, [PuppiAlgo.cc:138](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos/src/PuppiAlgo.cc#L138))

Apply CMS LV-adjust shift ([PuppiAlgo.cc:159](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos/src/PuppiAlgo.cc#L159)):
let `f = N_LV_below_med / (N_LV_below_med + 0.5·N_PU_cal)`. If 0 < f < 1,
shift α_med ← α_med − √(χ²_quantile(f, df=1) · α_rms) and
α_rms ← α_rms − √(χ²_quantile(f, df=1) · α_rms).

Per-neutral weight: signed χ² following CMS [PuppiAlgo.cc:206](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos/src/PuppiAlgo.cc#L206):

L_i = (α_i − α_med) · |α_i − α_med| / α_rms²
w_i = χ²_cdf(max(L_i, 0), df=1)

Negative L (α below median) → weight = 0. The CDF maps positive L_i to (0, 1),
peaking at 1 as α_i grows above the PU median.

Apply category overrides: w[LV] = 1, w[PU] = 0, w[invalid] = 0.

Apply cuts:

- `w < min_weight` → 0
- For neutrals only: `w · pT < min_neutral_pt + min_neutral_pt_slope · n_pu_proxy` → 0
  (CMS PuppiAlgo::neutralPt, [PuppiContainer.cc:319](file:///storage/agrp/barakma/cmssw/CommonTools/PileupAlgos/src/PuppiContainer.cc#L319))

## Hyperparameters

Final defaults in [`compute_puppi_weights`](puppi.py):

| Param                       | Value | CMS phase2 | Why deviate                                                                                              |
|-----------------------------|-------|------------|----------------------------------------------------------------------------------------------------------|
| `R0` (cone)                 | 0.2   | 0.4        | ODD operates in cluster space (~30× more clusters per PU vertex than CMS PFlow has candidates per vertex). A tighter cone exploits HS clusters sitting in jet cores (dense LV neighbors at ΔR<0.2) while accidental PU-near-LV doesn't have that local density. |
| `rms_pt_min`                | 0.1 GeV | 0.1 GeV  | Same.                                                                                                    |
| `min_neutral_pt`            | 0.3 GeV | 0.2 GeV  | Tuned on this dataset (events 0..200) for nconst-distribution shape match to truth. Verified to generalize on events 200..2000. |
| `min_neutral_pt_slope`      | 0.0   | 0.015      | Slope only kicks in if a PU proxy is supplied; default proxy is 0 (flat threshold).                      |
| `n_pu_proxy`                | 0.0   | NPU vertices | The infrastructure exists to plug in a real per-event PU proxy if you have one. Default 0 makes the threshold flat. |
| `min_weight`                | 0.01  | 0.01       | Hard floor: drop weights below 1%.                                                                       |
| `eta_max_extrap`            | 2.0   | 2.0        | Same.                                                                                                    |
| `apply_lv_adjust`           | True  | True       | Same.                                                                                                    |

## Cluster-only jets (no double-counting)

`cluster_puppi_jets` clusters jets from **weighted clusters only** — tracks are
NOT added to the jet inputs. CMS PUPPI consumes PFlow particles, where the
charged-cluster-track linking has already happened, so charged HS energy is
counted exactly once. We don't have PFlow linking here, so adding tracks to
the jet inputs would double-count any charged HS particle (once via its track,
once via its cluster deposit). Tracks therefore only contribute as α-seeds.

This makes our PUPPI structurally a "calo-with-PUPPI-weights" baseline —
directly comparable to `cluster_calo_jets` (raw calo) and `cluster_calo_hs_jets`
(truth-cheating calo with HS-only energy).

## Required H5 fields

Loaded by `load_pflow_data` from `node_metadata`:

`node_valid`, `node_is_track`, `tracks_mask`, `node_pt`, `node_eta`, `node_phi`, `node_e`

If `tracks_mask` is missing, `compute_puppi_weights` returns `None` and the
`PUPPI` overlay is silently skipped from `plot_jet_resolution_with_calo`.

## Performance (held-out generalization)

Tuning was done on events 0..200 of
[`epoch=099-val_loss=13.99981__test_latest_256_dim.h5`](logs/odd_pflow_reco_20260421-T104425/ckpts/epoch=099-val_loss=13.99981__test_latest_256_dim.h5).
The remaining 1800 events were never touched during tuning.

| Sample             | nJets/ev | nconst med/p99 | pT med/p99 | bias    | IQR   |
|--------------------|----------|----------------|------------|---------|-------|
| TUNE (0..200)      | 6.24     | 16/59          | 52/272     | +0.062  | 0.401 |
| HELD-OUT (200..2000) | 6.37   | 16/58          | 49/284     | +0.047  | 0.374 |

For reference on the held-out 1800 events:

| Method   | nJets/ev | nconst med/p99 | pT med/p99 | bias    | IQR   |
|----------|----------|----------------|------------|---------|-------|
| truth    | 7.00     | 16/62          | 45/331     | —       | —     |
| pflow    | 6.40     | 14/56          | 46/340     | -0.018  | 0.216 |
| calo-HS  | 6.38     | 35/127         | 36/260     | -0.246  | 0.159 |
| **PUPPI**| **6.37** | **16/58**      | **49/284** | **+0.047** | **0.374** |
| calo     | (~45)    | 85/200         | ~110/280   | +2.75   | 2.6   |

PUPPI:

- bias near zero (+0.05) — comparable to PFlow's −0.02
- IQR 0.37 — between PFlow (0.22) and ~2× Calo-HS's truth-cheating 0.16; ~7× tighter than raw Calo
- nconst median exactly matches truth (16); shape close to PFlow
- nJets/event matches Calo-HS and PFlow

## Where it plugs in

- [puppi.py](puppi.py) — algorithm + jet clustering (this module)
- [reco_analysis.py:52–256](reco_analysis.py#L52) — `load_pflow_data` extended to load `tracks_mask` from `node_metadata`
- [reco_analysis.py:2040](reco_analysis.py#L2040) — `plot_jet_resolution_with_calo` calls `cluster_puppi_jets` and overlays a `PUPPI` curve (`linestyle="-."`) in every panel of the 3×2 jet-resolution figure, alongside PFlow, Proxy, Calo, Calo-HS

## How to run

End-to-end on the full H5:

```
python -m hepattn.experiments.odd_pileup_reco.run_puppi_jet_resolution
```

(see [run_puppi_jet_resolution.py](run_puppi_jet_resolution.py) for `--h5`,
`--out`, `--event-start/--event-stop` and other flags.)

Sanity-check the per-node weights only:

```
python -m hepattn.experiments.odd_pileup_reco.puppi <H5> --max-events 20
```

prints mean weight by truth category (LV tracks ≈ 1.0, PU tracks ≈ 0.0,
neutrals near LV > neutrals far from LV), plus `α_med` / `α_rms` sanity values.

## Limitations / what could improve it further

- **No η-binning** of the α calibration. CMS uses separate central / forward bins (`puppiCentral`, `puppiForward`). For ODD's narrower acceptance this matters less, but a barrel/endcap split would likely tighten the IQR.
- **Test-set tuning of `R0` and `min_neutral_pt`**. A clean cross-validation against an independent calibration sample would give defense-in-depth, but the tune→held-out check above shows the chosen values are stable.
- **No photon / lepton special-casing.** CMS forces high-pT photons and leptons to weight=1; we don't have node-level PID, so this protection isn't applied. For HS jets dominated by photons (e.g. H→γγ), this would matter.
- **No PFlow linking** — clusters carry full energy including charged-particle deposits. Mitigated by clustering jets from clusters only (not tracks), but a real PFlow-style reconstruction (matching tracks to clusters and subtracting) could feed cleaner inputs to PUPPI and would close part of the IQR gap to Calo-HS.
