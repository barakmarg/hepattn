# PUPPI (Charged-Subtract) — Hyperparameter Optimisation

This note documents the Optuna-driven hyperparameter optimisation procedure
used to tune the **PUPPI charged-subtract** algorithm
(see [`puppi-charged-subtract.md`](puppi-charged-subtract.md) for the
algorithmic definition). Implementation:
[`optuna_puppi_charged_subtract.py`](optuna_puppi_charged_subtract.py).

The intent is to provide a self-contained reference suitable for a paper
methods section.

## 1) Why hyperparameter search is needed

PUPPI has seven free hyperparameters that control:

- The neighbourhood used for the α-shape (`R0`).
- The PU-track calibration sample (`rms_pt_min`, `eta_max_extrap`).
- A hard floor on the final per-particle weight (`min_weight`).
- A pile-up-aware neutral-pT threshold (`min_neutral_pt`,
  `min_neutral_pt_slope`).
- An optional CMS low-PU correction toggle (`apply_lv_adjust`).

The default CMS values were calibrated against the CMS Phase-2 PFlow
ensemble at PU=200. The ODD geometry, cluster granularity and pile-up
density differ enough that the CMS defaults are noticeably sub-optimal in
the cluster-space charged-subtract regime (where the neutral residual is
strongly suppressed by the explicit HS+PU charged deposit subtraction).
A dataset-specific search is therefore necessary for a fair comparison.

## 2) Objective

For every parameter set $\theta$ we cluster the resulting PUPPI-weighted
nodes into jets with FastJet (anti-kT, $R$ configurable), Hungarian-match
to the truth-HS jet collection within $\Delta R < \text{dr\_cut}$, and
form per-pair pT residuals

$$
 d_k(\theta) = \frac{p_T^{\text{PUPPI},k}(\theta) - p_T^{\text{truth},k}}{p_T^{\text{truth},k}}
$$

aggregated across all matched pairs in the event sample. Three scalar
metrics are extracted:

$$
 \text{bias}(\theta)   = \operatorname{median}\bigl(\{d_k(\theta)\}\bigr)
$$

$$
 \text{IQR}(\theta)    = \operatorname{IQR}\bigl(\{d_k(\theta)\}\bigr)
$$

$$
 \text{nc\_rel}(\theta) =
   \langle
   \tfrac{\text{nconst}^{\text{PUPPI}}_k - \text{nconst}^{\text{truth}}_k}{\max(\text{nconst}^{\text{truth}}_k,\,1)}
   \rangle_k
$$

The optimisation target is the scalar

$$
 \mathcal{L}(\theta) = |\text{bias}(\theta)| + \text{IQR}(\theta) + 0.5 \cdot |\text{nc\_rel}(\theta)|
$$

This combines closure (bias), resolution (IQR), and constituent-count
fidelity. Median/IQR are used instead of mean/std because the residual
distribution has heavy tails — a few badly-mismatched jets can dominate
mean/std and mislead the optimiser. The `0.5` weight on `nc_rel` was
selected empirically to keep all three terms in the same numerical
neighbourhood at the optimum (≈$0.01$–$0.3$ each), so no single term
dominates the gradient signal seen by TPE.

If fewer than 5 matched jets exist for a given trial, the trial returns a
sentinel value $10^6$ so TPE can prune the region quickly.

## 3) Search space

| Parameter              | Range          | Scale | Rationale |
|------------------------|----------------|-------|-----------|
| `R0`                   | $[0.10, 0.40]$ | lin   | α-shape cone size. CMS default 0.4. ODD cluster density wants smaller. |
| `rms_pt_min`           | $[0.02, 1.0]$  | log   | Minimum track pT entering the PU calibration sample. Log scale because the optimum is order-of-magnitude uncertain. |
| `min_neutral_pt`       | $[0.05, 3.0]$  | log   | Floor on PU-aware neutral threshold. After full charged subtraction the residual is soft, so the lower bound is pulled below CMS. |
| `min_neutral_pt_slope` | $[0.0, 0.5]$   | lin   | Per-PU-vertex slope of the neutral pT floor. CMS default $\sim 0.015$; we widen to allow stronger PU scaling. |
| `min_weight`           | $[0.0, 0.5]$   | lin   | Hard cut: any neutral with PUPPI weight below this is zeroed. |
| `eta_max_extrap`       | $[1.0, 3.5]$   | lin   | Central-η cut for both the PU calibration sample and the LV-adjust sample. Widened from $[1.5, 3.0]$ so the on/off LV-adjust comparison has enough room to find its joint optimum (see §5). |
| `apply_lv_adjust`      | `{True, False}`| cat   | CMS low-PU symmetric-RMS correction (`PuppiAlgo.cc:159`). Functionally entangled with `eta_max_extrap`. |

Conceptually independent: `subtract_pu_charged` is **fixed** to `True`
during the search; the algorithm is being tuned, not ablated. Setting it
to `False` recovers the puppi_perfect operating point and is reserved for
the sanity-check flag `--no-subtract-pu`.

## 4) Sampler

We use **Optuna's TPE sampler** (Tree-structured Parzen Estimator) with
`seed=42` for reproducibility. TPE is well-suited here because:

- The search space mixes continuous, log-continuous and categorical
  variables.
- The objective is non-differentiable (passes through Hungarian matching
  and the FastJet binary).
- The dimensions exhibit non-trivial pairwise interactions
  (`apply_lv_adjust` × `eta_max_extrap`; `min_neutral_pt` ×
  `min_neutral_pt_slope`; `R0` × `rms_pt_min`). TPE natively models
  joint posteriors and concentrates samples in promising sub-volumes.

A persistent SQLite backend (`<study_name>.db`) is used so studies can be
resumed or extended.

## 5) Two-stage selection (search + held-out validation)

A naive "best of the search" is fragile when only $N_{\text{search}}\sim 100$
events are used per trial: the variance of the objective over event
samples is comparable to the trial-to-trial improvement near the optimum.
The "best" trial in the search can therefore be one that got lucky on the
small search set.

We mitigate this with a **two-stage selection**:

1. **Search**: $N_{\text{trials}}$ trials are run on the first
   $N_{\text{search}}$ events (event ids $0..N_{\text{search}}-1$).
2. **Validation**: the top $K$ trials (lowest $\mathcal{L}$ during search)
   are re-evaluated on a disjoint set of $N_{\text{val}}$ events
   (event ids $N_{\text{search}}..N_{\text{search}}+N_{\text{val}}-1$).
3. **Selection**: the parameter set with the lowest $\mathcal{L}$ on the
   *validation* set is the final output.

This is a standard "model selection" pattern adapted from ML practice
(train→val) and is appropriate here because the search budget is small
relative to per-event variance. The published numbers are then the
validation metrics of the selected parameter set, evaluated on
$N_{\text{val}}$ statistically independent events.

The validation cost is modest: $K$ evaluations × $(N_{\text{val}}/N_{\text{search}})$
≈ $10 \times 10 = 100$ search-trial-equivalents, i.e. ~ half the search
budget for $K=10$, $N_{\text{val}}=1000$, $N_{\text{search}}=100$.

## 6) Defaults

| Knob                  | Default | Meaning |
|-----------------------|---------|---------|
| `--n-trials`          | 200     | Search trials. |
| `--n-events`          | 100     | Events used per search trial. |
| `--n-validation-events`| 1000   | Events used per validation evaluation (disjoint from search). |
| `--top-k`             | 10      | Top trials taken from search to validation. |
| `--jet-R`             | 0.7     | Anti-kT $R$. Use 0.4 for LHC-standard jets. |
| `--min-constituents`  | 3       | Per-jet constituent floor. |
| `--min-pt`            | 10.0 GeV| Per-jet pT floor. |
| `--dr-cut`            | 0.4     | Hungarian matching radius. |
| `--seed`              | 42      | TPE sampler seed. |
| `--jet-algorithm`     | antikt  | FastJet algorithm; `kt` available for legacy comparison. |

## 7) Output

After the search and the validation pass, the script writes a JSON file
`<out_dir>/<study_name>_best.json` with the following fields:

```jsonc
{
  // Validation-selected best (this is what the analysis uses)
  "best_params":            { /* the 7 PUPPI hyperparameters */ },
  "best_value":             /* L on validation set */,
  "best_bias":              /* bias on validation */,
  "best_iqr":               /* IQR on validation */,
  "best_nc_rel":            /* nc_rel on validation */,
  "best_n_matched":         /* # matched jets in validation */,

  // Provenance: which search trial was selected
  "selected_from_trial":    /* optuna trial index */,
  "selected_search_value":  /* its L on the search set */,

  // Search summary (top-1 by search L — typically differs from selected)
  "search_best_value":      /* L of the top-1 search trial */,
  "search_best_params":     { /* its params */ },

  // Run configuration (for reproducibility)
  "n_trials":               200,
  "n_events":               100,
  "n_validation_events":    1000,
  "top_k":                  10,
  "subtract_pu_charged":    true,
  "jet_algorithm":          "antikt",
  "jet_R":                  0.7,

  // Full top-K table: search vs validation, for transparency
  "top_k_validation":       [ {
      "rank": 1, "trial_number": ..., "params": { ... },
      "search_value": ..., "search_bias": ..., "search_iqr": ..., "search_nc_rel": ...,
      "val_value": ..., "val_bias": ..., "val_iqr": ..., "val_nc_rel": ..., "val_n_matched": ...
  }, ... ]
}
```

The downstream resolution-plot scripts (e.g.
`run_puppi_jet_resolution_charged_subtract_full.py`) load this JSON via
`--best-json` and pull `best_params` to configure PUPPI.

## 8) Reproducibility

Anti-kT R=0.4 reference run (LHC-standard jets):

```bash
python -m hepattn.experiments.odd_pileup_reco.optuna_puppi_charged_subtract \
  --jet-algorithm antikt \
  --jet-R 0.4 \
  --n-events 100 \
  --n-trials 200 \
  --top-k 10 \
  --n-validation-events 1000 \
  --study-name puppi_charged_subtract_v2_antikt_R04
```

All randomness is controlled by `--seed`. The parquet dataset
(`ttbar_pu200_all_vertices_chunked`) is read-only and shard-ordered, so
the same `(event_start, event_stop)` slices always return the same events.
Re-running the command above with an unchanged DB will resume the study
where it stopped.

## 9) Summary (paragraph-form, for paper text)

> The PUPPI weight algorithm has seven free hyperparameters whose CMS
> defaults are not optimal for the ODD detector geometry in the
> cluster-space charged-subtract regime. We optimise them with the
> Tree-structured Parzen Estimator (Optuna) on a combined objective
> $\mathcal{L} = |\text{bias}| + \text{IQR} + 0.5 \cdot |\text{nc\_rel}|$
> evaluated on Δp_T/p_T of Hungarian-matched truth-HS jets. The search
> uses 200 trials on 100 events; the top 10 trials are then re-evaluated
> on 1000 disjoint validation events and the parameter set with the best
> validation score is taken as the final operating point. This held-out
> selection step suppresses the trial-to-trial sampling noise that would
> otherwise contaminate a small-sample "best of search" choice. The
> reported numbers are validation-set metrics for the selected parameter
> set, and the optimisation procedure is fully reproducible from a
> single command and a fixed seed.
