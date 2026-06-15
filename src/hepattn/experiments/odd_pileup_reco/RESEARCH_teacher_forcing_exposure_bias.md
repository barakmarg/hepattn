# Closing the Stream-C Exposure-Bias Gap in `odd_pileup_reco`

A complete record of the investigation into why the model's **validation loss (~14)** sat far
above its **training loss (~10.2)**, the root-cause diagnosis, and the distribution-matched
teacher-forcing fix that was implemented and verified.

---

## 1. The symptom

- Training loss: steadily decreasing, ≈ **10.2**.
- Validation loss: bouncing, ≈ **13–14**, not converging toward train.
- Suspicion: the train/inference mismatch in how Stream C (reconstruction) is fed — i.e. the
  reconstruction network sees **out-of-distribution inputs at inference** relative to training.

The model is a three-stream MaskFormer (see `readme_odd_reco.md`):
- **Stream A** — which tracks are hard-scatter (HS).
- **Stream B** — which calo clusters are HS (pileup removal).
- **Stream C** — reconstruct particles from the *selected* HS nodes (class + mask + incidence + kinematics).

Stream C's node selection is gated by `self.training and self.teacher_forcing`
(`model.py`, `forward` / `_forward_reco`):
- **Training (teacher forcing):** truth HS masks + a fixed number of sampled false-positive pileup nodes.
- **Validation / inference:** Stream A/B *predicted* masks, then top-1200 calo by energy.

Because the two paths feed Stream C **different input distributions**, `train/loss` and `val/loss`
are not directly comparable — the gap could be exposure bias rather than overfitting.

---

## 2. Tooling built (all in `eval_data.py`, driven by `run_forward_pass.py`)

| Function | Purpose |
|---|---|
| `compare_tf_vs_pred_loss` | Run the same held-out events through the model twice (teacher-forcing vs prediction path) and report per-stream + per-reco-task losses, plus per-event input diagnostics (occupancy, purity, recall). |
| `analyze_tf_mask_strategies` | First-pass overlay of a few candidate TF selections vs the inference target. |
| `sweep_tf_mask_schemes` | Sweep many combined **FP+FN** teacher-forcing schemes; score each against the `pred` distribution (Wasserstein on per-event metrics + Jensen–Shannon on energy spectra); rank and plot. |
| `diagnose_track_class_attribution` | **Factorial track/calo ablation** of the classification gap (Idea 1). |
| `verify_tf_selection` | Verify the model's *actual* in-model TF selection (the new config recipe) reproduces the `pred` distribution. |

Enabling change in `model.py::TwoStreamMaskFormer.forward` / `_forward_reco`: an optional
`use_teacher_forcing` override (and independent `track_tf` / `calo_tf` overrides) that force either
conditioning path regardless of train/eval mode. Behavior is **byte-identical** when the overrides
are `None`.

Data-module fix discovered along the way: `ODDDataModule` silently mishandled an explicit
`files_list` (it leaked into `dataset_kwargs` and collided in `_make_dataset`). Made `files_list`
a first-class constructor parameter that the stages fall back to when not splitting.

---

## 3. Diagnosis: the gap is real exposure bias, and it's the classification head

### 3.1 TF vs pred, on identical events (checkpoint `epoch=073-val_loss=13.679`)

```
stream            pred path      TF path        delta
track                0.959        0.959        0.000
calo                 2.437        2.329        0.108
reco                10.382        7.402        2.980
TOTAL               13.777       10.690        3.087
```

- **pred total ≈ 13.78** reproduces the logged `val_loss` → the harness is correct.
- **TF total ≈ 10.69** reproduces the training loss → the model is fine *when fed training-like inputs*.
- The entire gap (**+3.09**) lives in **Stream C reconstruction**; Stream A is identical (it doesn't
  depend on the conditioning) and Stream B barely moves.

### 3.2 Which reco objective breaks — sub-task breakdown

```
reco task        pred path    TF path      delta
classification      5.330       2.278      +3.05
regression          0.355       0.315      +0.04
incidence           0.006       0.007       0.00
mask                4.690       4.802      -0.11
```

The gap is **almost entirely the classification head** (CE roughly doubles). Mask / incidence /
regression are essentially immune.

**Why classification specifically:** mask and incidence are *dense, per-node, averaged* losses —
dropping/altering a few nodes only perturbs them fractionally. Classification is a **single,
global, discrete decision per object** that depends on the object's *complete* hit signature.
The clearest mechanism is **trackless reclassification**: a charged particle with no surviving
track reads as neutral (class 0→3) — so an incomplete/altered hit set flips the categorical
answer and incurs the full CE penalty, while the mask head shrugs it off.

---

## 4. Characterizing the input shift (what's actually different at inference)

Per-event distributions of the Stream-C input, teacher forcing vs prediction path:

| quantity | TF (training) | pred (inference) | meaning |
|---|---|---|---|
| **calo purity** ΣHS/Σtotal | ≈ 0.37 | ≈ 0.37 | FP contamination — **essentially equal** |
| **HS energy recall** | ≈ 0.91 | ≈ 0.82 | training delivers ~9% more HS energy (inference misses some) |
| **occupancy** (valid nodes) | ≈ 458 | ≈ 399 | inference selects **fewer** nodes |
| **selected calo nodes** | ≈ 402 | ≈ 349 | inference is sparser, not dirtier |

Key surprise that redirected the fix: **purity is the same** — the earlier "inference floods Stream C
with high-energy pileup" story was **wrong** on this data. The real shift is **under-selection**:
inference packs a sparser buffer with slightly less HS energy and altered per-particle completeness.
The current teacher forcing only ever *adds* false-positive pileup and **never drops true signal**,
so the class head never learned to name a particle from an incomplete hit set — exactly what
inference hands it.

---

## 5. Factorial track/calo ablation (Idea 1) — is it tracks or calo?

Run 4 reco-selection conditionings (independent `track_tf` / `calo_tf`) and read classification CE:

```
 run   calo  track     cls_CE       mask   reco_tot
   A     TF     TF     2.4622     4.7296     7.5161
   B     TF   pred     2.6392     4.7775     7.7517
   C   pred     TF     5.1715     4.6357    10.1648
   D   pred   pred     5.4004     4.6697    10.4315

classification CE gap (vs A = TF/TF):
  from tracks (B-A): +0.177
  from calo   (C-A): +2.709
  total       (D-A): +2.938   (additivity: tracks+calo = +2.886)
  -> tracks explain 6% of the gap, calo 92%
```

**Conclusion: the classification gap is ~92% the calo selection, ~6% tracks.** Despite the
trackless-reclassification intuition, the dominant driver is the calo node set Stream C receives,
not low-energy track mis-prediction. (`mask` stays flat across A–D — a good consistency check.)

---

## 6. Designing the matched teacher-forcing selection

Premise (validated by Stream B's good purity): the **`pred` selection is the target distribution**;
a good TF mask should reproduce it on **purity, recall, size, and the per-node energy spectra**
(clusters and tracks). The fix must combine **false negatives (drop true HS) + false positives
(add pileup)** — dropping alone keeps purity too high (no contamination), adding alone over-selects.

### 6.1 Scheme sweep (random vs energy-biased, fractions/counts)

- **`energy_low` FN dropping can't reach pred's recall** (stuck ≈ 0.89 — faint clusters carry little
  energy), so **random FN** wins.
- **`fp=energy` worsened the spectrum** (over-populates the high-E tail); **random FP** wins.
- Best single scheme on means: **`fn=random:0.10` + `fp=random:170`**:

```
              purity  recall    occ   n_calo  n_trk   SCORE
pred (target)  0.373   0.825   405.6   354.4   51.2     —
fn0.10/fp170   0.369   0.822   407.0   356.8   50.3   0.120   <- best
truth+FP(cur)  0.367   0.915   464.6   407.9   56.7   0.454   <- current TF (over-selects, recall too high)
```

### 6.2 The width problem and the fix

With a **fixed** FP count the per-event size distribution was far too narrow (sharp peak ~400)
versus pred's broad ~150–800. Cause: real Stream-B FP count **varies per event and correlates with
event activity**; a constant kills that variance. Fix: **per-event FP count `~ N(mean, std)`**.
Sweeping σ at fixed mean=170 (mean-preserving) broadened occupancy/n_calo onto pred; **σ ≈ 100**
matched the spread → final recipe **FP = 170 ± 100**.

### 6.3 Final decided recipe

- **Calo:** drop **10%** of true HS clusters (random FN) + add **170 ± 100** FP pileup clusters
  (random, Gaussian per-event count), then the existing top-1200-by-energy cap.
- **Tracks:** drop **10%** of true HS tracks + add **~10%** proportional FP tracks (small effect — the
  ~6% contributor — but included).

---

## 7. Implementation (config-driven, compile-safe)

All in `model.py::TwoStreamMaskFormer`, applied **only in the teacher-forcing branch** of
`_forward_reco`. New / repurposed `__init__` knobs:

| knob | meaning | default |
|---|---|---|
| `reco_calo_fn_frac` | random fraction of true HS calo dropped | 0.0 |
| `reco_calo_noise_mean` | mean FP calo count | (config) |
| `reco_calo_noise_std` | **repurposed** (was vestigial): Gaussian std of per-event FP calo count | 0.0 |
| `reco_track_fn_frac` | random fraction of true HS tracks dropped | 0.0 |
| `reco_track_fp_frac` | FP tracks added ≈ `frac × (#true HS tracks)` | 0.0 |

### Compile-safety (the model is `torch.compile(dynamic=True)`'d via `callbacks/compile.py`)

The variable per-event count is selected **without any `.item()` / Python-int-from-tensor / dynamic
shape** — a fully-tensorized **sort + per-row gather threshold**:

```python
# calo TF branch (model.py, ~L351-368)
calo_keep = torch.rand(B, N, device=device) >= self.reco_calo_fn_frac        # FN drop
truth_calo_kept = truth_calo_mask & calo_keep

extra_pred = pred_calo_mask & ~truth_calo_mask                                # Stream-B FP pool
calo_scores = torch.where(extra_pred, torch.rand(B, N, device), -inf)
k_calo = (torch.randn(B, device) * self.reco_calo_noise_std                   # per-event count ~ N(mean,std)
          + self.reco_calo_noise_mean).round().clamp_(0, N - 1).long()
sorted_c, _ = calo_scores.sort(dim=-1, descending=True)
thr_c = sorted_c.gather(1, k_calo.unsqueeze(1))                               # k-th score = threshold
sampled_extra = extra_pred & (calo_scores > thr_c)                           # ~k FP nodes that event
reco_calo_mask = truth_calo_kept | sampled_extra
```

Tracks use the same pattern with a proportional count `k = round(frac × #trueHStracks)`. With
`std=0` and `fn_frac=0` the calo path reduces to "keep all truth + top-`mean` FP" = the old
behavior. (A single clean path — no fixed-count fallback, per the readability requirement.)

### `configs/base.yaml` (the shipped recipe)

```yaml
reco_calo_noise_mean: 170      # mean FP pileup calo nodes per event
reco_calo_noise_std:  100.0    # per-event Gaussian spread of the FP count
reco_calo_fn_frac:    0.10     # random fraction of true HS calo dropped
reco_track_fn_frac:   0.10     # random fraction of true HS tracks dropped
reco_track_fp_frac:   0.10     # FP tracks added ~ frac * (#true HS tracks)
```

A one-step compile + train smoke test passed.

---

## 8. Verification

`verify_tf_selection` runs the **real** model (TF vs pred), reads its actual stored selection
(`net._reco_node_indices` / `_reco_filtered_valid`), and overlays `TF (model)` vs `pred` on the same
metric panels + energy spectra as the sweep. It **overrides the recipe knobs from the current
`base.yaml`** onto the loaded checkpoint (because `load_from_checkpoint` restores the *old* saved
recipe), so it genuinely tests the new recipe. Success criterion: the `TF (model)` curve sits on
`pred` for purity, recall, occupancy, n_calo and both energy spectra
(should match `tf_sweep_metrics__…__fn_random_0_10_fp_170_100.png`).

---

## 9. Conclusions & next steps

- The val>train gap was **exposure bias in Stream C**, not overfitting; it is **~entirely the
  classification head** and **~92% driven by the calo selection**.
- The current teacher forcing was non-representative: it **never dropped true signal** and used a
  **fixed, too-narrow** FP count.
- The matched recipe — **calo FN 10% (random) + FP 170±100, tracks 10%/10%** — reproduces the
  inference selection on purity, recall, size, and energy spectra.
- Implemented config-driven and compile-safe; defaults preserve old behavior, `base.yaml` turns it on.

**Next:** retrain with the new `base.yaml`; expect the train(TF) and val(pred) classification CE to
converge (the exposure gap to shrink). Re-running `compare_tf_vs_pred_loss` after training quantifies
the remaining gap (a residual is expected from genuine Stream-B false negatives, which teacher
forcing makes the model robust to but cannot recover).

---

## File map (what changed in this work)

| File | Change |
|---|---|
| `model.py` | `use_teacher_forcing` + independent `track_tf`/`calo_tf` overrides; FN-drop + Gaussian/proportional FP selection in the TF branch; new config knobs |
| `configs/base.yaml` | the matched recipe values |
| `eval_data.py` | `compare_tf_vs_pred_loss`, `analyze_tf_mask_strategies`, `sweep_tf_mask_schemes`, `diagnose_track_class_attribution`, `verify_tf_selection`, plotting helpers; `files_list` data-module fix |
| `pflow_data.py` | `files_list` as a first-class `ODDDataModule` parameter |
| `run_forward_pass.py` | driver wiring for each analysis above |
