# Teacher-Forcing Setup — `odd_pileup_reco`

The exact, current setup for how Stream C (reconstruction) is conditioned during training,
and how the train/inference exposure-bias gap is closed. This is the **"what it is" reference**;
for the investigation that led here see [`RESEARCH_teacher_forcing_exposure_bias.md`](RESEARCH_teacher_forcing_exposure_bias.md).

---

## The short version

Stream C reconstructs particles from a **selected set of hard-scatter (HS) nodes**. The selection
comes from one of two sources:

- **Teacher-forcing (TF) path** — built from *truth* HS masks (plus injected noise). Used while training when `teacher_forcing: true`.
- **Prediction path** — built from the *predicted* Stream A/B masks, then top-1200 calo by energy. Used at validation/inference always, and also during training when `teacher_forcing: false`.

The fix is a **two-stage training curriculum**:

| Stage | `teacher_forcing` | Stream-C input source | Purpose |
|---|---|---|---|
| **1. Warm-up** | `true` | truth HS + matched FN/FP noise | Stable learning signal; teaches the reco head to name particles from realistic-but-guided inputs. |
| **2. Fine-tune** | `false` | the real prediction path | Train on the **exact inference distribution** → zero train/val mismatch; closes the residual exposure-bias gap. |

Stage 2 starts from the stage-1 checkpoint. Because stage-2 training feeds Stream C the same
predicted selection used at validation, `train/loss` and `val/loss` become directly comparable and
the gap collapses.

> **Current config state:** `configs/base.yaml` has `teacher_forcing: false` → it is the **stage-2
> (fine-tune)** config. To run stage 1, set `teacher_forcing: true`.

---

## The master switch

`model.py::TwoStreamMaskFormer.forward` (~L148):

```python
tf_mode = (self.training and self.teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
```

- `self.teacher_forcing` is the config flag (`base.yaml` → `model.init_args.teacher_forcing`).
- TF is only ever active **while training** (`self.training`) **and** the flag is on.
- Validation/test (`self.training == False`) → always the prediction path, regardless of the flag.
- `use_teacher_forcing` / `track_tf` / `calo_tf` overrides exist **only for diagnostics**
  (`eval_data.py`); they are `None` in normal train/eval, so behavior is unchanged.

`tf_mode` then routes both the Stream-A/B → Stream-C bridge and the Stream-C node selection in
[`_forward_reco`](model.py) (~L276–395).

---

## What each path feeds Stream C

### Prediction path (`tf_mode == False`) — stages 2 + all validation/inference
`model.py::_forward_reco`, `else` branches (~L337, L377):

- **Tracks:** `sigmoid(track mask logits) >= 0.5`, restricted to track nodes.
- **Calo:** `sigmoid(calo mask logits) >= calo_pred_threshold`, restricted to non-track valid nodes.
- Then the shared **top-`max_reco_calo_nodes` (1200) by energy** cap, union with tracks.

This is the distribution the model must ultimately perform on.

### Teacher-forcing path (`tf_mode == True`) — stage 1
Built from truth, then **distribution-matched to the prediction path** by injecting both
false negatives (drop true signal) and false positives (add predicted pileup):

**Calo** (`model.py` ~L342–368):
1. `truth_calo_mask` = clusters with `hs_frac > calo_hs_frac_threshold` **and** `hs_energy > calo_hs_energy_threshold`.
2. **FN drop:** randomly remove `reco_calo_fn_frac` of them.
3. **FP add:** from the Stream-B predicted-but-not-truth pool, sample a **per-event count `~ N(reco_calo_noise_mean, reco_calo_noise_std)`**.
4. Union, then the same top-1200-by-energy cap as the prediction path.

**Tracks** (`model.py` ~L313–334):
1. `truth_track_mask` = truth HS tracks.
2. **FN drop:** random `reco_track_fn_frac`.
3. **FP add:** `~ reco_track_fp_frac × (#true HS tracks)` from the predicted-but-not-truth pool.
4. Union.

The variable per-event FP count is selected **compile-safely** (no `.item()` / dynamic shapes):
a tensorized `sort` + per-row `gather` threshold (the model is `torch.compile(dynamic=True)`'d via
`callbacks/compile.py`). With `std=0` and `fn_frac=0` the TF path reduces to "all truth + top-`mean`
FP" — the original pre-fix behavior.

---

## The knobs

`TwoStreamMaskFormer.__init__` (`model.py` ~L55–63), shipped values from `configs/base.yaml`:

| knob | meaning | base.yaml |
|---|---|---|
| `teacher_forcing` | master switch (stage 1 = `true`, stage 2 = `false`) | `false` |
| `calo_hs_frac_threshold` | min HS energy fraction for a truth HS cluster | `0.10` |
| `calo_hs_energy_threshold` | min HS energy for a truth HS cluster | `0.15` |
| `calo_pred_threshold` | sigmoid threshold on predicted calo mask | `0.3` |
| `reco_calo_noise_mean` | mean FP pileup calo nodes added per event | `170` |
| `reco_calo_noise_std` | per-event Gaussian spread of the FP count | `100.0` |
| `reco_calo_fn_frac` | random fraction of true HS calo dropped | `0.10` |
| `reco_track_fn_frac` | random fraction of true HS tracks dropped | `0.10` |
| `reco_track_fp_frac` | FP tracks added ≈ `frac × (#true HS tracks)` | `0.10` |
| `max_reco_calo_nodes` | top-k calo-by-energy cap | `1200` |
| `max_reco_nodes` | total Stream-C node buffer | `1400` |

The TF-noise knobs (`*_noise_*`, `*_fn_frac`, `*_fp_frac`) only matter in **stage 1**; in stage 2
(`teacher_forcing: false`) the TF branch is never taken, so they are inert.

---

## Running each stage

```bash
# Stage 1 — warm-up with teacher forcing (set teacher_forcing: true in base.yaml)
python main.py fit --config configs/base.yaml

# Stage 2 — fine-tune on the prediction path (teacher_forcing: false), from the stage-1 checkpoint
python main.py fit --config configs/base.yaml --ckpt_path path/to/stage1.ckpt
```

(Lightning CLI; see `main.py`.)

---

## Why two stages instead of just `teacher_forcing: false` from scratch

Training Stream C directly on the prediction path from random init is unstable: early in training the
Stream-A/B masks are garbage, so Stream C is fed near-random node sets and never gets a clean learning
signal for the global, discrete **classification** decision (the part most sensitive to incomplete hit
sets — see the research doc, §3). Stage 1's matched TF gives a clean-but-realistic signal to learn the
task; stage 2 then removes the last bit of train/inference mismatch by training on exactly what
inference sees.
