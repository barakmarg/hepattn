# Plan: Exposure-bias diagnostic loss + buffer-occupancy histogram

## Context

The model trains Stream C (reconstruction) under **teacher forcing** (truth masks + bounded
sampled noise) but validates under the **prediction path** (Stream A/B predicted masks).
Because the branch is gated by `self.training and self.teacher_forcing`
(`model.py:204`, `model.py:276`), `train/loss` and `val/loss` are computed on **different input
distributions to Stream C** and are not directly comparable. The observed gap (val≈13 bouncing
vs train≈10.2) is therefore largely **exposure bias**, not overfitting — consistent with the
purity plot (training-mask purity 0.307 vs inference 0.188).

This change adds a **flag-gated diagnostic** (default OFF, zero cost in normal runs) that, when
enabled, runs the *opposite* conditioning path so the four numbers decompose the gap:

- `val/loss` (pred path, existing) vs **`val/loss_tf`** (new, teacher-forcing path) → exposure gap
- `train/loss` (TF path, existing) vs **`train/loss_pred`** (new, pred path) → confirms the same gap from the train side
- `val/loss_tf` vs `train/loss` → the *true* overfitting gap

It also adds a **buffer-occupancy histogram** comparing how many of the 1400 reco-buffer slots
are filled per event under teacher forcing vs the prediction path, captured on the **same**
validation events via the same pair of forwards. Expectation: the TF curve sits lower/cleaner,
the pred curve fills toward 1400 (recall-biased Stream B floods Stream C).

Decisions: diagnostic is **behind an off-by-default flag** (when on, runs every val batch);
diagnostic losses logged as **totals only** (no per-stream breakdown).

## Changes

### 1. `model.py` — add a teacher-forcing override (no behavior change when unused)

Thread an optional `use_teacher_forcing: bool | None = None` through the forward path so the
lightning module can force either conditioning path regardless of `self.training`.

- `forward(self, inputs, targets=None, use_teacher_forcing=None)` (`model.py:126`). After the
  encoder, compute once:
  ```python
  tf_mode = (self.training and self.teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
  ```
- Bridge branch (`model.py:204`): replace `if self.training and self.teacher_forcing:` with `if tf_mode:`.
- Pass `tf_mode` into the `_forward_reco(...)` call (`model.py:248`) and add a `tf_mode`
  parameter to `_forward_reco` (`model.py:257`).
- In `_forward_reco`, replace `if self.training and self.teacher_forcing and targets is not None:`
  (`model.py:276`) with `if tf_mode and targets is not None:`, and the debug branch
  (`model.py:322`) with `elif (not tf_mode) and self.reco_debug_use_truth_masks and targets is not None:`.

When `use_teacher_forcing is None`, `tf_mode == self.training and self.teacher_forcing`, so all
existing training/inference/predict-writer behavior is **byte-identical**.

### 2. `reco_analysis.py` — new occupancy plot helper

Add `plot_reco_buffer_occupancy(occ_tf, occ_pred, max_reco_nodes=1400)` right after
`plot_calo_mask_energy_purity` (`reco_analysis.py:3428`), mirroring its exact visual style: two
`histtype="step", linewidth=2.2` curves (TF in `#55a868`, pred in `#dd8452`), x-axis = "valid
reco-buffer nodes per event", legend labels `f"... n={size} mean={m:.1f} median={md:.1f}"`,
title noting `max_reco_nodes`. Inputs are 1-D arrays of per-event valid-node counts. Returns a
`plt.Figure`.

### 3. `lightning_module.py` — flag, dual forward, occupancy capture, plot

- **`__init__`** (`lightning_module.py:49`): add `diagnostic_dual_loss: bool = False` parameter,
  store as `self.diagnostic_dual_loss`. Add a small `_total_loss(losses)` helper that sums all
  leaf tensors in the nested losses dict (so we log a single total without the per-stream
  sub-keys that `log_losses` emits).
- **`on_validation_epoch_start`** (`lightning_module.py:147`): add `self._val_occ_data = defaultdict(list)`.
- **`validation_step`** (`lightning_module.py:114`): after the existing pred-path
  forward/loss/metrics, when `self.diagnostic_dual_loss`:
  1. capture pred-path occupancy: `occ_pred = self.model._reco_filtered_valid.sum(1)` (guarded by `hasattr`);
  2. run the TF forward under `torch.no_grad()`:
     `outputs_tf = self.model(inputs, targets=targets, use_teacher_forcing=True)`,
     `losses_tf = self.model.loss(outputs_tf, targets)`;
  3. `self.log("val/loss_tf", self._total_loss(losses_tf), sync_dist=True)`;
  4. capture `occ_tf = self.model._reco_filtered_valid.sum(1)`;
  5. append both (`.detach().cpu().numpy()`) to `self._val_occ_data["pred"]` / `["tf"]`.
- **`training_step`** (`lightning_module.py:88`): when
  `self.diagnostic_dual_loss and batch_idx % self.trainer.log_every_n_steps == 0`, run a
  `torch.no_grad()` pred-path forward (`use_teacher_forcing=False`) + loss and
  `self.log("train/loss_pred", self._total_loss(losses_pred), sync_dist=True)`. Placed after the
  existing metric block, before the `mtl` branch. Does not touch `total_loss`/backward.
- **`on_validation_epoch_end`** (`lightning_module.py:154`): after the existing concat block, if
  `self._val_occ_data["tf"]`, concat both arrays and
  `figs["reco_analysis/buffer_occupancy"] = plot_reco_buffer_occupancy(occ_tf, occ_pred, max_reco_nodes=self.model.max_reco_nodes)`.
  Existing Comet logging loop (`lightning_module.py:302`) handles it.
- Add `plot_reco_buffer_occupancy` to the import from `reco_analysis` (`lightning_module.py:11`).

### 4. `configs/base.yaml` — expose the flag

Add `diagnostic_dual_loss: false` under the `model:` block (sibling of `mtl`, `base.yaml:125`).
Default off → no change to normal runs.

## Notes / correctness

- The TF forward in validation uses `targets` (available) and runs with `self.training=False`
  (dropout off) — a clean loss estimate. The pred forward in training runs with dropout on; it is
  a monitoring scalar only, under `no_grad`, so no gradient/DDP interference.
- `self.model._reco_filtered_valid` is `(B, 1400)` bool; occupancy = `.sum(1)`. Set every forward
  when `reco_decoder is not None` (always true here); guarded with `hasattr`.
- Under DDP, occupancy arrays accumulate per-rank and the figure is logged on the logger rank —
  identical to all existing epoch-end plots (e.g. `plot_calo_mask_energy_purity`).

## Verification

1. **Flag off (regression):** run a few train+val steps with `diagnostic_dual_loss: false`;
   confirm no `val/loss_tf` / `train/loss_pred` keys and unchanged timing/behavior.
2. **Flag on (smoke):** set `diagnostic_dual_loss: true`, run 1 epoch on a small subset
   (~500 events/group). Confirm in Comet:
   - scalars `val/loss_tf` and `train/loss_pred` appear;
   - figure `reco_analysis/buffer_occupancy` logs with two overlaid curves + n/mean/median labels.
3. **Sanity of the decomposition:** expect `val/loss_tf ≈ train/loss` (TF path both) and
   `val/loss > val/loss_tf` (exposure gap); occupancy: pred curve fuller (toward 1400) than the
   TF curve. If `val/loss_tf` collapses to ≈ `train/loss`, the gap is confirmed exposure bias,
   not overfitting — the intended diagnostic outcome.

## Files to touch

- `src/hepattn/experiments/odd_pileup_reco/model.py`
- `src/hepattn/experiments/odd_pileup_reco/lightning_module.py`
- `src/hepattn/experiments/odd_pileup_reco/reco_analysis.py`
- `src/hepattn/experiments/odd_pileup_reco/configs/base.yaml`
