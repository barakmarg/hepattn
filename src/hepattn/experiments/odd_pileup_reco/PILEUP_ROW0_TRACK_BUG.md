# Pileup Row-0 Track Leakage Bug

## Summary

The pileup token (object row `0`) is intended to represent **calorimeter-only pileup energy**.
It should not own track columns in the truth incidence matrix.

However, in the target builder, negative track-to-particle indices can leak into row `0`
because of direct `+1` indexing during incidence assignment.

This makes diagnostics look like the model predicts many false positives for pileup-track
associations, even when part of the issue is in the truth target construction.

## Root Cause

In `pflow_data.py`, track incidence assignment is currently:

```python
# Tracks: HS tracks -> shifted rows (+1). PU tracks stay unassigned.
track_idx = np.arange(len(t_particle_idx))
t_pidx_np = t_particle_idx.numpy() if isinstance(t_particle_idx, torch.Tensor) else t_particle_idx
if self.is_inference:
    t_pidx_np[t_pidx_np < 0] = 0
incidence_matrix[t_pidx_np + 1, track_idx] = 1.0
```

### Why this is wrong

When `self.is_inference == False` (the common eval/training path), negative indices are not removed
before `+1` indexing. For values like `-1`, we get:

$$-1 + 1 = 0$$

So unmatched or invalid tracks can be written into row `0`, which is supposed to be pileup-calo only.

## Supporting Inconsistency

Earlier in the same method, invalid track indices are filtered for class relabeling:

```python
valid_idx = t_particle_idx[(t_particle_idx >= 0) & (t_particle_idx < n_particles)]
trackless_particle_mask[valid_idx] = False
```

But this validation is not reused in the actual incidence write, creating a mismatch between
class logic and incidence logic.

## Fix (Data Source)

Replace direct indexing with a validity mask and only write valid HS associations.

### Proposed patch

```python
# Tracks: only valid HS track -> particle assignments go to rows 1..n_particles.
track_idx = np.arange(len(t_particle_idx), dtype=np.int64)
t_pidx_np = t_particle_idx.numpy() if isinstance(t_particle_idx, torch.Tensor) else np.asarray(t_particle_idx)

valid_track_assoc = (t_pidx_np >= 0) & (t_pidx_np < n_particles)
if np.any(valid_track_assoc):
    incidence_matrix[t_pidx_np[valid_track_assoc] + 1, track_idx[valid_track_assoc]] = 1.0
```

This guarantees row `0` does not receive track assignments from invalid indices.

## Optional Fixes (Analysis Layer)

In `reco_analysis.py::plot_incidence_track_match`, row `0` is currently included by default:

```python
include_pileup_row: bool = True
```

For track-only match plots, safer defaults are:

1. Set `include_pileup_row=False` by default, or always exclude row `0` in track-only mode.
2. Explicitly zero row `0` on track columns in both truth and prediction before binary matching.
3. Keep numerical floors (`numerical_eps`, `pileup_pred_floor`) to suppress tiny bf16 noise.

These improve interpretability, but they do **not** replace the source fix in dataset construction.

## Validation Checklist

After patching `pflow_data.py`, validate with a small sample:

1. Load eval H5 and map truth incidence to filtered reco-node space.
2. Compute count of truth row-0 positives on track columns.
3. Expect near-zero (ideally zero) row-0/track truth occupancy.
4. Re-run track-match metrics excluding pileup row to measure real model precision/recall.

Suggested quick checks:

- `truth_filt[0, track_mask] > 0`
- `pred_filt[0, track_mask] > threshold`
- per-particle selected track counts after `select_charged_tracks`

## Impact

- Removes a systematic target contamination in row `0` track columns.
- Prevents inflated apparent false positives for pileup-token tracks.
- Aligns semantics with design intent: row `0` is pileup-calorimeter, not track-associated.