# Investigation: Inf in `reco_*_mask_mask_bce`

## What We Know for Sure

### 1. The Inf appears in the reco stream mask BCE loss (only)

From `aaa.out.txt` and `fx.out`:
```
Inf in loss: reco_layer_0_mask_mask_bce at batch 0
Inf in loss: reco_layer_1_mask_mask_bce at batch 1
...
```
Every batch, every decoder layer, every epoch. Only `mask_bce` — not `mask_dice`, not classification, not regression.

### 2. The logits are NOT extreme

Debug output (from compiled model, sanity check / validation):
```
[DBG logit] min=-42.75 max=30.62 mean=-1.04
[DBG logit] isinf=0 isnan=0
[DBG node_valid] valid_per_sample: min=1400 max=1400 mean=1400.0
```

No Inf, no NaN. Max |logit| = 42.75. The `finfo.min → -100` masking at `task.py:320`
**never fires** for the reco stream because `node_valid` is all True (all 1400 filtered
nodes are valid).

### 3. The `finfo.min → -100` fix still resolved the Inf

Commit `1f3cc36` ("no inf loss, first big train") applied the `-100` fix and the Inf
disappeared. But as shown above, the reco stream's `node_valid` is all True, so the
masking at line 320 never fires. This seems contradictory.

**Resolution**: the fix also changed `CaloHitMaskTask` and `CaloNodeMaskTask` in
`tasks.py`. These are in Stream B (calo), not Stream C (reco). The Inf might have been
reported at the wrong layer/task, OR the calo stream's Inf was propagating through
shared state.

**Actually — more likely:** the git commit `1f3cc36` also included config changes
(`base.yaml`) — switching `overtrain: false`, `num_train: -1`, etc. A configuration
change might be the real fix, not the code change. Need to verify by reverting just
the code or just the config.

### 4. There is a broadcasting bug in `mask_bce_loss`

**File**: `models/loss.py:229-258`

When `object_valid_mask` filters objects, it flattens the batch+object dims:
```python
pred_logits = pred_logits[object_valid_mask]   # (B, 400, 1400) → (N_valid, 1400)
```

But `input_pad_mask` is NOT filtered and retains its (B, N) shape:
```python
loss = loss * input_pad_mask.unsqueeze(1)      # (N_valid, 1400) * (64, 1, 1400)
```

PyTorch broadcasts this to `(64, N_valid, 1400)` — a massive tensor with wrong
semantics (each sample's pad mask gets applied to ALL samples' valid objects, not
just its own).

**Why it doesn't crash here**: `node_valid` is all True, so `input_pad_mask` is all 1s,
making the multiplication a no-op. The broadcast creates a larger tensor but with
identical values.

**This is still a latent bug** that would produce wrong loss values if `node_valid` had
any False entries in the reco stream.

### 5. `sss.out` (no Inf) vs `aaa.out.txt`/`fx.out` (Inf) — code differences

`sss.out` (April 1 16:07, NO Inf) used the **old** maskformer code with:
```python
# Forced pileup assignment in cost matrix
cost[:, 0, 1:] = 1e6    # query 0 can't match any real particle
cost[:, 1:, 0] = 1e6    # no other query can match pileup target
cost[:, 0, 0] = -1e6    # forced match
pred_idxs = self.matcher(cost, targets["reco_particle_valid"])
```

`aaa.out.txt`/`fx.out` (March 31-April 1, Inf) used the **new** code with:
```python
# Exclude pileup from matching entirely
cost_no_pu = cost[:, 1:, 1:]
valid_no_pu = targets["reco_particle_valid"][:, 1:]
pred_idxs_no_pu = self.matcher(cost_no_pu, valid_no_pu)
pred_idxs = cat([zeros, pred_idxs_no_pu + 1], dim=1)
```

The pileup-exclusion approach introduced the Inf.

## What I Don't Know / Hypotheses

### Hypothesis A: The broadcast bug amplifies loss for the pileup token

With the new matching approach, the pileup token (query 0) has target ≈ all 1s (1344/1400
nodes). At random init, roughly half of logits are negative, giving BCE ≈ 10-20 per node.

After `object_valid_mask` boolean indexing → `(N_valid, 1400)`, then broadcasting with
`(64, 1, 1400)` → `(64, N_valid, 1400)`. The `.mean()` now divides by `64 * N_valid`
instead of `N_valid`. This **reduces** the loss by 64x, not amplifies it. So this doesn't
explain Inf.

### Hypothesis B: `torch.compile` + bf16 computes BCE differently during training

The debug prints fire during validation (sanity check). The Inf appears during training.
Both are compiled, both use the same model, same data (overtrain mode).

Under `torch.compile(dynamic=True)` with `bf16-mixed`, the compiler might fuse operations
differently between `model.train()` and `model.eval()` modes (e.g., dropout, different
autocast regions). If torch.compile bypasses the autocast fp32 promotion for BCE (computing
in bf16 instead of fp32), then:

- `BCE(logit=-42.75, target=1)` in bf16: `log(1 + exp(42.75))` — `exp(42.75)` in bf16
  overflows (bf16 max ≈ 65504, `exp(42.75) ≈ 3.7e18` — OK actually, doesn't overflow)
- But the numerically stable formulation: `max(x,0) - x*y + log(1+exp(-|x|))` should
  handle this fine.

**This hypothesis needs testing**: run without `torch.compile` and check if Inf persists.

### Hypothesis C: The per-element sum overflows in bf16 intermediate

If torch.compile fuses `BCE (reduction="none")` followed by `loss * pad_mask` and
`.sum(-1)` into a single bf16 kernel, the running sum over 1400 elements could overflow:

- Per-element BCE ≈ 10-20 on average
- Cumulative sum after 1400 elements: ~14000-28000
- bf16 max: 65504

This barely fits. But with sample_weight and worst-case logits, individual BCE values
can reach 42.75. If 1400 * 42.75 = 59850 < 65504, it still fits. So probably not this.

### Hypothesis D: Something specific about the pileup token + new matching

With the old matching, query 0 was forced onto target 0 in the cost matrix. The matcher
returns indices, and outputs are permuted. But the **loss function** always saw position 0
as pileup, consistently.

With the new matching, position 0 is kept fixed (no permutation). But the loss code at
`maskformer.py:278-304` has special handling only for `IncidenceBasedRegressionTask` and
`ObjectClassificationTask` — NOT for the mask task. The mask task's loss includes the
pileup token without any special treatment.

The pileup token (query 0) is supposed to capture ALL pileup nodes. Its target mask has
~1344 True entries out of 1400. The mask BCE loss on this token is:
```
BCE(logit, target=1) for 1344 nodes + BCE(logit, target=0) for 56 nodes
≈ 1344 * avg_bce_pos + 56 * avg_bce_neg
```

With random init, this could be ~1344*10 + 56*5 ≈ 13720 per object. Divided by
valid_counts=1400 ≈ 9.8. This is large but finite.

**I cannot definitively identify the mechanism producing Inf from code analysis alone.**

## Recommended Next Steps

1. **Add targeted debug in `mask_bce_loss` itself** — print the loss tensor just before
   `.mean()` to see where exactly Inf appears:
   ```python
   loss = F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight, reduction="none")
   if loss.isinf().any():
       print(f"[BCE DEBUG] Inf in raw BCE: {loss.isinf().sum()}, shape={loss.shape}")
       print(f"[BCE DEBUG] pred range: [{pred_logits.min():.2f}, {pred_logits.max():.2f}]")
       print(f"[BCE DEBUG] target range: [{targets.min():.4f}, {targets.max():.4f}]")
       print(f"[BCE DEBUG] pred dtype: {pred_logits.dtype}, target dtype: {targets.dtype}")
   ```

2. **Test without torch.compile** — remove the `Compile` callback from config and rerun.
   If Inf disappears, the issue is torch.compile fusing ops in bf16.

3. **Test without bf16** — set `precision: 32` and rerun. If Inf disappears, it's a
   precision issue in the loss computation.

4. **Fix the broadcasting bug** — even though it's currently harmless (all-True mask),
   it's a correctness issue that should be fixed. In `mask_bce_loss`, the
   `input_pad_mask` should be filtered/aligned with the `object_valid_mask`-filtered
   tensors, or don't filter by `object_valid_mask` at all and instead multiply loss
   by both masks.

## The Broadcasting Bug in Detail

```
mask_bce_loss(pred_logits, targets, object_valid_mask, input_pad_mask, sample_weight)
```

Shapes:
```
pred_logits:      (B, num_objects, num_inputs)  = (64, 400, 1400)
targets:          (B, num_objects, num_inputs)  = (64, 400, 1400)
object_valid_mask: (B, num_objects)              = (64, 400)
input_pad_mask:   (B, num_inputs)               = (64, 1400)
sample_weight:    (B, num_objects, num_inputs)  = (64, 400, 1400)
```

After `pred_logits = pred_logits[object_valid_mask]`:
```
pred_logits:  (N_valid, 1400)     # N_valid = sum of True in object_valid_mask
targets:      (N_valid, 1400)
sample_weight: (N_valid, 1400)
```

The `input_pad_mask` is still `(64, 1400)`. At line 251:
```python
loss = loss * input_pad_mask.unsqueeze(1)
# (N_valid, 1400) * (64, 1, 1400) → broadcasts to (64, N_valid, 1400)  ← WRONG SHAPE
```

Then at line 256:
```python
loss = loss.sum(-1) / valid_counts   # (64, N_valid) / (64, 1) → (64, N_valid)
return loss.mean()                    # mean over 64 * N_valid elements
```

The loss is divided by 64x too much (each valid object's loss gets counted 64 times,
then averaged over 64*N_valid instead of N_valid). When `input_pad_mask` is all True
the values are correct by coincidence, but the mean normalization is wrong.
