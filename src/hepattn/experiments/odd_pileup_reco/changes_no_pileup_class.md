# Change Set: Remove Pileup Classification Class

## Summary

The design decision: **query 0 is a dedicated pileup energy sink** — it always matches target position 0
(fixed, no Hungarian matching), but it receives **no classification gradient**. There is no longer a
"pileup class" in the classification head. The only special class is `5 = null` (padding / residual).

```
BEFORE                                    AFTER
──────────────────────────────────────    ──────────────────────────────────────
class 0-4 : real particle types           class 0-4 : real particle types
class 5   : null / residual               class 5   : null / residual  (unchanged)
class 6   : pileup (query 0 was           (no pileup class)
            classified as class 6)
                                          query 0 target_class := -100
                                          → excluded from CE loss entirely
```

---

## File 1: `configs/base.yaml` — Classification task config

### BEFORE

```yaml
# 1. Classification task (7 classes: 0-4 real + 5 null + 6 pileup)
- class_path: hepattn.models.task.ObjectClassificationTask
  init_args:
    name: classification
    input_object: query
    output_object: reco_pflow
    target_object: reco_particle
    num_classes: 6
    losses:
      object_ce: 2
    costs:
      object_ce: 2
    net:
      class_path: hepattn.models.Dense
      init_args:
        input_size: *dim
        activation: torch.nn.SiLU
        output_size: 7
        hidden_layers: [128, 64, 32]
    null_weight: 0.5
    loss_class_weights: [1.0, 3.0, 8.0, 1.5, 1.0, 0.5]
    mask_queries: false
    has_intermediate_loss: true
```

### AFTER

```yaml
# 1. Classification task (6 classes: 0-4 real + 5 null)
- class_path: hepattn.models.task.ObjectClassificationTask
  init_args:
    name: classification
    input_object: query
    output_object: reco_pflow
    target_object: reco_particle
    num_classes: 5
    losses:
      object_ce: 2
    costs:
      object_ce: 2
    net:
      class_path: hepattn.models.Dense
      init_args:
        input_size: *dim
        activation: torch.nn.SiLU
        output_size: 6
        hidden_layers: [128, 64, 32]
    null_weight: 0.5
    loss_class_weights: [1.0, 3.0, 8.0, 1.5, 1.0]
    mask_queries: false
    has_intermediate_loss: true
```

**What changed:**
- `num_classes`: 6 → 5 (pileup class removed)
- `output_size`: 7 → 6 (one fewer logit)
- `loss_class_weights`: 6 weights → 5 weights (pileup weight `0.5` removed)

---

## File 2: `pflow_data.py` — Pileup token class in `load_event()`

### BEFORE

```python
# Shift particle data by 1 to make room for pileup token at position 0
for key, val in particle_data.items():
    padded = torch.zeros(self.num_objects, *val.shape[1:]) if val.dim() > 1 else torch.zeros(self.num_objects)
    padded[1:n_particles + 1] = val[:n_particles]
    particle_data[key] = padded

# Pileup token at position 0: class 6 (pileup) — always valid for cost computation
particle_data["class"][0] = 6
particle_data["is_charged"][0] = 0
```

### AFTER

```python
# Shift particle data by 1 to make room for pileup token at position 0
for key, val in particle_data.items():
    padded = torch.zeros(self.num_objects, *val.shape[1:]) if val.dim() > 1 else torch.zeros(self.num_objects)
    padded[1:n_particles + 1] = val[:n_particles]
    particle_data[key] = padded

# Pileup token at position 0: dummy class 0 (ignored in loss via ignore_index=-100)
particle_data["class"][0] = 0
particle_data["is_charged"][0] = 0
```

**What changed:**
- `particle_data["class"][0]` = 6 → 0
- Class 0 is a valid real-particle class index, so `object_ce_cost` (`torch.gather`) doesn't crash
  during matching. The value only matters for cost computation; in the actual loss it is replaced
  with `-100` by `maskformer.py` (see below).

---

## File 3: `maskformer.py` — Loss function, classification handling

### BEFORE

The `loss()` method had no special handling for query 0 in the classification task:

```python
# Compute the losses for each task in each block
for layer_name in outputs:
    losses[layer_name] = {}
    for task in self.tasks:
        if layer_name != "final" and not task.has_intermediate_loss:
            continue

        # Mask position 0 for regression (pileup has no kinematics)
        if isinstance(task, IncidenceBasedRegressionTask):
            masked_valid = targets[valid_key].clone()
            masked_valid[:, 0] = False
            masked_targets = {**targets, valid_key: masked_valid}
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], masked_targets)
        else:
            # Classification, mask, incidence: all tasks see position 0 normally
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)
```

### AFTER

The `loss()` method adds an `elif` branch for `ObjectClassificationTask` that sets position 0 to
`ignore_index=-100`, which is the default ignore index in `F.cross_entropy`:

```python
# Compute the losses for each task in each block
for layer_name in outputs:
    losses[layer_name] = {}
    for task in self.tasks:
        if layer_name != "final" and not task.has_intermediate_loss:
            continue

        # Mask position 0 for regression (pileup has no kinematics)
        if isinstance(task, IncidenceBasedRegressionTask):
            masked_valid = targets[valid_key].clone()
            masked_valid[:, 0] = False
            masked_targets = {**targets, valid_key: masked_valid}
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], masked_targets)
        # Exclude position 0 (pileup) from classification loss via ignore_index=-100
        elif isinstance(task, ObjectClassificationTask):
            masked_class = targets[class_key].clone()
            masked_class[:, 0] = -100
            masked_targets = {**targets, class_key: masked_class}
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], masked_targets)
        else:
            losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)
```

**What changed:**
- New `elif isinstance(task, ObjectClassificationTask)` branch
- `target_class[:, 0] = -100` using a shallow-copy dict (`{**targets, class_key: masked_class}`)
  to avoid in-place autograd mutation errors
- `F.cross_entropy` with `ignore_index=-100` (its default) silently skips position 0

The `predict()` method also forces query 0 to be marked valid after classification, since it never
receives a classification gradient and might be predicted as null:

```python
def predict(self, outputs: dict) -> dict:
    preds = {}
    for layer_name, layer_outputs in outputs.items():
        preds[layer_name] = {}
        for task in self.tasks:
            if layer_name != "final" and not task.has_intermediate_loss:
                continue
            preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

            # Query 0 (pileup sink) is always valid — bypass null filtering
            # since query 0 receives no classification gradient
            if isinstance(task, ObjectClassificationTask):
                valid_key = task.output_object + "_valid"
                if valid_key in preds[layer_name][task.name]:
                    preds[layer_name][task.name][valid_key][:, 0] = True
```

This block is **new** — it did not exist before. Without it, query 0 would frequently be predicted
as null (class 5) and filtered out during inference, erasing the pileup energy prediction.

---

## File 4: `lightning_module.py` — Metrics

### BEFORE

```python
self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=7, average="micro")
self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=7, average="macro")
```

### AFTER

```python
self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=6, average="micro")
self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=6, average="macro")
```

**What changed:** `num_classes`: 7 → 6 (matches `output_size` in config).

---

## Design Rationale

| Concern | Resolution |
|---|---|
| `object_ce_cost` uses `torch.gather(logits, target_class)` — needs a valid (0–4) index at pos 0 to avoid index out-of-bounds during matching | `particle_data["class"][0] = 0` in data; real loss sets it to `-100` after matching |
| `F.cross_entropy` ignores index `-100` by default | Setting `target_class[:, 0] = -100` requires no changes to `task.py` |
| Query 0 never learns a class, but `ObjectClassificationTask.predict()` may mark it invalid (null) | `predict()` in `maskformer.py` hardcodes `valid[:, 0] = True` after task prediction |
| Shallow-copy pattern `{**targets, key: new_tensor}` | Avoids in-place mutation of `targets` dict, which caused `CUDABoolType version mismatch` autograd errors |
