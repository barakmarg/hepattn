# Trackless Reclassification and Inference-Mode Class Drift

## Summary

This note documents why particle class-distribution plots can change drastically when toggling `is_inference`, even for the same model and input data.

Primary finding:
- `is_inference` currently changes the **truth-label definition** in the dataset pipeline, not only runtime behavior.

Secondary finding:
- `is_inference` also toggles duplicate-track filtering, which can slightly change event composition.

## Observed Symptoms

- Class-distribution plots degrade in some inference-mode runs.
- Truth class composition shifts strongly between `is_inference=False` and `is_inference=True`.
- In test outputs, truth electrons can increase by about an order of magnitude when using `is_inference=True`.

## Root Cause Analysis

Two code paths are toggled by `is_inference` in `pflow_data.py`.

### 1. Trackless charged particle reclassification (major effect)

In `load_event`, the reclassification of trackless charged particles is only applied when `is_inference=False`:
- trackless charged hadron/electron are shifted to neutral-like classes
- trackless muons are mapped to neutral-hadron class

When `is_inference=True`, this reclassification is skipped.

Consequence:
- truth `reco_particle_class` semantics differ across modes.
- prediction writer stores these truth labels into H5 `object_class`, so downstream analysis sees different truth definitions.

### 2. Duplicate-track filtering (secondary effect)

In `preprocess_hook`, duplicate matched tracks are filtered only when `is_inference=False`.

Consequence:
- track counts differ slightly between modes.
- event filtering (`max_nodes`) can differ slightly, producing small event-set drift.

## Quantitative Evidence

Dataset-level A/B on the same shard and aligned event IDs (common events):
- common events: 79
- truth electron count (class 1): 69 (`is_inference=False`) -> 696 (`is_inference=True`)
- dominant remaps matched expected trackless logic:
  - class 4 -> class 1: 627
  - class 3 -> class 0: 1064
  - class 3 -> class 2: 23

H5-level comparison on common event IDs (300-event slice, 297 common IDs):
- truth class histogram changed strongly
- predicted class histogram changed only mildly
- truth class-1 count: 530 -> 5314

This confirms that the largest shift comes from truth-definition changes, not from model prediction instability.

## Important Clarification

`run_forward_pass(..., inference_mode=...)` uses:
- `inference_mode` for Lightning Trainer runtime context
- `is_inference` for dataset preprocessing/label definition

These are different switches.

So setting `inference_mode=False` does not restore non-inference truth reclassification if the datamodule is still built with `is_inference=True`.

## Does Trackless Reclassification Make Sense?

It depends on the objective.

### When it makes sense

If the target definition should reflect detector-observable behavior, reclassifying trackless charged objects to neutral-like classes can be justified. Without a matched track, charged identity is often not robustly reconstructable.

### When it causes problems

For benchmarking and run-to-run comparability, tying this truth-label rule to `is_inference` is problematic because:
- truth semantics change between runs,
- class-distribution comparisons become apples-to-oranges.

## Recommendation

- Keep one truth-label policy across all evaluation runs used for comparison.
- If trackless reclassification is desired, expose it as a separate explicit config/flag (independent from runtime inference toggles).
- Record that choice in experiment metadata and plot captions.

## Practical Interpretation for Current Results

The observed "much more electrons in inference mode" is expected from current logic and is primarily due to trackless reclassification being disabled in inference mode.
