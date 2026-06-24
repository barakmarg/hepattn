# Preprocessing issues — pu0-overlay-pu200 production

What's wrong in the data preprocessing, where, why, and the fix. Two layers:
the **dataset-creation script** (root cause) and the **loader `preprocess_hook`**
(where we currently band-aid it). Fixing it at creation lets the loader stay simple.

Files:
- Creation: `/storage/agrp/barakma/PileupODD/primary/create_training_dataset_pileup_overlay.py`
- Loader:   `src/hepattn/experiments/odd_pileup_reco/pflow_data.py`

---

## A. `vertex_primary` is degenerate (≡ 1) — carries NO hard-scatter/pileup signal
- **Fact:** in the produced parquet, *every* track and *every* target particle has `vertex_primary == 1` (raw `value_counts` = `{1: all}`).
- **Why:** HS particles are filtered to `vertex_primary == 1` in `_preprocess_source` (creation, lines 879–891: `_indices = vertex_primary.list.eval((element()==1).arg_true())` then `list.gather`). Each source event (HS and every PU minbias) marks its own primary-vertex particles `==1`; `_overlay_tracks` (1098–1140) just concatenates them → uniform `1`.
- **Consequence:** any code keying off `vertex_primary == 1` to mean "hard scatter" is a no-op. The real HS/PU signal is **`source_pileup_event_id`** (null ⇒ HS) on tracks, and **`target_particles` is HS-only by construction**.
- **Action:** stop using `vertex_primary` for HS/PU anywhere. Treat HS = `source_pileup_event_id` is null (tracks) / membership in `target_particles` (particles).

## B. (ROOT BUG) Track→particle uses event-local `particle_id` → cross-source collisions → ~45% pileup contamination
- **Where:** `filter_orphans_and_reindex` (creation, 562–601) maps tracks to `particle_idx` by joining `majority_particle_id` on `(event_id, particle_id)` against the HS particle mapping. It runs **after** overlay (`filter_orphans_and_reindex` @1358, `_overlay_tracks` @1198), so HS + PU tracks already share the HS `event_id`.
- **Why it's wrong:** `particle_id` is **event-local** (reused across the HS event and each PU source event; only ~10k distinct ids over ~150k particles). A pileup track whose PU-local `majority_particle_id` happens to equal a HS target particle's `particle_id` **matches the join** and gets a valid HS `particle_idx ≥ 0` → it is wired to a hard-scatter particle.
- **Measured impact:** ~5–6% of pileup tracks get a spurious HS `particle_idx`; on a typical event ~**45% of all track→particle links** in the incidence matrix were pileup contamination (e.g. event 10000: 47 real HS links vs 39 spurious). Also corrupts the HS-track label.
- **Fix (cheap, at creation):** `source_pileup_event_id` is already on the tracks at this point. In `filter_orphans_and_reindex` where it does `pl.col('particle_idx').fill_null(-1)` (line 581), additionally force `particle_idx = -1` for any track with `source_pileup_event_id` non-null. PU tracks then route to the pileup token (row 0) via the existing `-1` sentinel, and `particle_idx ≥ 0` becomes a clean HS-track flag.

## C. (REDUNDANT) The `vertex_primary` join onto tracks is dead weight
- **Where:** `_preprocess_source` (creation, 834–858) explodes tracks, joins `particles.vertex_primary` on `(event_id, majority_particle_id↔particle_id)`, and carries `vertex_primary` through the group-back.
- **Why:** since `vertex_primary ≡ 1` and HS/PU is decided by `source_pileup_event_id`, this join + the carried column produce information that is never usefully consumed. It's an extra explode + join + list column per event.
- **Action:** drop the `vertex_primary` join/column from the track path (and the `majority_particle_vertex_primary` rename in the loader, see E). Saves work and removes a misleading field.

---

## F. Double-matched-track dedup keys on event-local `particle_id` (lower severity)
- **Where:** `preprocess_hook` non-inference block (~991–1006) removes double-matched tracks grouped by `(event_id, particle_id)`.
- **Why:** same event-local-id namespace — tracks from different PU sources that share a `particle_id` could be wrongly conflated/removed. Edge effect, but worth auditing once B lands.

---

## Recommended fix order
1. **Creation, B:** null `particle_idx` for `source_pileup_event_id`-non-null tracks in `filter_orphans_and_reindex`. This is the single root fix — it makes `particle_idx ≥ 0 ⟺ HS track` true in the parquet, so the incidence matrix, HS mask, and vz are all correct without loader patches.
2. **Creation, C:** drop the now-useless `vertex_primary` track join/column.
3. **Loader, D/E/F:** once B is in the regenerated data, simplify `preprocess_hook` — remove the dead `n_distinct_vp` branch, the `source_pileup_event_id` band-aid, and (optionally) the zero-HS drop if creation guarantees ≥1 HS track or defines vz a fallback.

> Trade-off: fixing at creation requires **regenerating the 235 GB dataset**. The loader patches (E) are already in place and equivalent for training *now*; do the creation fix when you next regenerate, then strip the loader band-aids.
