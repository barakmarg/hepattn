# Running `odd_pileup_reco` on the Weizmann cluster

Operational guide: how to get from a fresh shell to training, inference, paper
plots and PUPPI tuning.

The other docs here explain *what the model is* — `readme_odd_reco.md` (the
three-stream architecture, data pipeline, losses) and `README.md`
(TwoStreamMaskFormer walkthrough). **This file is the only one that explains how
to run it.** Read §8 (Gotchas) before your first run; most of the entries there
cost someone half a day to discover.

- [1. Environment](#1-environment)
- [2. Where things live](#2-where-things-live) — incl. [paper artifacts](#paper-artifacts--storageagrpbarakmaodd_paper)
- [3. Quick smoke test](#3-quick-smoke-test)
- [4. Training](#4-training)
- [5. Inference (forward pass)](#5-inference-forward-pass)
- [6. Paper plots](#6-paper-plots)
- [7. PUPPI baseline & tuning](#7-puppi-baseline--tuning)
- [8. Gotchas](#8-gotchas-read-this)

---

## 1. Environment

```bash
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
cd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco
```

The repo is importable as `hepattn.experiments.odd_pileup_reco` from this env;
there is nothing to `pip install`.

**The GPU matters.** The config uses flash attention
(`attn_type: flash-varlen`, `configs/base.yaml:178` and `:249`), so the model
needs a **GPU with flash-attention support** — the login nodes do not have one.
The cluster's A6000 nodes do, which is why the batch scripts request
`gputype=A6000`, but that is a convenience, not a requirement.

To run on an older GPU, switch the attention backend: `hepattn` also ships
`torch` (SDPA), `flex` and `flash` (see `ATTN_TYPES` in
`hepattn/models/attention.py`). Setting `attn_type: torch` at both config sites
works anywhere, at some speed cost.

CPU-only work (paper plots, PUPPI tuning) runs fine on a login node.

Environment variables used by the batch scripts:

| Variable | Value | Why |
|---|---|---|
| `COMET_API_KEY` | see `train_job.sh` | Comet logging; `LearningRateMonitor` needs *a* logger to exist |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `1` | stop BLAS oversubscription across DataLoader workers × ranks |
| `POLARS_MAX_THREADS` | `4` | polars threads × parallel processes otherwise oversubscribe / deadlock |
| `IOTHROTTLE_LIMIT` | `100` | Lustre I/O throttling |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduces allocator fragmentation |
| `ulimit -n` | `65536` | mmap backend opens one fd per shard per worker (best-effort; some interactive nodes have a lower hard limit) |

## 2. Where things live

**Datasets** (`/storage/agrp/barakma/PileupODD/data/`):

| Dir | Use |
|---|---|
| `ttbar_pu0_overlay_pu200/` | **training** — PU0 hard scatter with overlaid PU200 |
| `ttbar_pu200_all_vertices_paper/` | evaluation, in-distribution (fully simulated PU200) |
| `dihiggs_pu200_all_vertices_paper/` | evaluation, out-of-distribution |
| `ggf_pu200_all_vertices_paper/` | evaluation, out-of-distribution |

Paper samples are 100 events per parquet shard — the plotting code relies on
this (`--events-per-file`, default 100).

**Runs**: `logs/<run-name>/` holds `config.yaml` (a frozen copy of the config
used), `metadata.yaml`, and `ckpts/`. This directory is gitignored — it is
several hundred GB and lives only on disk.

**Config**: `configs/base.yaml` is the single source of truth for model, data
and trainer. `configs/odd_var_transform.yaml` holds the input feature scaling
and is referenced from `base.yaml:67` (`scale_dict_path`) — do not delete it.

### Paper artifacts — `/storage/agrp/barakma/odd_paper/`

**Everything the paper results were produced from is archived here** (385 MB),
outside the repo and outside the churn of `logs/`. Start here if you want to
reproduce or check a published number.

```
/storage/agrp/barakma/odd_paper/
├── models/
│   └── epoch=028-val_loss=10.15650.ckpt          the paper checkpoint (326 MB)
├── puppi/
│   ├── puppi_charged_subtract_v2_antikt_R04_best.{json,db}   <- the one in use
│   ├── puppi_charged_subtract_v3_10k_best.{json,db}          newer tunings,
│   └── puppi_charged_subtract_v4_10k_seeded_best.{json,db}   not used for the paper
├── paper_plots_epoch028_overlay_finetune/         ttbar (in-distribution)
└── paper_plots_dihiggs_epoch028_overlay_finetune/ dihiggs (out-of-distribution)
```

Notes:

- **One checkpoint, two samples.** Both figure sets come from the same
  `epoch=028` checkpoint; they differ only in the evaluation dataset (ttbar
  9,429 events / dihiggs 9,384). The matching H5 prediction shards live next to
  the checkpoint under `logs/odd_pflow_reco_20260620-T125301/ckpts/`.
- That run continued past epoch 28 and `epoch=037` reached a lower `val_loss`
  (10.106 vs 10.157) — epoch 28 is what the paper used regardless.
- Each plot dir keeps its `merged_aggregate.pkl` and `state/`, so **every figure
  can be re-rendered without touching the H5 shards** — point
  `run_paper_performance_plots.py --out-dir` at a copy of one of these.
- `v2_antikt_R04` is the tuning actually used; v3/v4 are later, unused ones
  (see §7).

## 3. Quick smoke test

Before a long job, prove the model builds and takes a step. Needs a
flash-attention-capable GPU (or `attn_type: torch` — see §1).

```bash
export COMET_API_KEY=<key from train_job.sh> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=4 IOTHROTTLE_LIMIT=100
python main.py fit --config configs/base.yaml --trainer.fast_dev_run=1 --trainer.devices=1 --data.num_train=200 --data.num_val=100 --data.num_test=100 --data.num_workers=0
```

Why each flag:

- `--data.num_train/num_val/num_test` — **this is what makes it fast.** The
  eager loader stops reading parquet as soon as it has this many events
  (`pflow_data.py:244`), so it reads a couple of parquet shards instead of the whole directory
  (~1000 shards x 4 parquet types).
- `--trainer.fast_dev_run=1` — 1 train + 1 val batch, checkpointing disabled, so
  nothing is written to `logs/`.
- `--data.num_workers=0` — no worker subprocesses, which sidesteps the eager
  backend's fork deadlock (§8).
- Batch size is deliberately **not** overridden — it stays at the config's 32.

## 4. Training

Submit with `start.sh`, which qsubs `train_job.sh`:

```bash
./start.sh
```

That is:

```bash
qsub -o output.log -e error.log -q N -N pflow-4gpu-big-finetune \
  -l walltime=72:00:00,mem=350gb,ncpus=32,ngpus=4,io=0.1,gputype=A6000 train_job.sh
```

`train_job.sh` then, on the compute node:

1. sets the threading/fd/allocator guards from §1;
2. **builds the mmap dataset locally** — 10 parallel `prep_mmap.py` jobs convert
   the Lustre parquet into packed `shard_*.pt` on node-local NVMe (`$TMPDIR/mmap`),
   then `--index-only` merges the per-shard metas into `shards_index.pt`. Takes
   ~20–30 min and is redone every job, since `$TMPDIR` is wiped at job end;
3. runs 4-GPU DDP training on the **mmap** backend:

```bash
python main.py fit --config configs/base.yaml \
    --data.backend mmap \
    --data.unify_path "$DST_MMAP" \
    --model.init_from_ckpt "$CKPT"
```

`--model.init_from_ckpt` copies **weights only** and restarts at epoch 0 with a
fresh optimizer and LR schedule. To genuinely resume a run, use `--ckpt_path`
instead.

To evaluate a checkpoint on the test split, edit `CKPT_PATH` in `run_eval.sh`
and run it — it calls `main.py test --config configs/base.yaml --ckpt_path …`.

### eager vs mmap

`configs/base.yaml` defaults to `backend: eager` (reads parquet directly).
Training uses `mmap` because it is much faster and avoids the fork deadlock.
Use eager for short/interactive work, mmap for real training.

## 5. Inference (forward pass)

Writes model predictions to H5 shards for the plotting stage.

```bash
./submit_forward.sh
```

It qsubs one single-GPU job per checkpoint listed in the `CKPTS` array,
passing each in via the `FWD_CKPT` env var, which `run_forward_pass.py:86-89`
reads. Output H5 shards land in a folder **next to the checkpoint**, named
`<ckpt-stem>__test<test_suff>/`, so runs never collide.

Edit `run_forward_pass.py` to choose the sample and size:

| Variable | Meaning |
|---|---|
| `DATA_DIR` | which evaluation dataset |
| `DATA_NUM_EVENTS` | how many events (10000 for the paper) |
| `test_suff` | names the output folder |
| `events_per_file` | events per H5 shard (1000) |
| `predict_only` | `True` skips the Hungarian-matching loss — **much faster, but predictions are then UNMATCHED**, so per-class and residual plots are meaningless. Use `False` for anything needing truth↔pred pairing. |

## 6. Paper plots

CPU-only; runs on a login node. Map-reduce over the H5 shards, cached and
resumable.

```bash
python run_paper_performance_plots.py \
  --h5 "logs/<run>/ckpts/<ckpt-stem>__test<suff>" \
  --parquet-dir /storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_paper \
  --out-dir paper_plots_<name>
```

Paste it as **one line** — see §8.

How the caching works, and when you need `--force`:

- Each shard's aggregate is cached at `<out-dir>/state/<shard>.pkl`, and the
  merge at `<out-dir>/merged_aggregate.pkl`.
- **Styling / layout changes**: just re-run. Figures re-render from cache in
  seconds.
- **New or changed aggregate** (a new quantity, changed binning): re-run with
  `--force`, otherwise you silently render stale or mis-shaped data.

The PUPPI comparison is mandatory: if the parquet dir does not contain the
events referenced by the H5, the run **fails loudly** rather than quietly
dropping PUPPI.

The exact figures used in the paper, with their cached aggregates, are archived
at `/storage/agrp/barakma/odd_paper/` (§2).

## 7. PUPPI baseline & tuning

The baseline is the **charged-subtract** variant: `puppi_charged_subtract.py`
(cluster-space, subtracts both HS- and PU-charged calo deposits using truth
track→cluster deposits) on top of the shared α-shape engine in `puppi.py`.

The tuned working point is read from a JSON hardcoded at
`run_paper_performance_plots.py:125-128` (`DEFAULT_BEST_JSON`):

```
optuna_puppi_charged_subtract/puppi_charged_subtract_v2_antikt_R04_best.json
```

**This file is load-bearing.** If it is missing, the script does *not* fail — it
falls back to `puppi_params = {}`, silently running PUPPI with the function's
default parameters and producing different figures. It is force-added to git
despite the `*.json` ignore rule; keep it that way.

Three tunings are kept: `v2_antikt_R04` (the one in use), `v3_10k` and
`v4_10k_seeded`. Each has a `.json` (parameters + provenance: trial number,
search vs validation scores, event counts, top-10 table) and a `.db` (full
Optuna study).

To re-tune (CPU, two-stage TPE search then held-out validation):

```bash
python optuna_puppi_charged_subtract.py --n-trials 200 --n-events 7000 \
  --n-validation-events 3000 --top-k 10 --study-name <name> --out-dir optuna_puppi_charged_subtract
```

Shrink `--n-trials 2 --n-events 20 --n-validation-events 20` for a smoke test,
and point `--out-dir` at a scratch directory so you do not touch the real studies.

## 8. Gotchas (read this)

**Flash attention.** Login nodes cannot run the model, and you will see this as
a crash or hang rather than a clear message. Use a flash-attention-capable GPU,
or set `attn_type: torch` at `configs/base.yaml:178` and `:249` to run anywhere.

**Forked-worker deadlock.** The eager backend can hang after the model summary
prints, when DataLoader workers fork — polars/BLAS thread pools do not survive
`fork()` cleanly. Fixes: `--data.num_workers=0` (smoke tests), or the mmap
backend plus the thread caps in §1 (real training).

**`--trainer.logger=false` crashes.** `base.yaml:106` registers
`LearningRateMonitor`, which raises `MisconfigurationException: Cannot use
LearningRateMonitor callback with Trainer that has no logger`. Leave the logger
alone and set `COMET_API_KEY`; under `fast_dev_run` Lightning swaps in a dummy
logger, so no Comet experiment is actually created.

**`ulimit -n 65536` may fail** on interactive nodes whose hard limit is lower
(`value exceeds hard limit`). Harmless for eager/`num_workers=0`; it only
matters for the mmap backend. If you chain it with `&&`, its failure will abort
your whole command.

**Old checkpoints embed module paths.** Lightning stores class paths inside
every `.ckpt` (`checkpoint["hyper_parameters"]`), and
`load_from_checkpoint()` imports them literally. Checkpoints trained before the
ODD experiments were consolidated reference
`hepattn.experiments.odd_pileup_maskformer.decoder.PileupMaskFormerDecoder`,
which is why a tiny compat shim package still exists — see its `__init__.py`.
Training from scratch never touches it. If you rename a model module, **existing
checkpoints will stop loading.**

**`*.json` is gitignored** repo-wide (`.gitignore:9`). Any JSON that code depends
on must be `git add -f`'d, or it silently vanishes from a fresh clone.

**Paste commands as one line.** In zsh, a `\` line-continuation followed by a
blank line terminates the command — the rest is then run as separate commands.
The symptom is `command not found: --h5` plus a run that used all-default
arguments. Check the first line of output: it echoes the resolved shard count and
out-dir, so you can see immediately whether your arguments took effect.

**Hungarian matching.** Predictions are only paired with truth when the matching
loss runs. With `predict_only=True` the model's output slots are unordered, so
per-class confusion, efficiency and residual plots are meaningless — the
feature-scatter diagonal will look wrong. Use `predict_only=False` for those.
