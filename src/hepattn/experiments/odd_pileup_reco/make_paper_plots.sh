#!/bin/bash
#
# Regenerate the GLOW-UP paper figures from the prediction-writer H5 shards.
#
# THIS IS WHAT WAS USED IN THE PAPER. The arguments below are the exact ones
# that produced the published figures: the epoch-28 fine-tuned model (stage 2,
# teacher forcing off) evaluated on two fully simulated <mu>=200 samples.
# Everything except the output directory is pinned, so a re-run reproduces the
# same figures rather than a re-tuned or re-binned variant of them.
#
# Two runs are performed, one per evaluation sample, each into its own
# subdirectory of OUT_DIR:
#
#   <OUT_DIR>/ttbar/     ttbar <mu>=200 -- the in-distribution sample. Source of
#                        every figure in Secs. 6.1-6.3 and App. C: track F1,
#                        calorimeter cluster energies, particle multiplicity,
#                        per-class residuals, the four jet residual panels, the
#                        jet residual box plots, and the per-class efficiency /
#                        fake rate / target pT spectrum.
#
#   <OUT_DIR>/dihiggs/   gluon-fusion di-Higgs (gg->HH) <mu>=200 -- the
#                        out-of-distribution sample the model never saw in
#                        training. Only its four jet panels are used, as Fig. 7
#                        (jet_relative_pt, jet_energy, jet_delta_eta,
#                        jet_delta_phi), copied into the paper with a dihiggs_
#                        prefix. The rest of its figures are produced too and
#                        simply go unused.
#
# Each run writes the same 14 figures -- exactly the set the paper draws from,
# verified against the original paper_plots_* output directories. The other
# diagnostics the analysis can produce are commented out in render_all() in
# run_paper_performance_plots.py. Output is PDF only, the format the paper
# includes.
#
# Usage:
#     ./make_paper_plots.sh [OUT_DIR]
#
#     OUT_DIR   Parent directory for the two subdirectories above. Optional;
#               defaults to this script's own directory. Only its owner can
#               write there -- if you have read/execute access only, pass a
#               directory of your own, e.g.
#                   ./make_paper_plots.sh $HOME/glowup_paper_plots
#               Nothing else is configurable; the script refuses extra
#               arguments so a stray flag cannot silently change the figures.
#
# Weizmann cluster only: every input path is absolute under /storage/agrp and
# the environment is the shared conda env, activated below. Budget roughly
# 10-20 minutes per sample on a login node, most of it in the PUPPI baseline;
# the analysis fans out one worker process per shard (10 shards each).

set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "usage: $(basename "$0") [OUT_DIR]" >&2
    exit 2
fi

# The --h5 paths below are relative to this script's directory, so run from
# there regardless of where the caller invoked this from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="${1:-$SCRIPT_DIR}"

# Fail early and say why: the default output directory is inside the source
# tree, which a read/execute-only user cannot write to. Better a clear message
# now than a permission error twenty minutes into the analysis.
for sub in ttbar dihiggs; do
    if ! mkdir -p "$OUT_DIR/$sub" 2>/dev/null || [ ! -w "$OUT_DIR/$sub" ]; then
        echo "error: cannot write to output directory '$OUT_DIR/$sub'" >&2
        echo "       pass one you own, e.g.: $(basename "$0") \$HOME/glowup_paper_plots" >&2
        exit 1
    fi
done

# Shared cluster environment. `set +u` around the activation because conda's
# activate script dereferences unset variables (PS1) and would abort under -u.
set +u
source /usr/wipp/conda/24.5.0u/bin/activate common
set -u

# ---------------------------------------------------------------------------
# Arguments
#
#   --h5           The 10 prediction shards (part000..part009, 1k events each)
#                  written by run_forward_pass for the epoch-28 fine-tuned
#                  checkpoint. A directory is expanded to the shards inside it.
#                  Note the di-Higgs shards are *named* ggf_overlay_finetune
#                  inside a directory named dihiggs_overlay_finetune -- ggF and
#                  di-Higgs are the same gg->HH process here.
#
#   --parquet-dir  Source of the truth-assisted PUPPI baseline, and the one
#                  argument besides the paths that differs between the two
#                  samples. It has to be the same all-vertices "paper" dataset
#                  the H5 was generated from: PUPPI is looked up per event_id,
#                  and both samples cover ids 0-9999, so pointing this at the
#                  wrong sample yields a wrong PUPPI curve *without* any error.
#                  PUPPI is otherwise required -- if the events are genuinely
#                  absent the run aborts rather than quietly dropping PUPPI
#                  from the comparison.
#
#   --out-dir      Figures, plus the resume cache in <out-dir>/state/.
#
# Deliberately left at their defaults, which are the paper's values:
#
#   --best-json    The Optuna-tuned PUPPI hyperparameters of App. A, Table 5
#                  (puppi_charged_subtract_v2_antikt_R04_best.json). Note the
#                  script does not insist on this file: if it is missing, PUPPI
#                  silently runs with untuned CMS defaults and the baseline
#                  becomes weaker than the published one. Keep it in place.
#   --jet-R 0.4, --jet-algorithm antikt, --min-pt 10.0, --min-constituents 3
#                  The anti-kT R=0.4 jet definition of Sec. 6.3.
#   --dpi 300      Resolution of the rasterised panels embedded in the PDFs.
#   per-class      On by default. Do NOT pass --no-per-class: the per-class
#                  aggregates are what Fig. 4 (class residuals) and App. C
#                  (efficiency / fake rate / target pT spectrum) are built from.
#
# --force is NOT passed. Each shard's aggregates are cached to
# <out-dir>/state/<shard>.pkl and reused on a re-run, so:
#   * a fresh output directory computes everything from scratch anyway;
#   * re-running after editing only plotting code re-renders in seconds;
#   * --force is needed only when the *analysis* changes -- new or changed bin
#     edges, jet configuration, or per-shard aggregates -- since a stale cache
#     would otherwise be merged with the new code. Add it by hand in that case.
#
# The cache is keyed by shard filename only, which is why each sample gets its
# own output directory here: a shared one accumulates both samples' pickles.
# Harmless (only the shards --h5 resolves to are merged) but confusing.
# ---------------------------------------------------------------------------

CKPTS="logs/odd_pflow_reco_20260620-T125301/ckpts"
DATA="/storage/agrp/barakma/PileupODD/data"

echo "=== ttbar mu=200 (in-distribution) -> $OUT_DIR/ttbar ==="
python run_paper_performance_plots.py \
    --h5 "$CKPTS/epoch=028-val_loss=10.15650__test_ttbar_pu200_overlay_finetune_paper" \
    --parquet-dir "$DATA/ttbar_pu200_all_vertices_paper" \
    --out-dir "$OUT_DIR/ttbar"

echo "=== gg->HH di-Higgs mu=200 (out-of-distribution) -> $OUT_DIR/dihiggs ==="
python run_paper_performance_plots.py \
    --h5 "$CKPTS/epoch=028-val_loss=10.15650__test_ttbar_pu200_dihiggs_overlay_finetune_paper" \
    --parquet-dir "$DATA/dihiggs_pu200_all_vertices_paper" \
    --out-dir "$OUT_DIR/dihiggs"

echo "=== done: 14 figures per sample in $OUT_DIR/{ttbar,dihiggs} ==="
