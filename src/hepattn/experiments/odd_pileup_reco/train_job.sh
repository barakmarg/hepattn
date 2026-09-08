# train_job.sh — 4-GPU DDP training job body (submitted by start.sh). Builds the mmap dataset on the node (into the
# node-local NVMe $TMPDIR) straight from the Lustre parquet, then trains.
# Comet key: from ~/.comet.env (see comet_env.sh). Overridden by a key passed
# as an argument, or by COMET_KEY_OVERRIDE, which start.sh forwards via qsub -v.
source /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/comet_env.sh --require ${COMET_KEY_OVERRIDE:-} "$@" || exit 1
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

# --- Threading / fd / allocator hygiene -------------------------------------
# Pin BLAS/OMP to 1 thread per process: with many DataLoader workers x 4 ranks
# the numpy incidence-matrix build would otherwise oversubscribe the cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# mmap opens one fd per shard per worker; raise the limit well above 980.
ulimit -n 65536
export IOTHROTTLE_LIMIT=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Build the mmap dataset ON THE NODE, directly into $TMPDIR/mmap ----------
# Reads parquet from Lustre once, writes the packed .pt shards to local NVMe.
# Rebuilt every job since $TMPDIR is wiped at job end (~20-30 min in parallel).
PARQUET=/storage/agrp/barakma/PileupODD/data/ttbar_pu0_overlay_pu200
DST_MMAP=$TMPDIR/mmap
mkdir -p "$DST_MMAP"
df -h "$TMPDIR"

# Split shards 1..1000 into NJOBS contiguous ranges, one prep_mmap process each
# (shards are independent). Cap polars threads so NJOBS procs don't oversubscribe.
NJOBS=10
export POLARS_MAX_THREADS=4
TOTAL=1000
CHUNK=$(( (TOTAL + NJOBS - 1) / NJOBS ))
echo "Building mmap shards on node: $NJOBS parallel jobs x chunk $CHUNK ..."
for i in $(seq 0 $((NJOBS-1))); do
  LO=$(( i*CHUNK + 1 ))
  HI=$(( (i+1)*CHUNK ))
  [ "$LO" -gt "$TOTAL" ] && break
  [ "$HI" -gt "$TOTAL" ] && HI=$TOTAL
  python prep_mmap.py --config configs/base.yaml \
      --parquet-dir "$PARQUET" --out-dir "$DST_MMAP" --shards ${LO}-${HI} \
      > "$DST_MMAP/prep_${LO}_${HI}.log" 2>&1 &
done
wait
# Merge the per-shard metas into the single index the dataset reads at startup.
python prep_mmap.py --out-dir "$DST_MMAP" --index-only
echo "Built $(find "$DST_MMAP" -name 'shard_*.pt' ! -name '*.meta.pt' | wc -l) shards in $DST_MMAP."

# --- Train (4-GPU DDP, mmap backend) ----------------------------------------
# FRESH run warm-started from a stable checkpoint: --model.init_from_ckpt copies only
# the weights, then training begins at epoch 0 with a new optimizer + full LR schedule
# (and teacher_forcing:false from the config). This is NOT a resume — use --ckpt_path
# for that instead.
CKPT=/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260616-T142037/ckpts/epoch=030-val_loss=13.34838.ckpt
python main.py fit --config configs/base.yaml \
    --data.backend mmap \
    --data.unify_path "$DST_MMAP" \
    --model.init_from_ckpt "$CKPT"
popd
