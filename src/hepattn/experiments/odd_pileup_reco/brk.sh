# brk.sh — 4-GPU DDP training. Builds the mmap dataset on the node (into the
# node-local NVMe $TMPDIR) straight from the Lustre parquet, then trains.
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
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
python main.py fit --config configs/base.yaml \
    --data.backend mmap \
    --data.unify_path "$DST_MMAP"
popd
