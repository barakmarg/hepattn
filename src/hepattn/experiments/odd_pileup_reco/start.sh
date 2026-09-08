pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

# 4-GPU DDP: ncpus ~10/GPU for the DataLoader workers (num_workers=8/rank),
# mem generous so the OS page cache can hold a large mmap working set, io high
# to speed the one-time Lustre->NVMe staging in train_job.sh.
# The job reads the Comet key from ~/.comet.env on the compute node (see
# comet_env.sh). Pass a key as the first argument to override; it travels via
# qsub -v, so it is visible in `qstat -f` for the life of the job.
COMET_KEY_ARG="${1:-}"
QSUB_V=()
[ -n "$COMET_KEY_ARG" ] && QSUB_V=(-v "COMET_KEY_OVERRIDE=${COMET_KEY_ARG}")

qsub "${QSUB_V[@]}" -o output.log -e error.log -q N -N pflow-4gpu-big-finetune \
  -l walltime=72:00:00,mem=350gb,ncpus=32,ngpus=4,io=0.1,gputype=A6000 train_job.sh

popd
