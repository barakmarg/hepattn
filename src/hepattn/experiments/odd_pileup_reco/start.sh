pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

# 4-GPU DDP: ncpus ~10/GPU for the DataLoader workers (num_workers=8/rank),
# mem generous so the OS page cache can hold a large mmap working set, io high
# to speed the one-time Lustre->NVMe staging in brk.sh.
qsub -o output.log -e error.log -q N -N pflow-4gpu \
  -l walltime=72:00:00,mem=256gb,ncpus=32,ngpus=4,io=0.1,gputype=A6000 brk.sh

popd
