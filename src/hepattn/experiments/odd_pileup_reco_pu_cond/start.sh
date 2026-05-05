pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond

qsub -o output.log -e error.log -q N -N cond-pu-small -l walltime=72:00:00,mem=50gb,ncpus=8,ngpus=1,io=0.9,gputype=A5000 brk.sh

popd