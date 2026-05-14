pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond

qsub -o output.log -e error.log -q N -N cond-pu-200 -l walltime=72:00:00,mem=120gb,ncpus=8,ngpus=1,io=0.1,gputype=A6000 brk.sh

popd