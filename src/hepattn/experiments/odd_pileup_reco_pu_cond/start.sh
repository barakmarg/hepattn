pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond

qsub -o output.log -e error.log -q N -N 256dim-ris90keval -l walltime=72:00:00,mem=80gb,ncpus=8,ngpus=1,io=0.9,gputype=A6000 run_eval.sh

popd