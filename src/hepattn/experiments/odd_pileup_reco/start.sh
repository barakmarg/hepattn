pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

qsub -o output.log -e error.log -q N -N 256dim-ris90keval -l walltime=72:00:00,mem=80gb,ncpus=8,ngpus=1,io=0.1,gputype=A5000 brk.sh

popd