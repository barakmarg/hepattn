pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

qsub -o output.log -e error.log -q N -N CrossAttentionNoMaxFNAgressive -l walltime=72:00:00,mem=80gb,ncpus=8,ngpus=1,io=10,gputype=A5000 brk.sh

popd