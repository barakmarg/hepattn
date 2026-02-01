pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd

qsub -o output.log -e error.log -q N -N IndexCHange -l walltime=72:00:00,mem=48gb,ncpus=16,ngpus=1,io=10,gputype=A6000 run_eval.sh

popd