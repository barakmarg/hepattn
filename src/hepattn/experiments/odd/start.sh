pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd

qsub -o output.log -e error.log -q N -N IndexCHange -l walltime=72:00:00,mem=48gb,ncpus=2,ngpus=1,io=1.0,gputype=A5000 brk.sh

popd