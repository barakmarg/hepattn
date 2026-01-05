# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd
python main.py fit --config configs/base.yaml --trainer.fast_dev_run 3  --data.num_workers 0 --data.batch_size 32
popd
