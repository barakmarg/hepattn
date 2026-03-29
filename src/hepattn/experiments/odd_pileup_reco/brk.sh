# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py fit --config configs/base.yaml  #--data.num_workers 0
popd
