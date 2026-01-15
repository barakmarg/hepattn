# run_eval.sh

#odd_pflow_v1_clic_arch_20260114-T151040 - overtrainned, epoch=999-val_loss=1.05913.ckpt
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export M_ODD=/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd/logs/odd_pflow_v1_clic_arch_20260114-T151040
export CKPT_PATH=${M_ODD}/ckpts/epoch=999-val_loss=1.05913.ckpt # specify epoch and step
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py test --config configs/base_overfit.yaml --ckpt_path $CKPT_PATH --data.num_workers 0
popd
#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd/logs/odd_pflow_v1_clic_arch_20260112-T143417/ckpts/epoch=199-val_loss=4.48795__test.h5
