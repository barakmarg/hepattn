# run_eval.sh

#odd_pflow_v1_clic_arch_20260114-T151040 - overtrainned, epoch=999-val_loss=1.05913.ckpt
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
export M_ODD=/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260412-T163313
export CKPT_PATH=${M_ODD}/ckpts/epoch=028-val_loss=10.78560.ckpt
 # specify epoch and step
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IOTHROTTLE_LIMIT=100

python main.py test --config configs/base.yaml --ckpt_path $CKPT_PATH 
popd
#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260412-T163313/ckpts/epoch=028-val_loss=10.78560.ckpt
