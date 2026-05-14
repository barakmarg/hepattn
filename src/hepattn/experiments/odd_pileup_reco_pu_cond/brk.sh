# brk.sh
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond
export IOTHROTTLE_LIMIT=100
 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#python main.py fit --config configs/base.yaml --ckpt_path /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond/logs/odd_pflow_reco_pu_cond_20260507-T092044/ckpts/epoch=031-val_loss=12.58707.ckpt  #--data.num_workers 0
python main.py fit --config configs/base.yaml 
popd
