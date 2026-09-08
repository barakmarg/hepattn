# run_eval.sh

#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260421-T104425/ckpts/epoch=099-val_loss=13.99981.ckpt
# Comet key: from ~/.comet.env (see comet_env.sh), or pass one as an argument.
source /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/comet_env.sh --require "$@" || exit 1
export M_ODD=/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260421-T104425
export CKPT_PATH=${M_ODD}/ckpts/epoch=099-val_loss=13.99981.ckpt
 # specify epoch and step
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
pushd /storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IOTHROTTLE_LIMIT=100

python main.py test --config configs/base.yaml --ckpt_path $CKPT_PATH 
popd
#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260421-T104425/ckpts/epoch=099-val_loss=13.99981.ckpt
