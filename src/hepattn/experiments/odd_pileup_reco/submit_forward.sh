#!/bin/bash
# submit_forward.sh — run run_forward_pass.py on the ggf (dihiggs all-vertices
# paper) sample for each checkpoint below. One single-GPU job per checkpoint;
# the checkpoint is passed into run_forward_pass.py via the FWD_CKPT env var.
# Output H5 shards land in a folder next to each checkpoint, so the runs do not
# collide. DATA_DIR is already set to the ggf sample inside run_forward_pass.py.
set -euo pipefail

WORKDIR=/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco

CKPTS=(
  "${WORKDIR}/logs/odd_pflow_reco_20260615-T105829/ckpts/epoch=072-val_loss=12.74524.ckpt"
  "${WORKDIR}/logs/odd_pflow_reco_20260615-T233516/ckpts/epoch=083-val_loss=11.27779.ckpt"
)

i=0
for CKPT in "${CKPTS[@]}"; do
  i=$((i + 1))
  echo "Submitting forward pass ${i}: ${CKPT}"
  qsub -q N -N "fwd-ggf-${i}" \
    -o "${WORKDIR}/fwd_ggf_${i}.out.log" \
    -e "${WORKDIR}/fwd_ggf_${i}.err.log" \
    -l walltime=07:00:00,mem=48gb,ncpus=16,ngpus=1,io=1,gputype=A6000 \
    <<EOF
export COMET_API_KEY=rw9qVay7dAEGfWtM0hgakSmIh
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
cd ${WORKDIR}
export IOTHROTTLE_LIMIT=100
export FWD_CKPT="${CKPT}"
echo "Forward pass on FWD_CKPT=\$FWD_CKPT"
python run_forward_pass.py
EOF
done

echo "Submitted ${i} job(s). Check status with: qstat -u \$USER"
