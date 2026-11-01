#!/usr/bin/env bash
set -uo pipefail
[ -n "${SKIP_SETUP:-}" ] || bash /content/adapt_gpu_setup.sh || exit 1
cd /content
export PYTHONPATH=/content ADAPT_CKPT=/content/ckpt/uma-s-1p2-aqcat25-r2.pt ADAPT_REFS=/content/data/aq_elem_refs_r2.json \
    ADAPT_REPLAY=0 ADAPT_WARMUP_STEPS=4987
echo "STAGE lora2"
python adapt_launch.py B lora2 40000 8 evaluate_every_n_steps=2000 checkpoint_every_n_steps=5000 \
    weight_decay=0 runner.train_eval_unit.model._target_=adapt_lora.initialize_task_lora_model \
    runner.train_eval_unit.model.rank=8 runner.train_eval_unit.model.alpha=8.0 || exit 1
for i in $(seq 1 4320); do grep -q all_done /content/lora2.log && break; sleep 10; done
echo "STAGE verify"
python adapt_verify_frozen.py $ADAPT_CKPT $(ls /content/runs/lora2/*/checkpoints/final/inference_ckpt.pt) aqcat25 > /content/lora2_v1.json
echo "V1 exit $?"
echo "STAGE l2_complete"
