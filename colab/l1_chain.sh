#!/usr/bin/env bash
set -uo pipefail
bash /content/adapt_gpu_setup.sh || exit 1
cd /content
export PYTHONPATH=/content ADAPT_CKPT=/content/ckpt/uma-s-1p2-aqcat25-r2.pt ADAPT_REFS=/content/data/aq_elem_refs_r2.json ADAPT_REPLAY=1
echo "STAGE lora1"
python adapt_launch.py B lora1 110000 8 runner.max_steps=4000 evaluate_every_n_steps=1000 checkpoint_every_n_steps=4000 \
    weight_decay=0 runner.train_eval_unit.model._target_=adapt_lora.initialize_task_lora_model \
    runner.train_eval_unit.model.rank=8 runner.train_eval_unit.model.alpha=8.0 || exit 1
for i in $(seq 1 720); do grep -q all_done /content/lora1.log && break; sleep 10; done
echo "STAGE verify"
python adapt_verify_frozen.py $ADAPT_CKPT $(ls /content/runs/lora1/*/checkpoints/final/inference_ckpt.pt) aqcat25 > /content/lora1_v1.json
echo "V1 exit $?"
echo "STAGE l1_complete"
