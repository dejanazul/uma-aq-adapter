#!/usr/bin/env bash
set -uo pipefail
for i in $(seq 1 360); do
    grep -q 'STAGE complete' <(sed -n '/STAGE u1_replay/,$p' /content/u1_chain.log) && break; sleep 10
done
grep -q 'STAGE complete' <(sed -n '/STAGE u1_replay/,$p' /content/u1_chain.log) || { echo 'U1 not finished'; exit 1; }
echo "STAGE opt1"
cd /content
export PYTHONPATH=/content ADAPT_CKPT=/content/ckpt/uma-s-1p2-aqcat25-r2.pt ADAPT_REFS=/content/data/aq_elem_refs_r2.json ADAPT_REPLAY=1
python adapt_launch.py B opt1 110000 8 runner.max_steps=4000 evaluate_every_n_steps=1000 checkpoint_every_n_steps=4000 \
    weight_decay=0 runner.train_eval_unit.model._target_=adapt_freeze.initialize_task_only_model || exit 1
for i in $(seq 1 720); do grep -q all_done /content/opt1.log && break; sleep 10; done
echo "STAGE verify"
python adapt_verify_frozen.py $ADAPT_CKPT $(ls /content/runs/opt1/*/checkpoints/final/inference_ckpt.pt) aqcat25 > /content/opt1_v1.json
echo "V1 exit $?"
echo "STAGE o1_complete"
